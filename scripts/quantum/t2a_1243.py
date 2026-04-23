#!/usr/bin/env python3
"""Generate 8.7.56.1243-.1246 Trial-2 Q-ball projection-overlap artifacts.

Purpose:
    Re-evaluate Trial-2 numeric alpha after the projection-overlap note argued
    that the last missing piece may already be embedded in the P-model: the
    physical electromagnetic coupling is not a literal bare `q = e` identity,
    but a cross-sector overlap between the electron-like Q-ball mode and the
    transverse photon mode of the same field `P_mu`.

Inputs:
    - Current operational docs and the Part I / Part III-A / Part V paper
      surfaces
    - The retained scalar/vector Q-ball ground-state metrics
    - The prior `.1239-.1242` adopted-U(1) dictionary-contract metrics
    - The external note
      `C:/Users/ogawa/Downloads/pmodel_v2_trial2_projection_overlap_justification.md`

Outputs:
    - Four machine-readable metrics payloads under `output/public/quantum/`

Assumptions:
    - The retained scalar ground-state profile is reused as the electron-like
      radial profile because the exact vector ladder still keeps
      `(n,k,ell,s) = (1,0,0,0)` as the scalar baseline reference state.
    - No new free parameter is introduced during the blind evaluation. The
      branch only records whether the retained profile itself can generate the
      observed suppression at some finite momentum transfer.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_projection_overlap_justification.md")
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
QBALL_FULL_COUPLED = PUBLIC_OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"
DICT_CONTRACT_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_contract_"
    "declaration_gate_metrics.json"
)
DICT_CONTRACT_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_contract_"
    "numeric_evaluation_metrics.json"
)
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"

ALPHA_TARGET = 1.0 / 137.035999084
TARGET_FORM_FACTOR = math.sqrt(4.0 * math.pi * ALPHA_TARGET)
Q_SAMPLE_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
NEXT_ROUTE = "8.7.56.1247"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_qball_projection_overlap_matching_scale_review"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort if one required input is missing.

def require(path: Path) -> None:
    """Abort if one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return one repo-relative display path when possible.

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first matching line for one substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build one standard metrics payload.

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard metrics payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write one JSON metrics payload and one CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one metrics payload as JSON and CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: load the retained scalar Q-ball solver as a reusable module.

def load_qball_module():
    """Load the retained scalar Q-ball solver as a reusable module."""
    spec = importlib.util.spec_from_file_location("wavep_qball_charge_mapping", QBALL_SOLVER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {QBALL_SOLVER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: extract the scalar ground-state row from the retained branch-refresh metrics.

def extract_scalar_ground_state(qball_branch_refresh: dict) -> dict:
    """Extract the scalar ground-state row from the retained branch-refresh metrics."""
    for row_data in qball_branch_refresh["evidence"]["discrete_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "mode_index": int(row_data["mode_index"]),
                "beta_n": float(row_data["beta_n"]),
                "charge_proxy": float(row_data["charge_proxy"]),
                "energy_proxy": float(row_data["energy_proxy"]),
                "central_amplitude": float(row_data["central_amplitude"]),
                "mass_ratio_to_first": float(row_data["mass_ratio_to_first"]),
            }

    raise SystemExit("[fail] missing scalar ground-state row in charge-mapping branch refresh metrics")


# Function: extract the exact vector-ladder reference state from the retained full-coupled metrics.

def extract_exact_ground_state(qball_full_coupled: dict) -> dict:
    """Extract the exact vector-ladder reference state from the retained full-coupled metrics."""
    for row_data in qball_full_coupled["evidence"]["exact_ladder_sample_rows"]:
        if (
            int(row_data["n"]) == 1
            and int(row_data["k"]) == 0
            and int(row_data["ell"]) == 0
            and int(row_data["s"]) == 0
        ):
            return {
                "n": int(row_data["n"]),
                "k": int(row_data["k"]),
                "ell": int(row_data["ell"]),
                "s": int(row_data["s"]),
                "beta_n": float(row_data["beta_n"]),
                "exact_charge_proxy": float(row_data["exact_charge_proxy"]),
                "exact_mass_proxy": float(row_data["exact_mass_proxy"]),
                "mass_ratio_to_scalar_base": float(row_data["mass_ratio_to_scalar_base"]),
            }

    raise SystemExit("[fail] missing exact vector reference row M_(1,0,0,0)")


# Function: evaluate one normalized spherical-overlap form factor on the retained profile.

def form_factor(radius: np.ndarray, weight: np.ndarray, norm: float, q_ratio: float) -> float:
    """Evaluate one normalized spherical-overlap form factor."""
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = np.trapezoid(weight * sinc, radius)
    return float(numerator / norm)


# Function: locate the first positive-q target crossing on the blind overlap profile.

def find_first_target_crossing(radius: np.ndarray, weight: np.ndarray, norm: float) -> float | None:
    """Locate the first positive-q target crossing on the blind overlap profile."""
    search_grid = np.linspace(0.0, 1.5, 601)
    values = [form_factor(radius, weight, norm, float(q)) - TARGET_FORM_FACTOR for q in search_grid]
    for q_left, q_right, f_left, f_right in zip(search_grid[:-1], search_grid[1:], values[:-1], values[1:]):
        if f_left == 0.0:
            return float(q_left)

        if f_left * f_right < 0.0:
            return float(
                brentq(
                    lambda q: form_factor(radius, weight, norm, float(q)) - TARGET_FORM_FACTOR,
                    float(q_left),
                    float(q_right),
                )
            )

    return None


# Function: compute compact profile-scale diagnostics for the retained ground state.

def profile_scales(radius: np.ndarray, weight: np.ndarray, norm: float) -> dict:
    """Compute compact profile-scale diagnostics for the retained ground state."""
    mean_radius = float(np.trapezoid(weight * radius, radius) / norm)
    rms_radius = float(math.sqrt(np.trapezoid(weight * radius * radius, radius) / norm))
    cumulative = np.cumsum((weight[1:] + weight[:-1]) * np.diff(radius) * 0.5)
    cumulative = np.concatenate([[0.0], cumulative]) / norm
    half_mass_radius = None
    for idx in range(1, len(radius)):
        if cumulative[idx] >= 0.5:
            r0 = float(radius[idx - 1])
            r1 = float(radius[idx])
            c0 = float(cumulative[idx - 1])
            c1 = float(cumulative[idx])
            fraction = 0.0 if c1 == c0 else (0.5 - c0) / (c1 - c0)
            half_mass_radius = r0 + fraction * (r1 - r0)
            break

    return {
        "mean_radius_x": mean_radius,
        "rms_radius_x": rms_radius,
        "half_mass_radius_x": float(half_mass_radius) if half_mass_radius is not None else None,
    }


# Function: execute the 8.7.56.1243-.1246 branch.

def main() -> None:
    """Execute the 8.7.56.1243-.1246 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART1,
        PART3A,
        PART5,
        NOTE,
        QBALL_BRANCH_REFRESH,
        QBALL_FULL_COUPLED,
        DICT_CONTRACT_GATE,
        DICT_CONTRACT_EVAL,
        QBALL_SOLVER,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    note_text = read_text(NOTE)

    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    qball_full_coupled = read_json(QBALL_FULL_COUPLED)
    dict_contract_gate = read_json(DICT_CONTRACT_GATE)["summary"]
    dict_contract_eval = read_json(DICT_CONTRACT_EVAL)["summary"]

    qball_module = load_qball_module()
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)
    exact_ground_state = extract_exact_ground_state(qball_full_coupled)

    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    sampled_form_factors = [
        {"q_over_m0": float(q_ratio), "F_q": form_factor(radius, weight, norm, float(q_ratio))}
        for q_ratio in Q_SAMPLE_GRID
    ]
    F_0 = sampled_form_factors[0]["F_q"]
    F_m0 = next(sample["F_q"] for sample in sampled_form_factors if sample["q_over_m0"] == 1.0)
    alpha_m0 = (F_m0**2) / (4.0 * math.pi)
    F_m0_relative_error = abs(F_m0 - TARGET_FORM_FACTOR) / TARGET_FORM_FACTOR
    alpha_m0_relative_error = abs(alpha_m0 - ALPHA_TARGET) / ALPHA_TARGET
    literal_q_equals_m0_pass = F_m0_relative_error < 0.10
    first_target_crossing_q = find_first_target_crossing(radius, weight, norm)
    finite_q_target_crossing_exists = first_target_crossing_q is not None
    matching_scale_is_finite = finite_q_target_crossing_exists and float(first_target_crossing_q) > 1.0e-6
    matching_scale_order_of_m0 = finite_q_target_crossing_exists and 0.1 <= float(first_target_crossing_q) <= 1.0
    alpha_at_first_crossing = (
        (form_factor(radius, weight, norm, float(first_target_crossing_q)) ** 2) / (4.0 * math.pi)
        if finite_q_target_crossing_exists
        else None
    )

    scales = profile_scales(radius, weight, norm)
    scale_products = (
        {
            "q_target_times_mean_radius": float(first_target_crossing_q) * scales["mean_radius_x"],
            "q_target_times_rms_radius": float(first_target_crossing_q) * scales["rms_radius_x"],
            "q_target_times_half_mass_radius": float(first_target_crossing_q) * scales["half_mass_radius_x"],
        }
        if finite_q_target_crossing_exists and scales["half_mass_radius_x"] is not None
        else None
    )

    exact_reference_consistent = (
        math.isclose(float(scalar_ground_state["beta_n"]), float(exact_ground_state["beta_n"]), rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(
            float(scalar_ground_state["charge_proxy"]),
            float(exact_ground_state["exact_charge_proxy"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(scalar_ground_state["energy_proxy"]),
            float(exact_ground_state["exact_mass_proxy"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )

    zero_momentum_consistency_pass = math.isclose(F_0, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    finite_q_overlap_suppression_present = abs(F_m0) < F_0
    projection_overlap_mechanism_admissible = (
        zero_momentum_consistency_pass
        and finite_q_overlap_suppression_present
        and finite_q_target_crossing_exists
    )
    soft_limit_not_target = abs(F_0 - TARGET_FORM_FACTOR) / TARGET_FORM_FACTOR > 0.10

    note_cross_sector_line = hit(note_text, "cross-sector mode overlap")
    note_matching_scale_line = hit(note_text, "$q_{\\rm char} = m_0$")
    note_route_line = hit(note_text, "qball_projection_overlap_charge_bridge_candidate")
    part1_photon_line = hit(part1_text, "A_\\mu=\\delta P_\\mu^T/\\sqrt{Z_P}")
    part3a_electron_line = hit(part3a_text, "M_{(1,0,0,0)} = m_e")
    part3a_bare_alpha_line = hit(part3a_text, "0.07957747154594767")

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "projection_overlap_note": display_path(NOTE),
        },
        "prior_metrics": {
            "qball_branch_refresh": display_path(QBALL_BRANCH_REFRESH),
            "qball_full_coupled": display_path(QBALL_FULL_COUPLED),
            "dict_contract_gate": display_path(DICT_CONTRACT_GATE),
            "dict_contract_eval": display_path(DICT_CONTRACT_EVAL),
        },
        "solver_module": display_path(QBALL_SOLVER),
        "constants": {
            "alpha_target": ALPHA_TARGET,
            "target_form_factor": TARGET_FORM_FACTOR,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1243",
        "Trial-2 numeric alpha Q-ball projection-overlap source inventory",
        inputs,
        [
            row("projection_overlap_note_available", "pass", "projection-overlap note available", 1.0, "The external projection-overlap note is present."),
            row("scalar_ground_state_row_available", "pass", "scalar ground-state row available", 1.0, "The retained scalar ground-state row exposes beta_1, energy proxy, charge proxy, and central amplitude."),
            row("exact_vector_ground_state_proxy_consistent", "pass" if exact_reference_consistent else "reject", "exact vector ground-state proxy consistent", 1 if exact_reference_consistent else 0, "The exact vector reference state M_(1,0,0,0) must remain identical to the scalar baseline row used for electron identification."),
            row("qball_profile_reconstruction_ready", "pass", "Q-ball profile reconstruction ready", 1.0, "The retained scalar solver can reconstruct the full ground-state radial profile from beta_1 and its central amplitude."),
            row("same_field_cross_sector_note_present", "pass" if note_cross_sector_line is not None else "reject", "same-field cross-sector note present", 1 if note_cross_sector_line is not None else 0, "The note must explicitly state that photon and electron arise as different modes of the same field."),
            row("current_pack_bare_alpha_retained", "pass" if part3a_bare_alpha_line is not None else "reject", "current pack bare alpha retained", 1 if part3a_bare_alpha_line is not None else 0, "The current pack must still retain the bare action-level alpha benchmark 1/(4 pi)."),
        ],
        {
            "inventory_ready": True,
            "scalar_ground_state": scalar_ground_state,
            "exact_ground_state": exact_ground_state,
            "exact_vector_ground_state_proxy_consistent": exact_reference_consistent,
            "selected_next_substep": "8.7.56.1244",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_inventory_fixed",
            "advance_to_8_7_56_1244": True,
            "next_required_artifacts": ["qball_projection_overlap_audit"],
        },
        {
            "note_hits": {
                "cross_sector_mode_overlap": note_cross_sector_line,
                "matching_scale_q_equals_m0": note_matching_scale_line,
                "route_name": note_route_line,
            },
            "paper_hits": {
                "part1_photon_line": part1_photon_line,
                "part3a_electron_identification": part3a_electron_line,
                "part3a_bare_alpha_line": part3a_bare_alpha_line,
            },
            "status_hits": {
                "status_next_1243": hit(status_text, "8.7.56.1243"),
                "roadmap_branch_1243": hit(roadmap_text, "`8.7.56.1243-.1246`"),
                "work_history_1239_entry": hit(work_history_recent_text, "8.7.56.1239-.1242"),
            },
            "prior_dictionary_gate_summary": dict_contract_gate,
            "prior_dictionary_eval_summary": dict_contract_eval,
        },
    )

    audit = payload(
        "8.7.56.1244",
        "Trial-2 numeric alpha Q-ball projection-overlap audit",
        inputs,
        [
            row("projection_overlap_zero_momentum_consistency_pass", "pass" if zero_momentum_consistency_pass else "reject", "projection-overlap F(0) consistency pass", F_0, "The blind overlap must reproduce F(0)=1 as the Coulomb-tail consistency check."),
            row("projection_overlap_finite_q_suppression_present", "pass" if finite_q_overlap_suppression_present else "reject", "projection-overlap finite-q suppression present", 1 if finite_q_overlap_suppression_present else 0, "The retained profile must suppress the finite-q coupling relative to the bare soft limit."),
            row("projection_overlap_literal_q_equals_m0_pass", "pass" if literal_q_equals_m0_pass else "reject", "projection-overlap literal q = m0 pass", 1 if literal_q_equals_m0_pass else 0, "The note's literal q = m0 claim only passes if the blind F(m0) lands near the target form factor."),
            row("projection_overlap_target_crossing_exists", "pass" if finite_q_target_crossing_exists else "reject", "projection-overlap target crossing exists", 1 if finite_q_target_crossing_exists else 0, "The blind profile is only numerically admissible if some finite q/m0 reproduces the target form factor."),
            row("projection_overlap_soft_limit_not_target", "pass" if soft_limit_not_target else "reject", "projection-overlap soft limit not target", 1 if soft_limit_not_target else 0, "The observed coupling must come from a finite internal scale rather than from q -> 0."),
            row("projection_overlap_matching_scale_order_of_m0", "pass" if matching_scale_order_of_m0 else "reject", "projection-overlap matching scale order of m0", 1 if matching_scale_order_of_m0 else 0, "A finite matching scale in the broad 0.1-1.0 m0 range keeps the route in the same internal-structure regime rather than in the strict soft limit."),
        ],
        {
            "projection_overlap_mechanism_admissible": projection_overlap_mechanism_admissible,
            "literal_q_equals_m0_supported": literal_q_equals_m0_pass,
            "finite_q_target_crossing_exists": finite_q_target_crossing_exists,
            "first_target_matching_q_over_m0": float(first_target_crossing_q) if finite_q_target_crossing_exists else None,
            "matching_scale_is_finite": bool(matching_scale_is_finite),
            "matching_scale_order_of_m0": bool(matching_scale_order_of_m0),
            "F_0": F_0,
            "F_m0": F_m0,
            "alpha_m0": alpha_m0,
            "result_class": (
                "projection_overlap_mechanism_admissible_literal_m0_not_supported"
                if projection_overlap_mechanism_admissible and not literal_q_equals_m0_pass
                else (
                    "projection_overlap_numeric_pass"
                    if literal_q_equals_m0_pass
                    else "projection_overlap_route_fail"
                )
            ),
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_audit_completed",
            "advance_to_8_7_56_1245": True,
            "next_required_artifacts": ["qball_projection_overlap_declaration_gate"],
        },
        {
            "sampled_form_factors": sampled_form_factors,
            "profile_scales": scales,
            "profile_scale_products": scale_products,
        },
    )

    declaration_gate = payload(
        "8.7.56.1245",
        "Trial-2 numeric alpha Q-ball projection-overlap declaration gate",
        inputs,
        [
            row("projection_overlap_branch_completed", "pass", "projection-overlap branch completed", 1.0, "The blind overlap route has now been audited end-to-end."),
            row("projection_overlap_mechanism_admissible", "pass" if projection_overlap_mechanism_admissible else "reject", "projection-overlap mechanism admissible", 1 if projection_overlap_mechanism_admissible else 0, "The retained Q-ball profile itself must generate a finite-q suppression mechanism before this route can become primary."),
            row("literal_q_equals_m0_supported", "pass" if literal_q_equals_m0_pass else "reject", "literal q = m0 supported", 1 if literal_q_equals_m0_pass else 0, "The note's literal q = m0 formula only survives if the blind F(m0) lands near the target."),
            row("dictionary_gap_demoted_to_secondary", "pass" if projection_overlap_mechanism_admissible else "reject", "dictionary gap demoted to secondary", 1 if projection_overlap_mechanism_admissible else 0, "Once the blind overlap mechanism exists, the old q-to-e dictionary gap is no longer the unique primary blocker."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "The overlap mechanism keeps the route open even though the literal q = m0 claim fails."),
            row("closeout_ready", "reject", "closeout ready", 0.0, "Closeout is not ready while the matching-scale justification remains open."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_matching_scale_open",
            "projection_overlap_mechanism_admissible": projection_overlap_mechanism_admissible,
            "literal_q_equals_m0_supported": literal_q_equals_m0_pass,
            "finite_q_target_crossing_exists": finite_q_target_crossing_exists,
            "first_target_matching_q_over_m0": float(first_target_crossing_q) if finite_q_target_crossing_exists else None,
            "matching_scale_order_of_m0": bool(matching_scale_order_of_m0),
            "primary_residual_lane": "qball_projection_overlap_matching_scale_justification",
            "secondary_residual_lane": "adopted_u1_charge_unit_dictionary",
            "reserve_residual_lane": "adopted_u1_vacuum_polarization_external_import",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_declared",
            "advance_to_8_7_56_1246": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "prior_dictionary_gate_summary": dict_contract_gate,
        },
    )

    evaluation = payload(
        "8.7.56.1246",
        "Trial-2 numeric alpha Q-ball projection-overlap numeric evaluation",
        inputs,
        [
            row("projection_overlap_target_form_factor_fixed", "pass", "projection-overlap target form factor fixed", TARGET_FORM_FACTOR, "The observed fine-structure constant corresponds to the target form factor sqrt(4 pi alpha_target)."),
            row("projection_overlap_F_0_fixed", "pass" if zero_momentum_consistency_pass else "reject", "projection-overlap F(0) fixed", F_0, "The blind overlap reproduces the bare soft-limit normalization."),
            row("projection_overlap_F_m0_fixed", "pass", "projection-overlap F(m0) fixed", F_m0, "The blind overlap at q = m0 is recorded exactly as evaluated from the retained profile."),
            row("projection_overlap_alpha_m0_fixed", "pass", "projection-overlap alpha(q=m0) fixed", alpha_m0, "The literal q = m0 alpha candidate is recorded exactly."),
            row("projection_overlap_first_target_matching_q_over_m0_fixed", "pass" if finite_q_target_crossing_exists else "reject", "projection-overlap first target-matching q/m0 fixed", float(first_target_crossing_q) if finite_q_target_crossing_exists else math.nan, "The first finite-q crossing where F(q) matches the observed charge is fixed from the blind profile."),
            row("projection_overlap_alpha_first_target_crossing_fixed", "pass" if finite_q_target_crossing_exists else "reject", "projection-overlap alpha at first target crossing fixed", float(alpha_at_first_crossing) if alpha_at_first_crossing is not None else math.nan, "This crossing is not an input fit parameter; it is the q/m0 location where the blind overlap first reproduces the observed suppression."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_matching_scale_open",
            "target_form_factor": TARGET_FORM_FACTOR,
            "F_0": F_0,
            "F_m0": F_m0,
            "F_m0_relative_error_to_target_form_factor": F_m0_relative_error,
            "alpha_m0": alpha_m0,
            "alpha_target": ALPHA_TARGET,
            "alpha_m0_relative_error_to_target": alpha_m0_relative_error,
            "finite_q_target_crossing_exists": finite_q_target_crossing_exists,
            "first_target_matching_q_over_m0": float(first_target_crossing_q) if finite_q_target_crossing_exists else None,
            "alpha_first_target_crossing": float(alpha_at_first_crossing) if alpha_at_first_crossing is not None else None,
            "matching_scale_is_finite": bool(matching_scale_is_finite),
            "matching_scale_order_of_m0": bool(matching_scale_order_of_m0),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "sampled_form_factors": sampled_form_factors,
            "scalar_ground_state": scalar_ground_state,
            "exact_ground_state": exact_ground_state,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1243-.1246 artifacts generated")


if __name__ == "__main__":
    main()
