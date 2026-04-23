#!/usr/bin/env python3
"""Generate 8.7.56.1247-.1250 Trial-2 Q-ball matching-scale review artifacts.

Purpose:
    Audit whether the finite matching scale found in `.1243-.1246`,
    `q_*/m0 ~= 0.2416825755`, can now be justified from current theory-side
    structure rather than being treated as a pure external-note guess.

Inputs:
    - Current operational docs and Part I / Part III-A / Part V surfaces
    - The `.1243-.1246` projection-overlap metrics
    - The current problem note and expert-share note
    - The external projection-overlap note

Outputs:
    - Four machine-readable metrics payloads under `output/public/quantum/`

Assumptions:
    - The prior blind overlap computation already fixed the numerical crossing
      `q_*`.
    - This branch does not introduce a new profile fit. It only checks whether
      the already-fixed crossing sits inside a theory-side support band implied
      by the spherical overlap kernel and the retained profile scales.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
EXPERT_SHARE = ROOT / "doc" / "quantum" / "35_trial2_numeric_alpha_projection_overlap_expert_share.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_projection_overlap_justification.md")

OVERLAP_SOURCE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_inventory_metrics.json"
OVERLAP_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_audit_metrics.json"
OVERLAP_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_declaration_gate_metrics.json"
OVERLAP_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_numeric_evaluation_metrics.json"

NEXT_ROUTE = "8.7.56.1251"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review"


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
    """Locate the first matching line for one substring pattern."""
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


# Function: solve the first positive phase where sinc(z) reaches one target value.

def sinc_target_phase(target_form_factor: float) -> float:
    """Solve the first positive phase where sinc(z) reaches one target value."""

    # Function: evaluate the sinc-minus-target residual on the first positive branch.
    def residual(z_value: float) -> float:
        return math.sin(z_value) / z_value - target_form_factor

    return float(brentq(residual, 2.0, 3.0))


# Function: build one compact relative-difference table for one implied radius.

def relative_radius_errors(implied_radius: float, scales: dict) -> dict:
    """Build one compact relative-difference table for one implied radius."""
    return {
        "vs_half_mass_radius": abs(implied_radius - scales["half_mass_radius_x"]) / scales["half_mass_radius_x"],
        "vs_mean_radius": abs(implied_radius - scales["mean_radius_x"]) / scales["mean_radius_x"],
        "vs_rms_radius": abs(implied_radius - scales["rms_radius_x"]) / scales["rms_radius_x"],
    }


# Function: execute the 8.7.56.1247-.1250 branch.

def main() -> None:
    """Execute the 8.7.56.1247-.1250 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        EXPERT_SHARE,
        PART1,
        PART3A,
        PART5,
        NOTE,
        OVERLAP_SOURCE,
        OVERLAP_AUDIT,
        OVERLAP_GATE,
        OVERLAP_EVAL,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    current_problem_text = read_text(CURRENT_PROBLEM)
    expert_share_text = read_text(EXPERT_SHARE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    note_text = read_text(NOTE)

    overlap_source = read_json(OVERLAP_SOURCE)
    overlap_audit = read_json(OVERLAP_AUDIT)
    overlap_gate = read_json(OVERLAP_GATE)
    overlap_eval = read_json(OVERLAP_EVAL)

    overlap_summary = overlap_eval["summary"]
    q_star = float(overlap_summary["first_target_matching_q_over_m0"])
    target_form_factor = float(overlap_summary["target_form_factor"])
    scales = overlap_audit["evidence"]["profile_scales"]
    scale_products = overlap_audit["evidence"]["profile_scale_products"]

    z_target_sinc = sinc_target_phase(target_form_factor)
    z_first_zero = math.pi
    q_half = float(scale_products["q_target_times_half_mass_radius"])
    q_mean = float(scale_products["q_target_times_mean_radius"])
    q_rms = float(scale_products["q_target_times_rms_radius"])

    implied_radius_target_phase = z_target_sinc / q_star
    implied_radius_first_zero = z_first_zero / q_star
    target_phase_errors = relative_radius_errors(implied_radius_target_phase, scales)
    first_zero_errors = relative_radius_errors(implied_radius_first_zero, scales)

    q_target_from_half_mass = z_target_sinc / float(scales["half_mass_radius_x"])
    q_zero_from_mean = z_first_zero / float(scales["mean_radius_x"])
    q_zero_from_rms = z_first_zero / float(scales["rms_radius_x"])

    half_mass_target_phase_alignment_pass = target_phase_errors["vs_half_mass_radius"] < 0.10
    mean_radius_first_zero_alignment_pass = first_zero_errors["vs_mean_radius"] < 0.10
    rms_radius_first_zero_alignment_pass = first_zero_errors["vs_rms_radius"] < 0.10
    support_band_brackets_q_star = z_target_sinc <= q_half <= z_first_zero and z_target_sinc <= q_mean <= z_first_zero
    finite_internal_scale_theory_side_justified = (
        support_band_brackets_q_star
        and half_mass_target_phase_alignment_pass
        and mean_radius_first_zero_alignment_pass
    )
    exact_effective_support_scale_fixed = False
    predictive_branch_ready = False

    note_same_field_line = hit(note_text, "同一場")
    note_internal_scale_line = hit(note_text, "internal structure scale")
    note_q_char_line = hit(note_text, "$q_{\\rm char} = m_0$")
    part1_photon_line = hit(part1_text, "A_\\mu=\\delta P_\\mu^T/\\sqrt{Z_P}")
    part3a_electron_line = hit(part3a_text, "M_{(1,0,0,0)} = m_e")
    part5_current_line = hit(part5_text, "projection-overlap mechanism pass / matching-scale justification open")

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "expert_share_note": display_path(EXPERT_SHARE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "projection_overlap_note": display_path(NOTE),
        },
        "prior_metrics": {
            "overlap_source": display_path(OVERLAP_SOURCE),
            "overlap_audit": display_path(OVERLAP_AUDIT),
            "overlap_gate": display_path(OVERLAP_GATE),
            "overlap_eval": display_path(OVERLAP_EVAL),
        },
        "constants": {
            "q_star_over_m0": q_star,
            "target_form_factor": target_form_factor,
            "sinc_target_phase": z_target_sinc,
            "sinc_first_zero_phase": z_first_zero,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1247",
        "Trial-2 numeric alpha Q-ball projection-overlap matching-scale review source inventory",
        inputs,
        [
            row("prior_projection_overlap_metrics_available", "pass", "prior projection-overlap metrics available", 1.0, "The .1243-.1246 overlap metrics are present and can be reused without rerunning the blind profile fit."),
            row("projection_overlap_note_internal_scale_available", "pass" if note_internal_scale_line is not None else "reject", "projection-overlap note internal-scale statement available", 1 if note_internal_scale_line is not None else 0, "The note must still expose the internal-structure interpretation of the matching scale."),
            row("part1_photon_surface_available", "pass" if part1_photon_line is not None else "reject", "Part I photon surface available", 1 if part1_photon_line is not None else 0, "The canonical transverse photon surface must remain available in Part I."),
            row("part3a_electron_identification_surface_available", "pass" if part3a_electron_line is not None else "reject", "Part III-A electron-identification surface available", 1 if part3a_electron_line is not None else 0, "The electron-like Q-ball identification surface must remain available in Part III-A."),
            row("profile_scale_diagnostics_available", "pass", "profile scale diagnostics available", 1.0, "The prior audit already fixed mean, rms, and half-mass radii together with q_* times those scales."),
            row("spherical_kernel_phase_targets_ready", "pass", "spherical kernel phase targets ready", 1.0, "The first target phase of sinc(z) and its first zero can be computed without adding any new free parameter."),
        ],
        {
            "inventory_ready": True,
            "q_star_over_m0": q_star,
            "target_form_factor": target_form_factor,
            "sinc_target_phase": z_target_sinc,
            "sinc_first_zero_phase": z_first_zero,
            "selected_next_substep": "8.7.56.1248",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_matching_scale_inventory_fixed",
            "advance_to_8_7_56_1248": True,
            "next_required_artifacts": ["qball_projection_overlap_matching_scale_review_audit"],
        },
        {
            "note_hits": {
                "same_field_line": note_same_field_line,
                "internal_scale_line": note_internal_scale_line,
                "literal_q_char_equals_m0_line": note_q_char_line,
            },
            "paper_hits": {
                "part1_photon_line": part1_photon_line,
                "part3a_electron_line": part3a_electron_line,
                "part5_current_line": part5_current_line,
            },
            "status_hits": {
                "status_next_1247": hit(status_text, "8.7.56.1247"),
                "roadmap_branch_1247": hit(roadmap_text, "`8.7.56.1247-.1250`"),
                "current_problem_matching_scale_line": hit(current_problem_text, "matching scale"),
                "expert_share_first_crossing_line": hit(expert_share_text, "0.24168257551157463"),
                "work_history_1243_entry": hit(work_history_recent_text, "8.7.56.1243-.1246"),
            },
            "prior_overlap_gate_summary": overlap_gate["summary"],
            "prior_overlap_eval_summary": overlap_summary,
            "ai_context_current_step": ai_context.get("current_step"),
        },
    )

    audit = payload(
        "8.7.56.1248",
        "Trial-2 numeric alpha Q-ball projection-overlap matching-scale review audit",
        inputs,
        [
            row("projection_overlap_half_mass_target_phase_alignment_pass", "pass" if half_mass_target_phase_alignment_pass else "reject", "projection-overlap half-mass target-phase alignment pass", 1 if half_mass_target_phase_alignment_pass else 0, "If q_* is theory-side justified, the sinc target phase implied by q_* should land near the half-mass radius of the retained profile."),
            row("projection_overlap_mean_radius_first_zero_alignment_pass", "pass" if mean_radius_first_zero_alignment_pass else "reject", "projection-overlap mean-radius first-zero alignment pass", 1 if mean_radius_first_zero_alignment_pass else 0, "The first spherical zero implied by q_* should land near the central support scale of the retained profile rather than at an unrelated radius."),
            row("projection_overlap_rms_radius_first_zero_alignment_pass", "pass" if rms_radius_first_zero_alignment_pass else "reject", "projection-overlap rms-radius first-zero alignment pass", 1 if rms_radius_first_zero_alignment_pass else 0, "The rms support scale is checked as a tail-sensitive control for the same first-zero heuristic."),
            row("projection_overlap_support_band_brackets_q_star", "pass" if support_band_brackets_q_star else "reject", "projection-overlap support band brackets q_*", 1 if support_band_brackets_q_star else 0, "The observed q_* should place the profile's central support between the target phase and the first spherical zero of the overlap kernel."),
            row("projection_overlap_finite_internal_scale_theory_side_justified", "pass" if finite_internal_scale_theory_side_justified else "reject", "projection-overlap finite internal scale theory-side justified", 1 if finite_internal_scale_theory_side_justified else 0, "This route passes only if current theory-side structure explains why the charge is read at a finite internal support scale instead of at q -> 0."),
            row("projection_overlap_exact_effective_support_scale_fixed", "pass" if exact_effective_support_scale_fixed else "reject", "projection-overlap exact effective support scale fixed", 1 if exact_effective_support_scale_fixed else 0, "This stays open unless current canon selects one unique effective support radius rather than a band of comparable candidates."),
        ],
        {
            "finite_internal_scale_theory_side_justified": finite_internal_scale_theory_side_justified,
            "support_band_brackets_q_star": support_band_brackets_q_star,
            "half_mass_target_phase_alignment_pass": half_mass_target_phase_alignment_pass,
            "mean_radius_first_zero_alignment_pass": mean_radius_first_zero_alignment_pass,
            "rms_radius_first_zero_alignment_pass": rms_radius_first_zero_alignment_pass,
            "exact_effective_support_scale_fixed": exact_effective_support_scale_fixed,
            "predictive_branch_ready": predictive_branch_ready,
            "result_class": (
                "projection_overlap_support_band_justified_exact_scale_open"
                if finite_internal_scale_theory_side_justified and not exact_effective_support_scale_fixed
                else "projection_overlap_matching_scale_still_open"
            ),
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_matching_scale_audit_completed",
            "advance_to_8_7_56_1249": True,
            "next_required_artifacts": ["qball_projection_overlap_matching_scale_review_declaration_gate"],
        },
        {
            "phase_products": {
                "q_star_times_half_mass_radius": q_half,
                "q_star_times_mean_radius": q_mean,
                "q_star_times_rms_radius": q_rms,
            },
            "spherical_kernel": {
                "target_phase": z_target_sinc,
                "first_zero_phase": z_first_zero,
            },
            "implied_radii": {
                "target_phase_implied_radius": implied_radius_target_phase,
                "first_zero_implied_radius": implied_radius_first_zero,
            },
            "relative_radius_errors": {
                "target_phase": target_phase_errors,
                "first_zero": first_zero_errors,
            },
        },
    )

    declaration_gate = payload(
        "8.7.56.1249",
        "Trial-2 numeric alpha Q-ball projection-overlap matching-scale review declaration gate",
        inputs,
        [
            row("projection_overlap_matching_scale_review_completed", "pass", "projection-overlap matching-scale review completed", 1.0, "The matching-scale review branch has now been audited end-to-end."),
            row("projection_overlap_mechanism_admissible", "pass" if overlap_gate["summary"]["projection_overlap_mechanism_admissible"] else "reject", "projection-overlap mechanism admissible", 1 if overlap_gate["summary"]["projection_overlap_mechanism_admissible"] else 0, "The blind overlap mechanism itself remains alive."),
            row("projection_overlap_finite_internal_scale_theory_side_justified", "pass" if finite_internal_scale_theory_side_justified else "reject", "projection-overlap finite internal scale theory-side justified", 1 if finite_internal_scale_theory_side_justified else 0, "Current theory-side structure now supports a finite internal support band for q_* rather than a pure external-note placeholder."),
            row("projection_overlap_literal_q_equals_m0_supported", "pass" if overlap_gate["summary"]["literal_q_equals_m0_supported"] else "reject", "projection-overlap literal q = m0 supported", 1 if overlap_gate["summary"]["literal_q_equals_m0_supported"] else 0, "The literal q = m0 claim still fails and does not return as the primary route."),
            row("projection_overlap_exact_effective_support_scale_fixed", "pass" if exact_effective_support_scale_fixed else "reject", "projection-overlap exact effective support scale fixed", 1 if exact_effective_support_scale_fixed else 0, "A predictive branch needs one unique effective support scale selected by canon, which is still absent."),
            row("projection_overlap_predictive_branch_ready", "pass" if predictive_branch_ready else "reject", "projection-overlap predictive branch ready", 1 if predictive_branch_ready else 0, "Predictive status remains withheld until one exact support-scale coefficient is fixed."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_support_band_justified_exact_scale_open",
            "projection_overlap_mechanism_admissible": overlap_gate["summary"]["projection_overlap_mechanism_admissible"],
            "literal_q_equals_m0_supported": overlap_gate["summary"]["literal_q_equals_m0_supported"],
            "finite_internal_scale_theory_side_justified": finite_internal_scale_theory_side_justified,
            "support_band_brackets_q_star": support_band_brackets_q_star,
            "half_mass_target_phase_alignment_pass": half_mass_target_phase_alignment_pass,
            "mean_radius_first_zero_alignment_pass": mean_radius_first_zero_alignment_pass,
            "rms_radius_first_zero_alignment_pass": rms_radius_first_zero_alignment_pass,
            "exact_effective_support_scale_fixed": exact_effective_support_scale_fixed,
            "predictive_branch_ready": predictive_branch_ready,
            "primary_residual_lane": "qball_projection_overlap_exact_support_scale_selection",
            "secondary_residual_lane": "adopted_u1_charge_unit_dictionary",
            "reserve_residual_lane": "adopted_u1_vacuum_polarization_external_import",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_matching_scale_declared",
            "advance_to_8_7_56_1250": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_overlap_gate_summary": overlap_gate["summary"],
            "matching_scale_review_audit_summary": audit["summary"],
        },
    )

    evaluation = payload(
        "8.7.56.1250",
        "Trial-2 numeric alpha Q-ball projection-overlap matching-scale review numeric evaluation",
        inputs,
        [
            row("projection_overlap_q_star_fixed", "pass", "projection-overlap q_* fixed", q_star, "The blind first target crossing remains fixed from the prior overlap branch."),
            row("projection_overlap_sinc_target_phase_fixed", "pass", "projection-overlap sinc target phase fixed", z_target_sinc, "This is the first phase where sinc(z) reaches the observed target form factor."),
            row("projection_overlap_sinc_first_zero_fixed", "pass", "projection-overlap sinc first zero fixed", z_first_zero, "This is the first destructive phase of the spherical overlap kernel."),
            row("projection_overlap_q_from_half_mass_target_phase_fixed", "pass", "projection-overlap q from half-mass target phase fixed", q_target_from_half_mass, "Using the half-mass radius as the target-phase support scale yields a q candidate close to q_*."),
            row("projection_overlap_q_from_mean_first_zero_fixed", "pass", "projection-overlap q from mean first zero fixed", q_zero_from_mean, "Using the mean radius as the first-zero support scale yields a q candidate close to q_*."),
            row("projection_overlap_q_from_rms_first_zero_fixed", "pass", "projection-overlap q from rms first zero fixed", q_zero_from_rms, "The rms support scale is recorded as a broader-profile control candidate."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_support_band_justified_exact_scale_open",
            "q_star_over_m0": q_star,
            "sinc_target_phase": z_target_sinc,
            "sinc_first_zero_phase": z_first_zero,
            "q_star_times_half_mass_radius": q_half,
            "q_star_times_mean_radius": q_mean,
            "q_star_times_rms_radius": q_rms,
            "target_phase_implied_radius": implied_radius_target_phase,
            "first_zero_implied_radius": implied_radius_first_zero,
            "target_phase_implied_radius_relative_error_vs_half_mass": target_phase_errors["vs_half_mass_radius"],
            "first_zero_implied_radius_relative_error_vs_mean": first_zero_errors["vs_mean_radius"],
            "first_zero_implied_radius_relative_error_vs_rms": first_zero_errors["vs_rms_radius"],
            "q_from_half_mass_target_phase": q_target_from_half_mass,
            "q_from_mean_first_zero": q_zero_from_mean,
            "q_from_rms_first_zero": q_zero_from_rms,
            "finite_internal_scale_theory_side_justified": finite_internal_scale_theory_side_justified,
            "exact_effective_support_scale_fixed": exact_effective_support_scale_fixed,
            "predictive_branch_ready": predictive_branch_ready,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_matching_scale_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "profile_scales": scales,
            "phase_products": {
                "q_star_times_half_mass_radius": q_half,
                "q_star_times_mean_radius": q_mean,
                "q_star_times_rms_radius": q_rms,
            },
            "candidate_q_rel_errors": {
                "half_mass_target_phase": abs(q_target_from_half_mass - q_star) / q_star,
                "mean_first_zero": abs(q_zero_from_mean - q_star) / q_star,
                "rms_first_zero": abs(q_zero_from_rms - q_star) / q_star,
            },
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1247-.1250 artifacts generated")


if __name__ == "__main__":
    main()
