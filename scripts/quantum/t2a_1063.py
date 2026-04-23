#!/usr/bin/env python3
"""Generate 8.7.56.1063-.1066 Trial-2 numeric alpha alpha-is-prediction review artifacts."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EXPERT_NOTE_ALPHA = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_alpha_is_prediction.md")

SOURCE_1059 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "dimensionless_alpha_bridge_reclassification_source_inventory_metrics.json"
)
AUDIT_1060 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "dimensionless_alpha_bridge_reclassification_audit_metrics.json"
)
GATE_1061 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "dimensionless_alpha_bridge_reclassification_declaration_gate_metrics.json"
)
ROUTE_1062 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_second_refresh_metrics.json"
FINAL_SOURCE_979 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_source_inventory_metrics.json"
ELECTRON_AUDIT_732 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_electron_identification_audit_metrics.json"
QED_PRECISION = PUBLIC_OUT / "qed_vacuum_precision_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_alpha_is_prediction_review"
NEXT_UNIT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_unit_closure_review"
)
NEXT_UNIT_ARTIFACT = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_unit_closure_review_note"
)
NEXT_ROUTE = "8.7.56.1067"

ALPHA_NOTE_HEAD = "α は prediction であり parameter ではない"
ALPHA_NOTE_MCHI = "M_χ は Newton 定数から決まる"
ALPHA_NOTE_V = "v は既存拘束で決まる"
ALPHA_NOTE_H0 = "H_0^{(P)} = \\frac{m_0}{\\sqrt{Z_P^{\\rm grav}}}"
ALPHA_NOTE_CLOSED_FORM = "\\boxed{\\alpha = \\frac{c^3}{4\\pi v^2 \\hbar}}"
ALPHA_NOTE_PARAMETER = "prediction であり、実験から入れる parameter ではない"
PART3A_ALPHA_REVIEW = "alpha-is-prediction review"
PART5_ALPHA_SCOPE = "alpha-is-prediction review scope"
PART3A_BRIDGE_HEAD = "current checkpoint wording としては、電磁結合は Part I 2.7.0 の vector kinetic coefficient"
PART2_H0P_LAW = r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]"
PART1_SCALAR_KINETIC = r"\frac{M_\chi^2}{2}\partial_\mu\chi\,\partial^\mu\chi"

G_DIMS = {"kg": Fraction(-1), "m": Fraction(3), "s": Fraction(-2)}
C_DIMS = {"m": Fraction(1), "s": Fraction(-1)}
HBAR_DIMS = {"kg": Fraction(1), "m": Fraction(2), "s": Fraction(-1)}
MASS_DIMS = {"kg": Fraction(1)}
H0P_DIMS = {"s": Fraction(-1)}


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution when a required path is missing.

def require(path: Path) -> None:
    """Require one input path to exist before execution continues."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return a stable display path for repo files.

def display_path(path: Path) -> str:
    """Return a stable path relative to the repo root when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing a substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for the given substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard metrics payload.

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
    """Build one standard metrics payload."""
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


# Function: write a JSON metrics artifact and the matching CSV rows table.

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


# Function: scale a dimension dictionary by a rational factor.

def scale_dims(dims: dict[str, Fraction], factor: Fraction) -> dict[str, Fraction]:
    """Scale a dimension dictionary by one rational factor."""
    return {key: value * factor for key, value in dims.items()}


# Function: add two dimension dictionaries.

def add_dims(left: dict[str, Fraction], right: dict[str, Fraction]) -> dict[str, Fraction]:
    """Add two dimension dictionaries."""
    merged: dict[str, Fraction] = {}
    for key in set(left) | set(right):
        merged[key] = left.get(key, Fraction(0)) + right.get(key, Fraction(0))

    return {key: value for key, value in merged.items() if value != 0}


# Function: subtract the right dimension dictionary from the left.

def sub_dims(left: dict[str, Fraction], right: dict[str, Fraction]) -> dict[str, Fraction]:
    """Subtract the right dimension dictionary from the left."""
    return add_dims(left, scale_dims(right, Fraction(-1)))


# Function: take the square root of a dimension dictionary.

def sqrt_dims(dims: dict[str, Fraction]) -> dict[str, Fraction]:
    """Take the square root of one dimension dictionary."""
    return {key: value / Fraction(2) for key, value in dims.items()}


# Function: format one rational exponent for metrics output.

def format_exponent(value: Fraction) -> str:
    """Format one rational exponent as an integer or fraction string."""
    if value.denominator == 1:
        return str(value.numerator)

    return f"{value.numerator}/{value.denominator}"


# Function: format one dimension dictionary as a compact string.

def format_dims(dims: dict[str, Fraction]) -> str:
    """Format one dimension dictionary as a compact string."""
    if not dims:
        return "dimensionless"

    parts: list[str] = []
    for key in ("kg", "m", "s"):
        if key not in dims:
            continue

        parts.append(f"{key}^{format_exponent(dims[key])}")

    return " ".join(parts)


# Function: compare two dimension dictionaries.

def same_dims(left: dict[str, Fraction], right: dict[str, Fraction]) -> bool:
    """Return whether two dimension dictionaries are identical."""
    return left == right


# Function: classify the alpha-is-prediction review outcome.

def classify_review(alpha_dimensionless_in_si: bool, relative_error: float) -> str:
    """Classify the alpha-is-prediction route under unit and numeric gates."""
    if not alpha_dimensionless_in_si:
        if relative_error >= 0.50:
            return "unit_mismatch_dominant_with_large_numeric_tension"

        return "unit_mismatch_dominant"

    if relative_error < 0.10:
        return "numeric_pass"

    if relative_error < 0.50:
        return "numeric_constraint_watch"

    return "numeric_tension_reject"


# Function: execute the alpha-is-prediction review branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha alpha-is-prediction review branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART2,
        PART3A,
        PART5,
        EXPERT_NOTE_ALPHA,
        SOURCE_1059,
        AUDIT_1060,
        GATE_1061,
        ROUTE_1062,
        FINAL_SOURCE_979,
        ELECTRON_AUDIT_732,
        QED_PRECISION,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part2_text = read_text(PART2)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    alpha_note_text = read_text(EXPERT_NOTE_ALPHA)

    source_1059 = read_json(SOURCE_1059)["summary"]
    audit_1060 = read_json(AUDIT_1060)["summary"]
    gate_1061 = read_json(GATE_1061)["summary"]
    route_1062 = read_json(ROUTE_1062)["summary"]
    final_source_979 = read_json(FINAL_SOURCE_979)["summary"]
    electron_audit_732 = read_json(ELECTRON_AUDIT_732)["summary"]
    qed_precision = read_json(QED_PRECISION)

    status_has_1063_next_step = hit(status_text, "8.7.56.1063") is not None
    roadmap_has_1063_branch = hit(roadmap_text, "`8.7.56.1063-.1066`") is not None
    part3a_has_alpha_review = hit(part3a_text, PART3A_ALPHA_REVIEW) is not None
    part3a_has_bridge_head = hit(part3a_text, PART3A_BRIDGE_HEAD) is not None
    part5_has_alpha_scope = hit(part5_text, PART5_ALPHA_SCOPE) is not None
    part2_has_h0p_law = hit(part2_text, PART2_H0P_LAW) is not None
    part1_has_scalar_kinetic = hit(part1_text, PART1_SCALAR_KINETIC) is not None

    alpha_note_has_head = hit(alpha_note_text, ALPHA_NOTE_HEAD) is not None
    alpha_note_has_mchi = hit(alpha_note_text, ALPHA_NOTE_MCHI) is not None
    alpha_note_has_v = hit(alpha_note_text, ALPHA_NOTE_V) is not None
    alpha_note_has_h0 = hit(alpha_note_text, ALPHA_NOTE_H0) is not None
    alpha_note_has_closed_form = hit(alpha_note_text, ALPHA_NOTE_CLOSED_FORM) is not None
    alpha_note_declares_prediction_not_parameter = hit(alpha_note_text, ALPHA_NOTE_PARAMETER) is not None
    alpha_prediction_route_available = all(
        [
            alpha_note_has_head,
            alpha_note_has_mchi,
            alpha_note_has_v,
            alpha_note_has_h0,
            alpha_note_has_closed_form,
            alpha_note_declares_prediction_not_parameter,
        ]
    )

    prior_alpha_prediction_route_active = (
        source_1059["first_route_to_close_or_none"] == CURRENT_ROUTE
        and audit_1060["first_route_to_close_after_audit_or_none"] == CURRENT_ROUTE
        and gate_1061["selected_residual_route"] == CURRENT_ROUTE
        and route_1062["selected_next_generation_route"] == CURRENT_ROUTE
        and bool(route_1062["alpha_prediction_review_required"])
    )
    current_public_bridge_sentence_promoted = bool(
        route_1062["current_canon_bridge_statement_checkpoint_wording_promotion_completed"]
    )
    frozen_e_beta1_available = bool(final_source_979["E_beta1_available"])
    frozen_h0p_available = bool(final_source_979["H0P_si_available"])
    electron_identification_dictionary_ready = bool(
        electron_audit_732["absolute_normalization_dictionary_ready"]
    )

    inventory_ready = all(
        [
            status_has_1063_next_step,
            roadmap_has_1063_branch,
            part3a_has_alpha_review,
            part3a_has_bridge_head,
            part5_has_alpha_scope,
            part2_has_h0p_law,
            part1_has_scalar_kinetic,
            prior_alpha_prediction_route_active,
            current_public_bridge_sentence_promoted,
            alpha_prediction_route_available,
            frozen_e_beta1_available,
            frozen_h0p_available,
            electron_identification_dictionary_ready,
        ]
    )

    constants_si = qed_precision["constants_si"]
    g_si = float(constants_si["G_m3_kg_s2"])
    hbar_si = float(constants_si["hbar_j_s"])
    c_si = float(constants_si["c_m_per_s"])
    m_e_si = float(constants_si["m_e_kg"])
    alpha_target = 1.0 / float(qed_precision["alpha_precision"]["g2"]["alpha_inv"])
    e_beta1 = float(final_source_979["E_beta1_value"])
    h0p_si = float(final_source_979["H0P_si_value"])

    mchi_sq = (c_si**4) / (4.0 * math.pi * g_si)
    mchi = math.sqrt(mchi_sq)
    m0_kg = m_e_si / e_beta1
    v_candidate = h0p_si * mchi / m0_kg
    alpha_candidate = (c_si**3) / (4.0 * math.pi * (v_candidate**2) * hbar_si)
    alpha_ratio_to_target = alpha_candidate / alpha_target
    relative_error = abs(alpha_candidate - alpha_target) / alpha_target
    log10_gap_to_target = math.log10(alpha_target / alpha_candidate)

    mchi_sq_dims = sub_dims(scale_dims(C_DIMS, Fraction(4)), G_DIMS)
    mchi_dims = sqrt_dims(mchi_sq_dims)
    v_dims = add_dims(add_dims(H0P_DIMS, mchi_dims), scale_dims(MASS_DIMS, Fraction(-1)))
    alpha_dims = sub_dims(scale_dims(C_DIMS, Fraction(3)), add_dims(scale_dims(v_dims, Fraction(2)), HBAR_DIMS))
    mchi_candidate_has_mass_dimension_in_si = same_dims(mchi_dims, MASS_DIMS)
    alpha_candidate_dimensionless_in_si = not alpha_dims
    pass_10pct = alpha_candidate_dimensionless_in_si and relative_error < 0.10
    watch_10_to_50pct = alpha_candidate_dimensionless_in_si and 0.10 <= relative_error < 0.50
    large_numeric_tension_present = relative_error >= 0.50
    reject_gt_50pct = alpha_candidate_dimensionless_in_si and large_numeric_tension_present
    selected_review_class = classify_review(alpha_candidate_dimensionless_in_si, relative_error)

    common_inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "primary_sources_markdown": display_path(PRIMARY_SOURCES),
        "part1_markdown": display_path(PART1),
        "part2_markdown": display_path(PART2),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "alpha_is_prediction_note": display_path(EXPERT_NOTE_ALPHA),
        "prior_1059_json": display_path(SOURCE_1059),
        "prior_1060_json": display_path(AUDIT_1060),
        "prior_1061_json": display_path(GATE_1061),
        "prior_1062_json": display_path(ROUTE_1062),
        "retained_979_json": display_path(FINAL_SOURCE_979),
        "retained_732_json": display_path(ELECTRON_AUDIT_732),
        "qed_precision_json": display_path(QED_PRECISION),
    }

    inventory = payload(
        "8.7.56.1063",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization alpha-is-prediction review source inventory",
        common_inputs,
        "Freeze the alpha-is-prediction review pack: the external no-free-parameter note, retained E(beta_1) / H0^(P) frozen values, the electron-identification dictionary, the promoted bridge sentence, and the CODATA/QED constants used for the closed-form evaluation.",
        {
            "inventory_rule": "the alpha-is-prediction pack is ready when the note, frozen inputs, public bridge sentence, and retained electron-identification dictionary are visible in one pack",
            "review_rule": "the next honest computation is M_chi = c^2/sqrt(4*pi*G), m0 = m_e/E(beta_1), v = H0^(P) * M_chi / m0, alpha = c^3/(4*pi*v^2*hbar)",
        },
        [
            row(
                "trial2_numeric_alpha_alpha_prediction_inventory_complete",
                "pass" if inventory_ready else "reject",
                "alpha-is-prediction inventory complete",
                1 if inventory_ready else 0,
                "The note, retained frozen values, public bridge sentence, and retained electron-identification dictionary are assembled into one computation pack.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_note_available",
                "pass" if alpha_prediction_route_available else "reject",
                "alpha-is-prediction note available",
                1 if alpha_prediction_route_available else 0,
                "The note supplies the M_chi-from-G step, the v-from-H0/m0 step, and the closed-form alpha candidate.",
            ),
            row(
                "trial2_numeric_alpha_frozen_e_beta1_available_for_alpha_prediction",
                "pass" if frozen_e_beta1_available else "reject",
                "frozen E(beta_1) available for alpha prediction",
                1 if frozen_e_beta1_available else 0,
                "The retained final-computation input pack already froze E(beta_1) numerically.",
            ),
            row(
                "trial2_numeric_alpha_frozen_h0p_available_for_alpha_prediction",
                "pass" if frozen_h0p_available else "reject",
                "frozen H0^(P) available for alpha prediction",
                1 if frozen_h0p_available else 0,
                "The retained final-computation input pack already froze H0^(P) numerically.",
            ),
            row(
                "trial2_numeric_alpha_electron_identification_dictionary_retained_for_alpha_prediction",
                "pass" if electron_identification_dictionary_ready else "reject",
                "electron-identification dictionary retained for alpha prediction",
                1 if electron_identification_dictionary_ready else 0,
                "The absolute-normalization dictionary M_(1,0,0,0) = m_e remains the retained m0 anchor.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "prior_alpha_prediction_route_active": prior_alpha_prediction_route_active,
            "current_public_bridge_sentence_promoted": current_public_bridge_sentence_promoted,
            "alpha_prediction_route_available": alpha_prediction_route_available,
            "frozen_E_beta1_available": frozen_e_beta1_available,
            "frozen_E_beta1_value": e_beta1,
            "frozen_H0P_available": frozen_h0p_available,
            "frozen_H0P_value": h0p_si,
            "electron_identification_dictionary_ready": electron_identification_dictionary_ready,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_alpha_prediction_inventory_frozen",
            "advance_to_8_7_56_1064": inventory_ready,
            "next_required_artifacts": [CURRENT_ROUTE],
        },
        {
            "status_hits": {
                "status_next_1063": hit(status_text, "8.7.56.1063"),
                "roadmap_branch_1063": hit(roadmap_text, "`8.7.56.1063-.1066`"),
                "part3a_alpha_review": hit(part3a_text, PART3A_ALPHA_REVIEW),
                "part5_alpha_scope": hit(part5_text, PART5_ALPHA_SCOPE),
            },
            "expert_note_hits": {
                "alpha_note_head": hit(alpha_note_text, ALPHA_NOTE_HEAD),
                "alpha_note_mchi": hit(alpha_note_text, ALPHA_NOTE_MCHI),
                "alpha_note_v": hit(alpha_note_text, ALPHA_NOTE_V),
                "alpha_note_h0": hit(alpha_note_text, ALPHA_NOTE_H0),
                "alpha_note_closed_form": hit(alpha_note_text, ALPHA_NOTE_CLOSED_FORM),
                "alpha_note_prediction_not_parameter": hit(alpha_note_text, ALPHA_NOTE_PARAMETER),
            },
        },
    )

    audit = payload(
        "8.7.56.1064",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization alpha-is-prediction review audit",
        common_inputs,
        "Evaluate the no-free-parameter alpha note under the frozen SI inputs and classify both its numeric behavior and its SI dimensional consistency.",
        {
            "mchi_rule": "M_chi^2 = c^4 / (4*pi*G), M_chi = c^2 / sqrt(4*pi*G)",
            "m0_rule": "m0 = m_e / E(beta_1)",
            "v_rule": "v = H0^(P) * M_chi / m0",
            "alpha_rule": "alpha = c^3 / (4*pi*v^2*hbar)",
            "classification_rule": "unit mismatch dominates if the alpha candidate is not dimensionless in SI even when a large numeric tension is also present",
        },
        [
            row(
                "trial2_numeric_alpha_alpha_prediction_audit_complete",
                "pass" if inventory_ready else "reject",
                "alpha-is-prediction audit complete",
                1 if inventory_ready else 0,
                "The no-free-parameter chain is evaluated once under the retained frozen inputs.",
            ),
            row(
                "trial2_numeric_alpha_mchi_from_g_candidate_has_mass_dimension_in_si",
                "pass" if mchi_candidate_has_mass_dimension_in_si else "reject",
                "M_chi-from-G candidate has mass dimension in SI",
                1 if mchi_candidate_has_mass_dimension_in_si else 0,
                f"The direct SI readout gives M_chi dimensions {format_dims(mchi_dims)} rather than pure mass.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_candidate_dimensionless_in_si",
                "pass" if alpha_candidate_dimensionless_in_si else "reject",
                "alpha prediction candidate dimensionless in SI",
                1 if alpha_candidate_dimensionless_in_si else 0,
                f"The closed-form alpha candidate carries dimensions {format_dims(alpha_dims)} under the retained SI interpretation.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_pass_10pct",
                "pass" if pass_10pct else "reject",
                "alpha prediction relative error below 10%",
                1 if pass_10pct else 0,
                "A pass requires both SI-dimensional admissibility and <=10% agreement with the QED target.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_large_numeric_tension_present",
                "pass" if large_numeric_tension_present else "reject",
                "alpha prediction large numeric tension present",
                1 if large_numeric_tension_present else 0,
                "The quick pilot mismatch remains present under the official frozen constants as well.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_unit_mismatch_dominates_classification",
                "pass" if not alpha_candidate_dimensionless_in_si else "reject",
                "alpha prediction unit mismatch dominates classification",
                1 if not alpha_candidate_dimensionless_in_si else 0,
                "Because the alpha candidate is not dimensionless in SI, the numeric mismatch cannot yet be interpreted as a final physical reject.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "Mchi_sq_value_si": mchi_sq,
            "Mchi_sq_dimension_vector_si": format_dims(mchi_sq_dims),
            "Mchi_value_si": mchi,
            "Mchi_dimension_vector_si": format_dims(mchi_dims),
            "m0_kg": m0_kg,
            "m0_dimension_vector_si": format_dims(MASS_DIMS),
            "v_value_si": v_candidate,
            "v_dimension_vector_si": format_dims(v_dims),
            "alpha_candidate_value": alpha_candidate,
            "alpha_candidate_dimension_vector_si": format_dims(alpha_dims),
            "alpha_target": alpha_target,
            "alpha_ratio_to_target": alpha_ratio_to_target,
            "relative_error": relative_error,
            "log10_gap_to_target": log10_gap_to_target,
            "mchi_candidate_has_mass_dimension_in_si": mchi_candidate_has_mass_dimension_in_si,
            "alpha_candidate_dimensionless_in_si": alpha_candidate_dimensionless_in_si,
            "watch_10_to_50pct": watch_10_to_50pct,
            "reject_gt_50pct": reject_gt_50pct,
            "large_numeric_tension_present": large_numeric_tension_present,
            "selected_alpha_prediction_review_class": selected_review_class,
            "numeric_closeout_ready": False,
            "first_route_to_close_after_audit_or_none": NEXT_UNIT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_alpha_prediction_audited",
            "advance_to_8_7_56_1065": True,
            "next_required_artifacts": [NEXT_UNIT_ROUTE],
        },
        {
            "constants_si": {
                "G": g_si,
                "hbar": hbar_si,
                "c": c_si,
                "m_e": m_e_si,
            },
            "retained_input_summary": {
                "E_beta1": e_beta1,
                "H0P_si_s^-1": h0p_si,
                "electron_identification_dictionary_ready": electron_identification_dictionary_ready,
            },
            "expert_note_hits": inventory["evidence"]["expert_note_hits"],
        },
    )

    gate = payload(
        "8.7.56.1065",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization alpha-is-prediction review declaration gate",
        common_inputs,
        "Fix the official outcome of the alpha-is-prediction branch: the route is computation-side alive, but the current SI realization lands on unit mismatch before any final numeric pass/reject claim can be made.",
        {
            "gate_rule": "alpha-is-prediction review closes as unit-mismatch-first when the closed-form alpha candidate is not dimensionless in SI",
            "next_route_rule": "the next residual route is unit-closure review for the alpha-is-prediction chain",
        },
        [
            row(
                "trial2_numeric_alpha_alpha_prediction_gate_complete",
                "pass",
                "alpha-is-prediction gate complete",
                1,
                "The official gate is updated after evaluating the no-free-parameter chain.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_review_completed",
                "pass" if inventory_ready else "reject",
                "alpha-is-prediction review completed",
                1 if inventory_ready else 0,
                "The computation-side review itself is now complete even though closeout is not.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_unit_mismatch_selected",
                "pass" if not alpha_candidate_dimensionless_in_si else "reject",
                "alpha-is-prediction unit mismatch selected",
                1 if not alpha_candidate_dimensionless_in_si else 0,
                "The closed-form alpha candidate is not dimensionless in SI, so unit closure is the next honest blocker.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_closeout_ready",
                "reject",
                "alpha-is-prediction closeout ready",
                0,
                "The branch cannot close numerically because the current SI chain still needs a unit bridge before the mismatch can be interpreted physically.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_unit_mismatch_review",
            "trial2_numeric_alpha_text_search_continuation_justified": False,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": True,
            "trial2_numeric_alpha_alpha_prediction_review_completed": inventory_ready,
            "trial2_numeric_alpha_alpha_prediction_route_available": alpha_prediction_route_available,
            "trial2_numeric_alpha_alpha_prediction_candidate_dimensionless_in_si": alpha_candidate_dimensionless_in_si,
            "trial2_numeric_alpha_alpha_prediction_large_numeric_tension_present": large_numeric_tension_present,
            "trial2_numeric_alpha_alpha_prediction_selected_class": selected_review_class,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_UNIT_ROUTE,
            "missing_v2_artifact": NEXT_UNIT_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_alpha_prediction_gate_closed",
            "advance_to_8_7_56_1066": True,
            "next_required_artifacts": [NEXT_UNIT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_route_summary": route_1062,
        },
    )

    route = payload(
        "8.7.56.1066",
        "Trial-2 numeric alpha route contract one-hundred-sixty-third refresh",
        common_inputs,
        "Refresh the next-generation contract after alpha-is-prediction review: keep the precision-alpha mainline alive, record the large numeric tension as diagnostic only, and advance the route to unit-closure review.",
        {
            "next_route_rule": "the next route isolates the first missing SI/natural-unit bridge inside the alpha-is-prediction chain",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_sixty_third_refresh_complete",
                "pass",
                "route contract one-hundred-sixty-third refresh complete",
                1,
                "The alpha-is-prediction gate is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_alpha_prediction_unit_closure_review",
                "pass" if not alpha_candidate_dimensionless_in_si else "reject",
                "next route selected as alpha-is-prediction unit-closure review",
                1 if not alpha_candidate_dimensionless_in_si else 0,
                "The next official branch isolates the first missing unit bridge inside the alpha-is-prediction chain.",
            ),
            row(
                "trial2_numeric_alpha_alpha_prediction_large_numeric_tension_retained_as_diagnostic_only",
                "pass" if large_numeric_tension_present else "reject",
                "alpha-is-prediction large numeric tension retained as diagnostic only",
                1 if large_numeric_tension_present else 0,
                "The large mismatch is retained as evidence, but it is not promoted to final reject while the SI unit issue remains open.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_alpha_prediction_review",
                "pass" if bool(route_1062.get("precision_alpha_mainline_retained", False)) else "reject",
                "precision-alpha mainline retained after alpha-is-prediction review",
                1 if bool(route_1062.get("precision_alpha_mainline_retained", False)) else 0,
                "Trial-2 numeric alpha remains the precision mainline while the unit-closure residual is audited.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_UNIT_ROUTE,
            "strong_side_route_state": route_1062.get("strong_side_route_state"),
            "precision_alpha_mainline_retained": bool(route_1062.get("precision_alpha_mainline_retained", False)),
            "electron_identification_pivot_retained": bool(route_1062.get("electron_identification_pivot_retained", False)),
            "h0p_bridge_pivot_retained": bool(route_1062.get("h0p_bridge_pivot_retained", False)),
            "final_computation_branch_retained": True,
            "unit_consistency_audit_branch_retained": True,
            "dimensionless_alpha_bridge_branch_retained": True,
            "em_unit_convention_bridge_branch_retained": True,
            "mapping_statement_branch_retained": True,
            "mapping_literal_branch_retained": True,
            "two_sector_hierarchy_pivot_retained": True,
            "current_canon_bridge_statement_checkpoint_wording_promotion_completed": bool(
                route_1062.get("current_canon_bridge_statement_checkpoint_wording_promotion_completed", False)
            ),
            "dimensionless_alpha_bridge_reopen_completed": bool(
                route_1062.get("dimensionless_alpha_bridge_reopen_completed", False)
            ),
            "current_canon_no_go_closeout_candidate_retired": bool(
                route_1062.get("current_canon_no_go_closeout_candidate_retired", False)
            ),
            "alpha_prediction_review_completed": inventory_ready,
            "alpha_prediction_unit_mismatch_confirmed": not alpha_candidate_dimensionless_in_si,
            "alpha_prediction_large_numeric_tension_observed": large_numeric_tension_present,
            "external_dependency_active": False,
            "hard_conflict_reading_retired": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_sixty_third_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_UNIT_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "audit_summary": audit["summary"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_review_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_review_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_review_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_third_refresh", route)

    print("[done] 8.7.56.1063-.1066 artifacts generated:")
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_review_source_inventory_metrics.json"
    )
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_review_audit_metrics.json"
    )
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_review_declaration_gate_metrics.json"
    )
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_third_refresh_metrics.json")


if __name__ == "__main__":
    main()
