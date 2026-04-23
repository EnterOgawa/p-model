#!/usr/bin/env python3
"""Generate 8.7.56.1067-.1070 Trial-2 numeric alpha alpha-is-prediction unit-closure review artifacts."""

from __future__ import annotations

import csv
import json
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
EXPERT_NOTE_ZPEM = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_zp_em_equals_one.md")

SOURCE_1063 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_review_source_inventory_metrics.json"
)
AUDIT_1064 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_review_audit_metrics.json"
)
GATE_1065 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_review_declaration_gate_metrics.json"
)
ROUTE_1066 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_third_refresh_metrics.json"
FINAL_SOURCE_979 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_source_inventory_metrics.json"
UNIT_AUDIT_984 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_audit_metrics.json"
ELECTRON_AUDIT_732 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_electron_identification_audit_metrics.json"
QED_PRECISION = PUBLIC_OUT / "qed_vacuum_precision_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_unit_closure_review"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_alpha_formula_unit_bridge_review"
)
NEXT_ROUTE_ARTIFACT = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_alpha_formula_unit_bridge_review_note"
)
NEXT_ROUTE = "8.7.56.1071"

ALPHA_NOTE_HEAD = "α は prediction であり parameter ではない"
ALPHA_NOTE_H0 = "H_0^{(P)} = \\frac{m_0}{\\sqrt{Z_P^{\\rm grav}}}"
ALPHA_NOTE_V = "v = \\frac{H_0^{(P)} \\cdot M_\\chi}{m_0}"
ALPHA_NOTE_ALPHA = "\\boxed{\\alpha = \\frac{c^3}{4\\pi v^2 \\hbar}}"
ALPHA_NOTE_MCHI = "M_\\chi^2 = \\frac{c^4}{4\\pi G}"
ZPEM_NOTE_GRAV = "Z_P^{\\rm grav} = M_\\chi^2/v^2"
PART1_SCALAR_KINETIC = r"\frac{M_\chi^2}{2}\partial_\mu\chi\,\partial^\mu\chi"
PART2_H0P_LAW = r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]"
PART3A_UNIT_REVIEW_NEXT = "alpha-is-prediction unit-closure review next"
PART5_UNIT_SCOPE = "alpha-is-prediction unit-closure review scope"
PART5_NEXT_BRANCH = "8.7.56.1067-.1070"

G_DIMS = {"kg": Fraction(-1), "m": Fraction(3), "s": Fraction(-2)}
C_DIMS = {"m": Fraction(1), "s": Fraction(-1)}
HBAR_DIMS = {"kg": Fraction(1), "m": Fraction(2), "s": Fraction(-1)}
MASS_DIMS = {"kg": Fraction(1)}
FREQUENCY_DIMS = {"s": Fraction(-1)}


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: fail fast when one required input path is missing.

def require(path: Path) -> None:
    """Require one input path to exist before execution continues."""
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


# Function: return a stable display path.

def display_path(path: Path) -> str:
    """Return a path relative to the repo root when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing a substring pattern.

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


# Function: scale one dimension dictionary by a rational factor.

def scale_dims(dims: dict[str, Fraction], factor: Fraction) -> dict[str, Fraction]:
    """Scale one dimension dictionary by one rational factor."""
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
    """Format one rational exponent as an integer or a fraction string."""
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


# Function: classify the unit-closure review outcome.

def classify_unit_review(
    h0p_bridge_matches_c2_over_hbar: bool,
    corrected_v_matches_expected: bool,
    alpha_h0p_bridge_alone_resolves_units: bool,
) -> str:
    """Classify the alpha-is-prediction unit-closure review outcome."""
    if h0p_bridge_matches_c2_over_hbar and corrected_v_matches_expected and not alpha_h0p_bridge_alone_resolves_units:
        return "first_missing_h0p_mass_frequency_bridge_identified_but_alpha_formula_still_needs_additional_unit_bridge"

    if h0p_bridge_matches_c2_over_hbar and alpha_h0p_bridge_alone_resolves_units:
        return "h0p_mass_frequency_bridge_sufficient_for_unit_closure"

    return "unit_bridge_structure_unresolved"


# Function: execute the alpha-is-prediction unit-closure review branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha alpha-is-prediction unit-closure review branch."""
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
        EXPERT_NOTE_ZPEM,
        SOURCE_1063,
        AUDIT_1064,
        GATE_1065,
        ROUTE_1066,
        FINAL_SOURCE_979,
        UNIT_AUDIT_984,
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
    zpem_note_text = read_text(EXPERT_NOTE_ZPEM)

    source_1063 = read_json(SOURCE_1063)["summary"]
    audit_1064 = read_json(AUDIT_1064)["summary"]
    gate_1065 = read_json(GATE_1065)["summary"]
    route_1066 = read_json(ROUTE_1066)["summary"]
    final_source_979 = read_json(FINAL_SOURCE_979)["summary"]
    unit_audit_984 = read_json(UNIT_AUDIT_984)["summary"]
    electron_audit_732 = read_json(ELECTRON_AUDIT_732)["summary"]
    qed_precision = read_json(QED_PRECISION)

    status_has_1067_next_step = hit(status_text, "8.7.56.1067") is not None
    roadmap_has_1067_branch = hit(roadmap_text, "`8.7.56.1067-.1070`") is not None
    part1_has_scalar_kinetic = hit(part1_text, PART1_SCALAR_KINETIC) is not None
    part2_has_h0p_law = hit(part2_text, PART2_H0P_LAW) is not None
    part3a_has_unit_review_next = hit(part3a_text, PART3A_UNIT_REVIEW_NEXT) is not None
    part5_has_unit_scope = hit(part5_text, PART5_UNIT_SCOPE) is not None
    part5_has_next_branch = hit(part5_text, PART5_NEXT_BRANCH) is not None

    alpha_note_has_head = hit(alpha_note_text, ALPHA_NOTE_HEAD) is not None
    alpha_note_has_h0 = hit(alpha_note_text, ALPHA_NOTE_H0) is not None
    alpha_note_has_v = hit(alpha_note_text, ALPHA_NOTE_V) is not None
    alpha_note_has_alpha = hit(alpha_note_text, ALPHA_NOTE_ALPHA) is not None
    alpha_note_has_mchi = hit(alpha_note_text, ALPHA_NOTE_MCHI) is not None
    zpem_note_has_zpgrav = hit(zpem_note_text, ZPEM_NOTE_GRAV) is not None

    prior_unit_review_route_active = (
        gate_1065["selected_residual_route"] == CURRENT_ROUTE
        and route_1066["selected_next_generation_route"] == CURRENT_ROUTE
        and bool(route_1066["alpha_prediction_review_completed"])
        and bool(route_1066["alpha_prediction_unit_mismatch_confirmed"])
    )
    frozen_e_beta1_available = bool(final_source_979["E_beta1_available"])
    frozen_h0p_available = bool(final_source_979["H0P_si_available"])
    electron_identification_dictionary_ready = bool(
        electron_audit_732["absolute_normalization_dictionary_ready"]
    )

    inventory_ready = all(
        [
            status_has_1067_next_step,
            roadmap_has_1067_branch,
            part1_has_scalar_kinetic,
            part2_has_h0p_law,
            part3a_has_unit_review_next,
            part5_has_unit_scope,
            part5_has_next_branch,
            alpha_note_has_head,
            alpha_note_has_h0,
            alpha_note_has_v,
            alpha_note_has_alpha,
            alpha_note_has_mchi,
            zpem_note_has_zpgrav,
            prior_unit_review_route_active,
            frozen_e_beta1_available,
            frozen_h0p_available,
            electron_identification_dictionary_ready,
        ]
    )

    constants_si = qed_precision["constants_si"]
    c_si = float(constants_si["c_m_per_s"])
    hbar_si = float(constants_si["hbar_j_s"])
    c2_over_hbar = (c_si**2) / hbar_si
    hbar_over_c2 = hbar_si / (c_si**2)

    mchi_sq_dims = sub_dims(scale_dims(C_DIMS, Fraction(4)), G_DIMS)
    mchi_dims = sqrt_dims(mchi_sq_dims)
    current_v_dims = add_dims(add_dims(FREQUENCY_DIMS, mchi_dims), scale_dims(MASS_DIMS, Fraction(-1)))
    current_alpha_dims = sub_dims(scale_dims(C_DIMS, Fraction(3)), add_dims(scale_dims(current_v_dims, Fraction(2)), HBAR_DIMS))

    inferred_zpgrav_dimensionless = True
    expected_v_dims_if_zpgrav_dimensionless = mchi_dims
    h0p_rhs_bridge_required_dims = sub_dims(FREQUENCY_DIMS, MASS_DIMS)
    c2_over_hbar_dims = sub_dims(scale_dims(C_DIMS, Fraction(2)), HBAR_DIMS)
    h0p_bridge_matches_c2_over_hbar = same_dims(h0p_rhs_bridge_required_dims, c2_over_hbar_dims)
    v_bridge_multiplier_dims = sub_dims(HBAR_DIMS, scale_dims(C_DIMS, Fraction(2)))
    corrected_v_dims = add_dims(current_v_dims, v_bridge_multiplier_dims)
    corrected_v_matches_expected = same_dims(corrected_v_dims, expected_v_dims_if_zpgrav_dimensionless)
    alpha_dims_after_h0p_bridge = sub_dims(
        scale_dims(C_DIMS, Fraction(3)),
        add_dims(scale_dims(corrected_v_dims, Fraction(2)), HBAR_DIMS),
    )
    alpha_h0p_bridge_alone_resolves_units = not alpha_dims_after_h0p_bridge
    alpha_additional_bridge_required_dims = scale_dims(alpha_dims_after_h0p_bridge, Fraction(-1))
    mchi_direct_mass_readout_admissible_in_si = same_dims(mchi_dims, MASS_DIMS)
    mchi_mass_bridge_required_dims = sub_dims(MASS_DIMS, mchi_dims)

    selected_review_class = classify_unit_review(
        h0p_bridge_matches_c2_over_hbar,
        corrected_v_matches_expected,
        alpha_h0p_bridge_alone_resolves_units,
    )

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
        "zp_em_equals_one_note": display_path(EXPERT_NOTE_ZPEM),
        "prior_1063_json": display_path(SOURCE_1063),
        "prior_1064_json": display_path(AUDIT_1064),
        "prior_1065_json": display_path(GATE_1065),
        "prior_1066_json": display_path(ROUTE_1066),
        "retained_979_json": display_path(FINAL_SOURCE_979),
        "retained_984_json": display_path(UNIT_AUDIT_984),
        "retained_732_json": display_path(ELECTRON_AUDIT_732),
        "qed_precision_json": display_path(QED_PRECISION),
    }

    inventory = payload(
        "8.7.56.1067",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization alpha-is-prediction unit-closure review source inventory",
        common_inputs,
        "Freeze the unit-closure review pack: the alpha-is-prediction note chain, the Z_P^grav reading, the retained H0^(P)-Z_P bridge, the electron-identification dictionary, the prior unit-consistency audit, and the CODATA/QED constants.",
        {
            "inventory_rule": "the unit-closure pack is ready when the no-free-parameter alpha note, the retained H0^(P)-Z_P branch, the retained unit-consistency audit, and the public bridge sentence are visible together",
            "audit_rule": "the next audit checks the Newton-Mchi step, the H0^(P)=m0/sqrt(Z_P^grav) mapping, the v readout, and the closed-form alpha formula in SI dimensions",
        },
        [
            row(
                "trial2_numeric_alpha_alpha_prediction_unit_closure_inventory_complete",
                "pass" if inventory_ready else "reject",
                "alpha-is-prediction unit-closure inventory complete",
                1 if inventory_ready else 0,
                "The note chain, retained H0^(P)-Z_P bridge, prior unit-consistency audit, and retained electron-identification dictionary are assembled into one pack.",
            ),
            row(
                "trial2_numeric_alpha_h0p_mapping_surface_available",
                "pass" if alpha_note_has_h0 else "reject",
                "H0^(P) mapping surface available",
                1 if alpha_note_has_h0 else 0,
                "The alpha-is-prediction note explicitly states H0^(P) = m0 / sqrt(Z_P^grav).",
            ),
            row(
                "trial2_numeric_alpha_zpgrav_surface_available",
                "pass" if zpem_note_has_zpgrav else "reject",
                "Z_P^grav surface available",
                1 if zpem_note_has_zpgrav else 0,
                "The Z_P^grav = M_chi^2 / v^2 reading is retained from the expert clarification note.",
            ),
            row(
                "trial2_numeric_alpha_prior_unit_consistency_audit_retained",
                "pass" if "mass_frequency_bridge_alpha_dimension_vector_si" in unit_audit_984 else "reject",
                "prior unit-consistency audit retained",
                1 if "mass_frequency_bridge_alpha_dimension_vector_si" in unit_audit_984 else 0,
                "The older mass-frequency-bridge probe remains available as diagnostic context for the current branch.",
            ),
            row(
                "trial2_numeric_alpha_electron_identification_dictionary_retained_for_unit_closure_review",
                "pass" if electron_identification_dictionary_ready else "reject",
                "electron-identification dictionary retained for unit-closure review",
                1 if electron_identification_dictionary_ready else 0,
                "The absolute-normalization dictionary remains the m0 anchor while the unit bridge is audited.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "prior_unit_review_route_active": prior_unit_review_route_active,
            "alpha_prediction_review_completed": bool(route_1066["alpha_prediction_review_completed"]),
            "alpha_prediction_unit_mismatch_confirmed": bool(route_1066["alpha_prediction_unit_mismatch_confirmed"]),
            "h0p_mapping_surface_available": alpha_note_has_h0,
            "zpgrav_surface_available": zpem_note_has_zpgrav,
            "prior_unit_consistency_audit_retained": "mass_frequency_bridge_alpha_dimension_vector_si" in unit_audit_984,
            "electron_identification_dictionary_ready": electron_identification_dictionary_ready,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_alpha_prediction_unit_closure_inventory_frozen",
            "advance_to_8_7_56_1068": inventory_ready,
            "next_required_artifacts": [CURRENT_ROUTE],
        },
        {
            "status_hits": {
                "status_next_1067": hit(status_text, "8.7.56.1067"),
                "roadmap_branch_1067": hit(roadmap_text, "`8.7.56.1067-.1070`"),
                "part3a_unit_review_next": hit(part3a_text, PART3A_UNIT_REVIEW_NEXT),
                "part5_unit_scope": hit(part5_text, PART5_UNIT_SCOPE),
            },
            "note_hits": {
                "alpha_note_h0": hit(alpha_note_text, ALPHA_NOTE_H0),
                "alpha_note_v": hit(alpha_note_text, ALPHA_NOTE_V),
                "alpha_note_alpha": hit(alpha_note_text, ALPHA_NOTE_ALPHA),
                "alpha_note_mchi": hit(alpha_note_text, ALPHA_NOTE_MCHI),
                "zpem_note_zpgrav": hit(zpem_note_text, ZPEM_NOTE_GRAV),
            },
        },
    )

    audit = payload(
        "8.7.56.1068",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization alpha-is-prediction unit-closure review audit",
        common_inputs,
        "Audit where unit closure first fails in the alpha-is-prediction chain, and test whether the missing bridge is a simple c^2/hbar-type translation or whether the closed-form alpha formula still remains dimensionally open even after that fix.",
        {
            "newton_mchi_rule": "M_chi^2 = c^4 / (4*pi*G)",
            "zpgrav_rule": "Z_P^grav = M_chi^2 / v^2",
            "h0p_rule": "H0^(P) = m0 / sqrt(Z_P^grav)",
            "v_rule": "v = H0^(P) * M_chi / m0",
            "alpha_rule": "alpha = c^3 / (4*pi*v^2*hbar)",
            "bridge_probe_rule": "if H0^(P) is a frequency while m0 is a mass, the missing RHS bridge must carry dimensions s^-1 kg^-1 = c^2/hbar",
        },
        [
            row(
                "trial2_numeric_alpha_unit_closure_audit_complete",
                "pass" if inventory_ready else "reject",
                "unit-closure audit complete",
                1 if inventory_ready else 0,
                "The branch cuts the note chain into the Newton-Mchi step, the H0^(P) mapping, the v readout, and the alpha formula.",
            ),
            row(
                "trial2_numeric_alpha_h0p_mapping_requires_mass_frequency_bridge",
                "pass" if inferred_zpgrav_dimensionless else "reject",
                "H0^(P) mapping requires mass-frequency bridge",
                1 if inferred_zpgrav_dimensionless else 0,
                "If Z_P^grav is a normalization coefficient and therefore dimensionless, H0^(P) = m0 / sqrt(Z_P^grav) equates a frequency with a mass unless an explicit bridge is inserted.",
            ),
            row(
                "trial2_numeric_alpha_h0p_mapping_bridge_matches_c2_over_hbar_type",
                "pass" if h0p_bridge_matches_c2_over_hbar else "reject",
                "H0^(P) mapping bridge matches c^2/hbar type",
                1 if h0p_bridge_matches_c2_over_hbar else 0,
                "The required RHS bridge dimensions are s^-1 kg^-1, exactly the dimensions of c^2 / hbar.",
            ),
            row(
                "trial2_numeric_alpha_h0p_bridge_restores_v_to_zpgrav_expectation",
                "pass" if corrected_v_matches_expected else "reject",
                "H0^(P) bridge restores v to Z_P^grav expectation",
                1 if corrected_v_matches_expected else 0,
                "Multiplying the current v readout by hbar / c^2 makes v carry the same dimensions as M_chi, which is what Z_P^grav = M_chi^2 / v^2 requires if Z_P^grav is dimensionless.",
            ),
            row(
                "trial2_numeric_alpha_h0p_bridge_alone_resolves_alpha_units",
                "pass" if alpha_h0p_bridge_alone_resolves_units else "reject",
                "H0^(P) bridge alone resolves alpha units",
                1 if alpha_h0p_bridge_alone_resolves_units else 0,
                "Even after the H0^(P) bridge is inserted, the closed-form alpha formula still carries residual dimensions.",
            ),
            row(
                "trial2_numeric_alpha_newton_mchi_direct_mass_readout_admissible_in_si",
                "pass" if mchi_direct_mass_readout_admissible_in_si else "reject",
                "Newton-Mchi direct mass readout admissible in SI",
                1 if mchi_direct_mass_readout_admissible_in_si else 0,
                "The Newton step yields a scalar kinetic-scale readout with dimensions different from pure mass in direct SI.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "mchi_dimension_vector_si": format_dims(mchi_dims),
            "mchi_direct_mass_readout_admissible_in_si": mchi_direct_mass_readout_admissible_in_si,
            "mchi_mass_bridge_required_dimension_vector_si": format_dims(mchi_mass_bridge_required_dims),
            "current_v_dimension_vector_si": format_dims(current_v_dims),
            "expected_v_dimension_vector_if_zpgrav_dimensionless": format_dims(expected_v_dims_if_zpgrav_dimensionless),
            "h0p_mapping_required_rhs_bridge_dimension_vector_si": format_dims(h0p_rhs_bridge_required_dims),
            "c2_over_hbar_dimension_vector_si": format_dims(c2_over_hbar_dims),
            "h0p_mapping_bridge_is_c2_over_hbar_type": h0p_bridge_matches_c2_over_hbar,
            "c2_over_hbar_value_si": c2_over_hbar,
            "hbar_over_c2_value_si": hbar_over_c2,
            "v_bridge_multiplier_dimension_vector_si": format_dims(v_bridge_multiplier_dims),
            "corrected_v_dimension_vector_si": format_dims(corrected_v_dims),
            "corrected_v_matches_zpgrav_expectation": corrected_v_matches_expected,
            "current_alpha_dimension_vector_si": format_dims(current_alpha_dims),
            "alpha_dimension_vector_after_h0p_bridge": format_dims(alpha_dims_after_h0p_bridge),
            "alpha_h0p_bridge_alone_resolves_units": alpha_h0p_bridge_alone_resolves_units,
            "alpha_additional_bridge_required_after_h0p_bridge": not alpha_h0p_bridge_alone_resolves_units,
            "alpha_additional_bridge_required_dimension_vector_si": format_dims(alpha_additional_bridge_required_dims),
            "retained_final_computation_mass_frequency_bridge_alpha_dimension_vector_si": unit_audit_984[
                "mass_frequency_bridge_alpha_dimension_vector_si"
            ],
            "retained_alpha_candidate_value": audit_1064["alpha_candidate_value"],
            "retained_alpha_relative_error": audit_1064["relative_error"],
            "selected_unit_closure_review_class": selected_review_class,
            "first_missing_unit_bridge_location": "h0p_m0_mapping",
            "first_missing_unit_bridge_type": "mass_frequency_bridge_c2_over_hbar_or_equivalent",
            "numeric_closeout_ready": False,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {
            "overall_status": "trial2_numeric_alpha_alpha_prediction_unit_closure_audited",
            "advance_to_8_7_56_1069": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "retained_1064_summary": audit_1064,
            "retained_984_summary": unit_audit_984,
            "note_hits": inventory["evidence"]["note_hits"],
        },
    )

    gate = payload(
        "8.7.56.1069",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization alpha-is-prediction unit-closure review declaration gate",
        common_inputs,
        "Fix the official gate after unit review: the first missing bridge is the H0^(P)-m0 mass-frequency translation, but that bridge alone does not close the alpha formula, so the current pack still lacks an additional unit bridge.",
        {
            "gate_rule": "unit closure is not available under the current pack when the H0^(P) relation needs an explicit c^2/hbar-type bridge and the alpha formula still carries residual dimensions after that probe",
            "next_route_rule": "the next residual route isolates the remaining alpha-formula unit bridge after the first H0^(P) bridge is identified",
        },
        [
            row(
                "trial2_numeric_alpha_unit_closure_gate_complete",
                "pass",
                "unit-closure gate complete",
                1,
                "The official gate is updated after the computation-side unit review.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_unit_closure_possible",
                "pass" if alpha_h0p_bridge_alone_resolves_units else "reject",
                "current pack unit closure possible",
                1 if alpha_h0p_bridge_alone_resolves_units else 0,
                "The current pack would close only if the H0^(P) bridge probe already removed all residual dimensions from alpha.",
            ),
            row(
                "trial2_numeric_alpha_first_missing_unit_bridge_confirmed_at_h0p_mapping",
                "pass" if h0p_bridge_matches_c2_over_hbar else "reject",
                "first missing unit bridge confirmed at H0^(P) mapping",
                1 if h0p_bridge_matches_c2_over_hbar else 0,
                "The first mandatory extra bridge is the mass-frequency translation in H0^(P) = m0 / sqrt(Z_P^grav).",
            ),
            row(
                "trial2_numeric_alpha_alpha_formula_additional_unit_bridge_still_missing",
                "pass" if not alpha_h0p_bridge_alone_resolves_units else "reject",
                "alpha formula additional unit bridge still missing",
                1 if not alpha_h0p_bridge_alone_resolves_units else 0,
                "After the H0^(P) bridge probe, alpha still needs an additional bridge and therefore cannot close under the current pack.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_explicit_unit_bridge_missing",
            "trial2_numeric_alpha_text_search_continuation_justified": False,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": True,
            "trial2_numeric_alpha_unit_closure_review_completed": inventory_ready,
            "trial2_numeric_alpha_unit_closure_possible_under_current_pack": alpha_h0p_bridge_alone_resolves_units,
            "trial2_numeric_alpha_first_missing_unit_bridge_location": "h0p_m0_mapping",
            "trial2_numeric_alpha_first_missing_unit_bridge_type": "mass_frequency_bridge_c2_over_hbar_or_equivalent",
            "trial2_numeric_alpha_h0p_mass_frequency_bridge_alone_resolves_alpha_units": alpha_h0p_bridge_alone_resolves_units,
            "trial2_numeric_alpha_upstream_newton_mchi_direct_mass_readout_admissible_in_si": mchi_direct_mass_readout_admissible_in_si,
            "trial2_numeric_alpha_alpha_formula_additional_unit_bridge_missing": not alpha_h0p_bridge_alone_resolves_units,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "missing_v2_artifact": NEXT_ROUTE_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_alpha_prediction_unit_closure_gate_closed",
            "advance_to_8_7_56_1070": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "retained_1065_summary": gate_1065,
        },
    )

    route = payload(
        "8.7.56.1070",
        "Trial-2 numeric alpha route contract one-hundred-sixty-fourth refresh",
        common_inputs,
        "Refresh the next-generation contract after unit-closure review: the first missing H0^(P) bridge is identified, but full alpha closure remains open, so the next route isolates the remaining alpha-formula unit bridge while keeping the precision-alpha mainline active.",
        {
            "next_route_rule": "the next route isolates the remaining alpha-formula unit bridge after the H0^(P)-m0 mass-frequency bridge is identified as the first mandatory translation",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_sixty_fourth_refresh_complete",
                "pass",
                "route contract one-hundred-sixty-fourth refresh complete",
                1,
                "The unit-closure gate is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_alpha_formula_unit_bridge_review",
                "pass" if not alpha_h0p_bridge_alone_resolves_units else "reject",
                "next route selected as alpha-formula unit-bridge review",
                1 if not alpha_h0p_bridge_alone_resolves_units else 0,
                "The next official branch isolates the remaining alpha-formula unit bridge instead of treating the H0^(P) bridge as sufficient.",
            ),
            row(
                "trial2_numeric_alpha_first_missing_h0p_bridge_retained_as_fixed_result",
                "pass" if h0p_bridge_matches_c2_over_hbar else "reject",
                "first missing H0^(P) bridge retained as fixed result",
                1 if h0p_bridge_matches_c2_over_hbar else 0,
                "The H0^(P)-m0 mass-frequency bridge is retained as fixed diagnostic output from the current branch.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_unit_closure_review",
                "pass" if bool(route_1066.get("precision_alpha_mainline_retained", False)) else "reject",
                "precision-alpha mainline retained after unit-closure review",
                1 if bool(route_1066.get("precision_alpha_mainline_retained", False)) else 0,
                "Trial-2 numeric alpha remains the precision mainline while the remaining alpha-formula unit bridge is reviewed.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "strong_side_route_state": route_1066.get("strong_side_route_state"),
            "precision_alpha_mainline_retained": bool(route_1066.get("precision_alpha_mainline_retained", False)),
            "electron_identification_pivot_retained": bool(route_1066.get("electron_identification_pivot_retained", False)),
            "h0p_bridge_pivot_retained": bool(route_1066.get("h0p_bridge_pivot_retained", False)),
            "final_computation_branch_retained": True,
            "unit_consistency_audit_branch_retained": True,
            "dimensionless_alpha_bridge_branch_retained": True,
            "em_unit_convention_bridge_branch_retained": True,
            "mapping_statement_branch_retained": True,
            "mapping_literal_branch_retained": True,
            "two_sector_hierarchy_pivot_retained": True,
            "current_canon_bridge_statement_checkpoint_wording_promotion_completed": bool(
                route_1066.get("current_canon_bridge_statement_checkpoint_wording_promotion_completed", False)
            ),
            "current_canon_no_go_closeout_candidate_retired": True,
            "alpha_prediction_review_completed": bool(route_1066.get("alpha_prediction_review_completed", False)),
            "alpha_prediction_unit_mismatch_confirmed": bool(route_1066.get("alpha_prediction_unit_mismatch_confirmed", False)),
            "alpha_prediction_unit_closure_review_completed": inventory_ready,
            "first_missing_unit_bridge_location": "h0p_m0_mapping",
            "first_missing_unit_bridge_type": "mass_frequency_bridge_c2_over_hbar_or_equivalent",
            "alpha_formula_additional_unit_bridge_missing": not alpha_h0p_bridge_alone_resolves_units,
            "external_dependency_active": False,
            "hard_conflict_reading_retired": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_sixty_fourth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "gate_summary": gate["summary"],
            "audit_summary": audit["summary"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_unit_closure_review_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_unit_closure_review_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_unit_closure_review_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_fourth_refresh", route)

    print("[done] 8.7.56.1067-.1070 artifacts generated:")
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_unit_closure_review_source_inventory_metrics.json"
    )
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_unit_closure_review_audit_metrics.json"
    )
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_unit_closure_review_declaration_gate_metrics.json"
    )
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_fourth_refresh_metrics.json")


if __name__ == "__main__":
    main()
