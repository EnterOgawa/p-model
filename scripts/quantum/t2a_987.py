#!/usr/bin/env python3
"""Generate 8.7.56.987-.990 Trial-2 numeric alpha dimensionless-alpha bridge artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"

ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_final_computation.md")
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EM_DOC = ROOT / "doc" / "quantum" / "16_electromagnetism_charge_maxwell_photon.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

EM_MINIMAL = OUT / "electromagnetism_minimal_metrics.json"
QED_PRECISION = OUT / "qed_vacuum_precision_metrics.json"
UNIT_SOURCE = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_source_inventory_metrics.json"
UNIT_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_audit_metrics.json"
UNIT_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_declaration_gate_metrics.json"
UNIT_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_third_refresh_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge"
CURRENT_ARTIFACT = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge"
NEXT_ROUTE = "8.7.56.991"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution when a required path is missing.

def require(path: Path) -> None:
    """Require an input path to exist before execution continues."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read a UTF-8 text file.

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read a UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return a stable display path for repo or external files.

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
    """Build a standard metrics row."""
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
    """Build a standard metrics payload."""
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
    """Write a metrics payload as JSON and CSV."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: execute the dimensionless-alpha bridge branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha dimensionless-alpha bridge branch."""
    for path in (
        ADVICE,
        PART1,
        PART2,
        PART3A,
        PART5,
        EM_DOC,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        EM_MINIMAL,
        QED_PRECISION,
        UNIT_SOURCE,
        UNIT_AUDIT,
        UNIT_GATE,
        UNIT_ROUTE,
    ):
        require(path)

    advice_text = read_text(ADVICE)
    part1_text = read_text(PART1)
    part2_text = read_text(PART2)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    em_minimal = read_json(EM_MINIMAL)
    qed_precision = read_json(QED_PRECISION)
    unit_source = read_json(UNIT_SOURCE)["summary"]
    unit_audit = read_json(UNIT_AUDIT)["summary"]
    unit_gate = read_json(UNIT_GATE)["summary"]
    unit_route = read_json(UNIT_ROUTE)["summary"]

    prior_route_active = (
        unit_gate["selected_residual_route"] == CURRENT_ROUTE
        and unit_gate["missing_v2_artifact"] == CURRENT_ARTIFACT
        and unit_route["selected_next_generation_route"] == CURRENT_ROUTE
    )

    advice_has_final_formula = hit(advice_text, r"\alpha = \frac{4\pi G^2 Z_P}{\hbar c}") is not None
    part1_has_weak_field_normalization = hit(part1_text, r"g_P/Z_P=4\pi G") is not None
    part1_has_electron_identification = hit(part1_text, r"m_0 = \frac{m_e}{\mathcal{E}(\beta_1)}") is not None
    part2_has_h0p_background_law = hit(part2_text, r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]") is not None
    part3a_has_structural_charge_rule = hit(part3a_text, r"e=g_P/\sqrt{Z_P}") is not None
    part3a_has_structural_alpha_rule = hit(part3a_text, r"\alpha=g_P^2/(4\pi Z_P\hbar c)") is not None
    part3a_has_direct_si_alpha_rule = hit(part3a_text, r"\alpha=\frac{4\pi G^2 Z_P}{\hbar c}") is not None
    part5_has_current_dimensionless_bridge_wording = hit(part5_text, "missing dimensionless-α bridge") is not None
    em_doc_has_eps0_coulomb_surface = hit(em_doc_text, r"\varepsilon_0") is not None
    em_doc_has_coulomb_rule = hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}") is not None
    em_metrics_keep_local_maxwell_qed = (
        "Local Maxwell/QED is kept unchanged at this stage."
        in json.dumps(em_minimal, ensure_ascii=False)
    )
    elementary_charge_available = "e_charge_c" in qed_precision["constants_si"]
    status_has_next_987 = hit(status_text, "8.7.56.987") is not None
    roadmap_has_987_branch = hit(roadmap_text, "`8.7.56.987-.990`") is not None

    explicit_si_alpha_formula_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, r"\alpha=\frac{e^2}{4\pi\varepsilon_0\hbar c}"),
            hit(part3a_text, r"\alpha=\frac{e^2}{4\pi\varepsilon_0\hbar c}"),
            hit(part5_text, r"\alpha=\frac{e^2}{4\pi\varepsilon_0\hbar c}"),
            hit(em_doc_text, r"\alpha=\frac{e^2}{4\pi\varepsilon_0\hbar c}"),
        )
    )
    explicit_gp_to_elementary_charge_mapping_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, r"g_P=e\sqrt{Z_P}"),
            hit(part3a_text, r"g_P=e\sqrt{Z_P}"),
            hit(part5_text, r"g_P=e\sqrt{Z_P}"),
            hit(part1_text, "elementary charge"),
            hit(part3a_text, "elementary charge"),
            hit(part5_text, "elementary charge"),
        )
    )
    explicit_em_unit_convention_bridge_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, r"\varepsilon_0=1"),
            hit(part3a_text, r"\varepsilon_0=1"),
            hit(part5_text, r"\varepsilon_0=1"),
            hit(part1_text, "Heaviside-Lorentz"),
            hit(part3a_text, "Heaviside-Lorentz"),
            hit(part5_text, "Heaviside-Lorentz"),
            hit(part1_text, "rationalized electromagnetic units"),
            hit(part3a_text, "rationalized electromagnetic units"),
            hit(part5_text, "rationalized electromagnetic units"),
        )
    )

    inventory_ready = all(
        [
            bool(unit_source["inventory_ready"]),
            bool(unit_audit["audit_ready"]),
            bool(unit_gate["trial2_numeric_alpha_final_computation_performed"]),
            prior_route_active,
            advice_has_final_formula,
            part1_has_weak_field_normalization,
            part1_has_electron_identification,
            part2_has_h0p_background_law,
            part3a_has_structural_charge_rule,
            part3a_has_structural_alpha_rule,
            part3a_has_direct_si_alpha_rule,
            part5_has_current_dimensionless_bridge_wording,
            em_doc_has_eps0_coulomb_surface,
            em_doc_has_coulomb_rule,
            em_metrics_keep_local_maxwell_qed,
            elementary_charge_available,
            status_has_next_987,
            roadmap_has_987_branch,
        ]
    )

    explicit_dimensionless_alpha_bridge_available = any(
        [
            explicit_si_alpha_formula_available,
            explicit_gp_to_elementary_charge_mapping_available,
            explicit_em_unit_convention_bridge_available,
        ]
    )
    dominant_blocker_is_em_unit_convention = not explicit_dimensionless_alpha_bridge_available

    common_inputs = {
        "expert_note_markdown": display_path(ADVICE),
        "part1_markdown": display_path(PART1),
        "part2_markdown": display_path(PART2),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "electromagnetism_doc_markdown": display_path(EM_DOC),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "electromagnetism_minimal_metrics_json": display_path(EM_MINIMAL),
        "qed_vacuum_precision_metrics_json": display_path(QED_PRECISION),
        "unit_consistency_source_json": display_path(UNIT_SOURCE),
        "unit_consistency_audit_json": display_path(UNIT_AUDIT),
        "unit_consistency_gate_json": display_path(UNIT_GATE),
        "unit_consistency_route_json": display_path(UNIT_ROUTE),
    }

    inventory = payload(
        "8.7.56.987",
        "Trial-2 numeric alpha final-computation dimensionless-alpha bridge source inventory",
        common_inputs,
        "Freeze the dimensionless-alpha bridge pack: the structural e=g_P/sqrt(Z_P) and alpha=g_P^2/(4*pi*Z_P*hbar*c) route, the direct SI H0^(P)-Z_P substitution, the adopted Maxwell/QED sector, and the direct SI unit audit that showed the raw candidate is not dimensionless.",
        {
            "structural_alpha_rule": "e = g_P / sqrt(Z_P), alpha = g_P^2 / (4*pi*Z_P*hbar*c)",
            "direct_si_probe": "alpha_direct = 4*pi*G^2*Z_P / (hbar*c)",
            "bridge_rule": "an honest closeout needs an explicit bridge from the structural EM coupling convention to the direct SI substitution",
        },
        [
            row(
                "trial2_numeric_alpha_dimensionless_bridge_inventory_complete",
                "pass" if inventory_ready else "reject",
                "dimensionless-alpha bridge input-pack inventory complete",
                1 if inventory_ready else 0,
                "The bridge audit needs the structural alpha route, the adopted EM sector, the direct SI unit audit, and the current route contract in one pack.",
            ),
            row(
                "trial2_numeric_alpha_structural_charge_rule_available",
                "pass" if part3a_has_structural_charge_rule else "reject",
                "structural charge rule e=g_P/sqrt(Z_P) available",
                1 if part3a_has_structural_charge_rule else 0,
                "The structural alpha route already carries a coupling e, but the SI normalization of that coupling is not yet explicit.",
            ),
            row(
                "trial2_numeric_alpha_structural_alpha_rule_available",
                "pass" if part3a_has_structural_alpha_rule else "reject",
                "structural alpha rule g_P^2/(4*pi*Z_P*hbar*c) available",
                1 if part3a_has_structural_alpha_rule else 0,
                "The dimensionless-alpha bridge audit starts from the structural Maxwell/Coulomb alpha formula, not from raw SI substitution alone.",
            ),
            row(
                "trial2_numeric_alpha_em_coulomb_eps0_surface_available",
                "pass" if em_doc_has_eps0_coulomb_surface and em_doc_has_coulomb_rule else "reject",
                "EM Coulomb eps0 surface available",
                1 if em_doc_has_eps0_coulomb_surface and em_doc_has_coulomb_rule else 0,
                "The public EM minimal note retains the Coulomb sector with explicit eps0, so an SI EM-unit bridge is in scope.",
            ),
            row(
                "trial2_numeric_alpha_dimensionless_alpha_bridge_still_required",
                "pass" if not explicit_dimensionless_alpha_bridge_available else "reject",
                "dimensionless-alpha bridge still required",
                1 if not explicit_dimensionless_alpha_bridge_available else 0,
                "No explicit source in the current pack yet states how the structural EM coupling convention becomes the direct SI alpha candidate.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "structural_charge_rule_available": part3a_has_structural_charge_rule,
            "structural_alpha_rule_available": part3a_has_structural_alpha_rule,
            "direct_si_alpha_rule_available": part3a_has_direct_si_alpha_rule,
            "em_coulomb_eps0_surface_available": em_doc_has_eps0_coulomb_surface and em_doc_has_coulomb_rule,
            "local_maxwell_qed_adopted_surface_available": em_metrics_keep_local_maxwell_qed,
            "elementary_charge_constant_available": elementary_charge_available,
            "direct_si_alpha_dimension_vector_si": unit_audit["raw_alpha_candidate_dimension_vector_si"],
            "explicit_dimensionless_alpha_bridge_available": explicit_dimensionless_alpha_bridge_available,
            "first_route_to_close_or_none": NEXT_MISSING_ARTIFACT,
        },
        {
            "overall_status": "trial2_numeric_alpha_dimensionless_alpha_bridge_inventory_frozen",
            "advance_to_8_7_56_988": inventory_ready,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "ai_context_current_step": ai_context["current_step"],
            "part1_weak_field_normalization_hit": hit(part1_text, r"g_P/Z_P=4\pi G"),
            "part1_electron_identification_hit": hit(part1_text, r"m_0 = \frac{m_e}{\mathcal{E}(\beta_1)}"),
            "part3a_structural_charge_rule_hit": hit(part3a_text, r"e=g_P/\sqrt{Z_P}"),
            "part3a_structural_alpha_rule_hit": hit(part3a_text, r"\alpha=g_P^2/(4\pi Z_P\hbar c)"),
            "em_doc_eps0_hit": hit(em_doc_text, r"\varepsilon_0"),
            "em_doc_coulomb_hit": hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}"),
        },
    )

    audit = payload(
        "8.7.56.988",
        "Trial-2 numeric alpha final-computation dimensionless-alpha bridge audit",
        common_inputs,
        "Audit whether current canon explicitly bridges the structural EM coupling convention to the direct SI H0^(P)-Z_P substitution, or whether the honest blocker is a missing EM unit-convention / charge-normalization statement.",
        {
            "structural_rule": "alpha_structural = g_P^2 / (4*pi*Z_P*hbar*c), with e = g_P / sqrt(Z_P)",
            "si_probe_rule": "alpha_direct = 4*pi*G^2*Z_P / (hbar*c) carries raw SI dimensions kg^-1 m^3",
            "audit_rule": "closeout requires an explicit statement that maps the structural EM coupling into the direct SI substitution without losing dimensionlessness",
        },
        [
            row(
                "trial2_numeric_alpha_dimensionless_bridge_audit_complete",
                "pass",
                "dimensionless-alpha bridge audit complete",
                1,
                "The current branch evaluates whether the missing bridge is present as an explicit canon surface.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_structural_em_alpha_route",
                "pass" if part3a_has_structural_charge_rule and part3a_has_structural_alpha_rule else "reject",
                "current pack contains structural EM alpha route",
                1 if part3a_has_structural_charge_rule and part3a_has_structural_alpha_rule else 0,
                "The structural route exists and is not the blocker by itself.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_explicit_si_alpha_formula",
                "pass" if explicit_si_alpha_formula_available else "reject",
                "current pack contains explicit SI alpha formula",
                1 if explicit_si_alpha_formula_available else 0,
                "No public source currently states the SI-normalized alpha formula that would bridge the structural coupling to the direct SI substitution.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_explicit_gp_to_elementary_charge_mapping",
                "pass" if explicit_gp_to_elementary_charge_mapping_available else "reject",
                "current pack contains explicit g_P to elementary-charge mapping",
                1 if explicit_gp_to_elementary_charge_mapping_available else 0,
                "The public pack has e_charge_c and the structural e=g_P/sqrt(Z_P) rule separately, but not one explicit bridge between them.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_explicit_em_unit_convention_bridge",
                "pass" if explicit_em_unit_convention_bridge_available else "reject",
                "current pack contains explicit EM unit-convention bridge",
                1 if explicit_em_unit_convention_bridge_available else 0,
                "No public source explicitly states the rationalized / SI bridge that would keep alpha dimensionless after the G-based substitution.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_missing_em_unit_convention",
                "pass" if dominant_blocker_is_em_unit_convention else "reject",
                "dominant blocker is missing EM unit-convention / charge-normalization bridge",
                1 if dominant_blocker_is_em_unit_convention else 0,
                "The honest blocker is no longer a generic unit question; it is the absent statement that maps the structural EM coupling convention into the direct SI alpha readout.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "structural_em_alpha_route_available": part3a_has_structural_charge_rule and part3a_has_structural_alpha_rule,
            "explicit_si_alpha_formula_available": explicit_si_alpha_formula_available,
            "explicit_gp_to_elementary_charge_mapping_available": explicit_gp_to_elementary_charge_mapping_available,
            "explicit_em_unit_convention_bridge_available": explicit_em_unit_convention_bridge_available,
            "explicit_dimensionless_alpha_bridge_available": explicit_dimensionless_alpha_bridge_available,
            "raw_alpha_candidate_dimension_vector_si": unit_audit["raw_alpha_candidate_dimension_vector_si"],
            "dominant_blocker_is_missing_em_unit_convention": dominant_blocker_is_em_unit_convention,
            "first_route_to_close_after_audit_or_none": NEXT_MISSING_ARTIFACT,
        },
        {
            "overall_status": "trial2_numeric_alpha_dimensionless_alpha_bridge_audited",
            "advance_to_8_7_56_989": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "unit_consistency_summary": unit_audit,
            "em_minimal_postulates": em_minimal["postulates"],
            "em_minimal_open_problems": em_minimal["open_problems"],
            "qed_constants_available": {
                "e_charge_c": qed_precision["constants_si"]["e_charge_c"],
                "hbar_j_s": qed_precision["constants_si"]["hbar_j_s"],
                "c_m_per_s": qed_precision["constants_si"]["c_m_per_s"],
            },
        },
    )

    gate = payload(
        "8.7.56.989",
        "Trial-2 numeric alpha final-computation dimensionless-alpha bridge declaration gate",
        common_inputs,
        "Fix the official gate after the bridge audit: the raw final-computation value remains evidence, but closeout still depends on an explicit EM unit-convention / charge-normalization bridge that current canon does not yet provide.",
        {
            "gate_rule": "a numeric alpha candidate cannot become an honest closeout until the structural EM coupling convention is explicitly bridged into the direct SI substitution",
            "residual_rule": "the next blocker is an explicit EM unit-convention / charge-normalization bridge statement",
        },
        [
            row(
                "trial2_numeric_alpha_dimensionless_bridge_gate_complete",
                "pass",
                "dimensionless-alpha bridge declaration gate complete",
                1,
                "The official state is updated after the bridge audit.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_bridge_audit",
                "pass" if explicit_dimensionless_alpha_bridge_available else "reject",
                "numeric alpha from current pack ready after bridge audit",
                1 if explicit_dimensionless_alpha_bridge_available else 0,
                "Without an explicit EM unit bridge, the current direct SI readout is still only a pre-canonical diagnostic.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready_after_bridge_audit",
                "pass" if explicit_dimensionless_alpha_bridge_available else "reject",
                "Trial-2 numeric alpha closeout ready after bridge audit",
                1 if explicit_dimensionless_alpha_bridge_available else 0,
                "Closeout remains blocked while the EM unit-convention bridge is absent.",
            ),
            row(
                "trial2_numeric_alpha_result_class_retained_as_precanonical_unit_incomplete",
                "pass" if dominant_blocker_is_em_unit_convention else "reject",
                "result class retained as pre-canonical unit incomplete",
                1 if dominant_blocker_is_em_unit_convention else 0,
                "The missing bridge is still upstream of any final physical accept/reject on alpha.",
            ),
            row(
                "trial2_numeric_alpha_current_blocker_is_em_unit_convention_bridge",
                "pass" if dominant_blocker_is_em_unit_convention else "reject",
                "current blocker is EM unit-convention / charge-normalization bridge",
                1 if dominant_blocker_is_em_unit_convention else 0,
                "The honest next artifact is the explicit bridge that keeps alpha dimensionless when the structural EM route is read in direct SI terms.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": True,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": True,
            "trial2_numeric_alpha_raw_final_computation_value_available": bool(
                unit_gate["trial2_numeric_alpha_raw_final_computation_value_available"]
            ),
            "trial2_numeric_alpha_numeric_from_current_pack_ready": explicit_dimensionless_alpha_bridge_available,
            "trial2_numeric_alpha_closeout_ready": explicit_dimensionless_alpha_bridge_available,
            "trial2_numeric_alpha_final_computation_performed": bool(
                unit_gate["trial2_numeric_alpha_final_computation_performed"]
            ),
            "trial2_numeric_alpha_final_computation_result_class": "precanonical_unit_incomplete",
            "trial2_numeric_alpha_retry_loop_retired": bool(unit_gate["trial2_numeric_alpha_retry_loop_retired"]),
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_dimensionless_alpha_bridge_gate_closed",
            "advance_to_8_7_56_990": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "bridge_audit_summary": audit["summary"],
            "unit_consistency_gate_summary": unit_gate,
            "unit_consistency_route_summary": unit_route,
        },
    )

    route = payload(
        "8.7.56.990",
        "Trial-2 numeric alpha route contract one-hundred-forty-fourth refresh",
        common_inputs,
        "Refresh the next-generation contract after the dimensionless-alpha bridge audit: keep Trial-2 numeric alpha on the precision mainline, keep the strong side on reserve, and promote the EM unit-convention bridge as the next official blocker family.",
        {
            "next_route_rule": "the next route must determine whether current canon contains an explicit EM unit-convention / charge-normalization bridge",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_forty_fourth_refresh_complete",
                "pass",
                "route contract one-hundred-forty-fourth refresh complete",
                1,
                "The dimensionless-alpha bridge audit is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_em_unit_convention_bridge",
                "pass" if dominant_blocker_is_em_unit_convention else "reject",
                "next route selected as EM unit-convention bridge",
                1 if dominant_blocker_is_em_unit_convention else 0,
                "The generic dimensionless-alpha bridge has been narrowed to the missing EM unit-convention / charge-normalization bridge.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_bridge_audit",
                "pass" if unit_route["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after bridge audit",
                1 if unit_route["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline despite the unresolved bridge.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_bridge_audit",
                "pass" if unit_route["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after bridge audit",
                1 if unit_route["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains on reserve and is not promoted by the EM-unit bridge audit.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": unit_route["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(unit_route["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(unit_route["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(unit_route["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(unit_route["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": bool(unit_route["unit_consistency_audit_branch_retained"]),
            "dimensionless_alpha_bridge_branch_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_forty_fourth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "gate_summary": gate["summary"],
            "unit_consistency_route_summary": unit_route,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_fourth_refresh",
        route,
    )

    print("[done] 8.7.56.987-.990 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_fourth_refresh_metrics.json")
    print(f" - dominant_blocker = {NEXT_MISSING_ARTIFACT}")


# Function: run the dimensionless-alpha bridge branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha dimensionless-alpha bridge branch."""
    main()


if __name__ == "__main__":
    run_cli()
