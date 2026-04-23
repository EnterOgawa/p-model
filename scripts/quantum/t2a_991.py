#!/usr/bin/env python3
"""Generate 8.7.56.991-.994 Trial-2 numeric alpha EM unit-convention bridge artifacts."""

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
DIM_SOURCE = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_source_inventory_metrics.json"
DIM_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_audit_metrics.json"
DIM_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_declaration_gate_metrics.json"
DIM_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_fourth_refresh_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention"
CURRENT_ARTIFACT = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention"
NEXT_ROUTE = "8.7.56.995"
NEXT_RESIDUAL_ROUTE = (
    "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_statement"
)
NEXT_MISSING_ARTIFACT = NEXT_RESIDUAL_ROUTE


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


# Function: execute the EM unit-convention / charge-normalization bridge branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha EM unit-convention / charge-normalization bridge branch."""
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
        DIM_SOURCE,
        DIM_AUDIT,
        DIM_GATE,
        DIM_ROUTE,
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
    dim_source = read_json(DIM_SOURCE)["summary"]
    dim_audit = read_json(DIM_AUDIT)["summary"]
    dim_gate = read_json(DIM_GATE)["summary"]
    dim_route = read_json(DIM_ROUTE)["summary"]

    prior_route_active = (
        dim_gate["selected_residual_route"] == CURRENT_ROUTE
        and dim_gate["missing_v2_artifact"] == CURRENT_ARTIFACT
        and dim_route["selected_next_generation_route"] == CURRENT_ROUTE
    )

    advice_has_final_formula = hit(advice_text, r"\alpha = \frac{4\pi G^2 Z_P}{\hbar c}") is not None
    part1_has_weak_field_normalization = hit(part1_text, r"g_P/Z_P=4\pi G") is not None
    part1_has_electron_identification = hit(part1_text, r"m_0 = \frac{m_e}{\mathcal{E}(\beta_1)}") is not None
    part2_has_h0p_background_law = hit(part2_text, r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]") is not None
    part3a_has_structural_charge_rule = hit(part3a_text, r"e=g_P/\sqrt{Z_P}") is not None
    part3a_has_structural_alpha_rule = hit(part3a_text, r"\alpha=g_P^2/(4\pi Z_P\hbar c)") is not None
    part3a_has_em_bridge_wording = hit(part3a_text, "EM unit-convention / charge-normalization bridge") is not None
    part5_has_em_bridge_wording = hit(part5_text, "EM unit-convention bridge missing") is not None
    em_doc_has_eps0_surface = hit(em_doc_text, r"\varepsilon_0") is not None
    em_doc_has_coulomb_q_surface = hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}") is not None
    em_metrics_keep_local_maxwell_qed = (
        "Local Maxwell/QED is kept unchanged at this stage."
        in json.dumps(em_minimal, ensure_ascii=False)
    )
    elementary_charge_constant_available = "e_charge_c" in qed_precision["constants_si"]
    qed_target_available = "g2" in qed_precision["alpha_precision"]
    status_has_next_991 = hit(status_text, "8.7.56.991") is not None
    roadmap_has_991_branch = hit(roadmap_text, "`8.7.56.991-.994`") is not None

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
            hit(part1_text, r"e_{\mathrm{phys}}=g_P/\sqrt{Z_P}"),
            hit(part3a_text, r"e_{\mathrm{phys}}=g_P/\sqrt{Z_P}"),
            hit(part5_text, r"e_{\mathrm{phys}}=g_P/\sqrt{Z_P}"),
            hit(part1_text, r"e_{\mathrm{SI}}=g_P/\sqrt{Z_P}"),
            hit(part3a_text, r"e_{\mathrm{SI}}=g_P/\sqrt{Z_P}"),
            hit(part5_text, r"e_{\mathrm{SI}}=g_P/\sqrt{Z_P}"),
        )
    )
    explicit_em_unit_convention_bridge_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, "Heaviside-Lorentz"),
            hit(part3a_text, "Heaviside-Lorentz"),
            hit(part5_text, "Heaviside-Lorentz"),
            hit(part1_text, "rationalized electromagnetic units"),
            hit(part3a_text, "rationalized electromagnetic units"),
            hit(part5_text, "rationalized electromagnetic units"),
        )
    )

    dominant_blocker_is_missing_gp_to_elementary_charge_mapping = (
        part3a_has_structural_charge_rule
        and elementary_charge_constant_available
        and not explicit_gp_to_elementary_charge_mapping_available
    )

    inventory_ready = all(
        [
            bool(dim_source["inventory_ready"]),
            bool(dim_audit["audit_ready"]),
            prior_route_active,
            advice_has_final_formula,
            part1_has_weak_field_normalization,
            part1_has_electron_identification,
            part2_has_h0p_background_law,
            part3a_has_structural_charge_rule,
            part3a_has_structural_alpha_rule,
            part3a_has_em_bridge_wording,
            part5_has_em_bridge_wording,
            em_doc_has_eps0_surface,
            em_doc_has_coulomb_q_surface,
            em_metrics_keep_local_maxwell_qed,
            elementary_charge_constant_available,
            qed_target_available,
            status_has_next_991,
            roadmap_has_991_branch,
        ]
    )

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
        "dimensionless_bridge_source_json": display_path(DIM_SOURCE),
        "dimensionless_bridge_audit_json": display_path(DIM_AUDIT),
        "dimensionless_bridge_gate_json": display_path(DIM_GATE),
        "dimensionless_bridge_route_json": display_path(DIM_ROUTE),
    }

    inventory = payload(
        "8.7.56.991",
        "Trial-2 numeric alpha final-computation EM unit-convention bridge source inventory",
        common_inputs,
        "Freeze the EM unit-convention / charge-normalization bridge pack: the structural e=g_P/sqrt(Z_P) route, the public Coulomb eps0 surface, the adopted Maxwell/QED stance, the public elementary-charge constant, and the direct SI audit that still lacks an explicit bridge into a dimensionless alpha.",
        {
            "structural_rule": "e = g_P / sqrt(Z_P), alpha = g_P^2 / (4*pi*Z_P*hbar*c)",
            "public_em_surface": "Phi(r) = q / (4*pi*eps0*r), |E(r)| = |q| / (4*pi*eps0*r^2)",
            "inventory_rule": "the minimal bridge pack must contain both the structural e and the public elementary-charge surface before an honest SI alpha formula can be claimed",
        },
        [
            row(
                "trial2_numeric_alpha_em_unit_bridge_inventory_complete",
                "pass" if inventory_ready else "reject",
                "EM unit-convention bridge input-pack inventory complete",
                1 if inventory_ready else 0,
                "This branch needs the structural EM route, the public Coulomb charge surface, the adopted Maxwell/QED stance, the CODATA elementary-charge constant, and the prior dimensionless-alpha audit in one pack.",
            ),
            row(
                "trial2_numeric_alpha_structural_charge_rule_available_for_em_bridge",
                "pass" if part3a_has_structural_charge_rule else "reject",
                "structural charge rule e=g_P/sqrt(Z_P) available for EM bridge",
                1 if part3a_has_structural_charge_rule else 0,
                "The bridge starts from the structural e that already appears in the Trial-2 Maxwell route.",
            ),
            row(
                "trial2_numeric_alpha_public_coulomb_charge_surface_available",
                "pass" if em_doc_has_eps0_surface and em_doc_has_coulomb_q_surface else "reject",
                "public Coulomb charge surface available",
                1 if em_doc_has_eps0_surface and em_doc_has_coulomb_q_surface else 0,
                "The public EM note already fixes q and eps0 in the Coulomb sector.",
            ),
            row(
                "trial2_numeric_alpha_public_elementary_charge_constant_available",
                "pass" if elementary_charge_constant_available else "reject",
                "public elementary-charge constant available",
                1 if elementary_charge_constant_available else 0,
                "The QED precision pack already carries the CODATA elementary-charge constant in SI units.",
            ),
            row(
                "trial2_numeric_alpha_explicit_gp_to_elementary_charge_mapping_still_missing",
                "pass" if not explicit_gp_to_elementary_charge_mapping_available else "reject",
                "explicit g_P to elementary-charge mapping still missing",
                1 if not explicit_gp_to_elementary_charge_mapping_available else 0,
                "The current pack contains structural e and physical e separately, but not one explicit public statement identifying them.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "structural_charge_rule_available": part3a_has_structural_charge_rule,
            "structural_alpha_rule_available": part3a_has_structural_alpha_rule,
            "public_coulomb_charge_surface_available": em_doc_has_eps0_surface and em_doc_has_coulomb_q_surface,
            "local_maxwell_qed_adopted_surface_available": em_metrics_keep_local_maxwell_qed,
            "elementary_charge_constant_available": elementary_charge_constant_available,
            "explicit_gp_to_elementary_charge_mapping_available": explicit_gp_to_elementary_charge_mapping_available,
            "explicit_si_alpha_formula_available": explicit_si_alpha_formula_available,
            "explicit_em_unit_convention_bridge_available": explicit_em_unit_convention_bridge_available,
            "first_route_to_close_or_none": NEXT_MISSING_ARTIFACT,
        },
        {
            "overall_status": "trial2_numeric_alpha_em_unit_bridge_inventory_frozen",
            "advance_to_8_7_56_992": inventory_ready,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "ai_context_current_step": ai_context["current_step"],
            "part3a_structural_charge_rule_hit": hit(part3a_text, r"e=g_P/\sqrt{Z_P}"),
            "em_doc_coulomb_q_hit": hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}"),
            "em_doc_eps0_hit": hit(em_doc_text, r"\varepsilon_0"),
            "qed_constants_available": {
                "e_charge_c": qed_precision["constants_si"]["e_charge_c"],
                "hbar_j_s": qed_precision["constants_si"]["hbar_j_s"],
                "c_m_per_s": qed_precision["constants_si"]["c_m_per_s"],
            },
        },
    )

    audit = payload(
        "8.7.56.992",
        "Trial-2 numeric alpha final-computation EM unit-convention bridge audit",
        common_inputs,
        "Audit whether current canon explicitly identifies the structural e=g_P/sqrt(Z_P) coupling with the public elementary charge entering the Coulomb/QED sector, or whether that charge-normalization statement is still absent.",
        {
            "structural_rule": "e_structural = g_P / sqrt(Z_P)",
            "public_em_rule": "Phi(r) = q / (4*pi*eps0*r), alpha_SI = e^2 / (4*pi*eps0*hbar*c)",
            "audit_rule": "a dimensionless alpha bridge needs an explicit public statement that the structural e is the physical elementary charge used in the SI/QED surface",
        },
        [
            row(
                "trial2_numeric_alpha_em_unit_bridge_audit_complete",
                "pass",
                "EM unit-convention / charge-normalization bridge audit complete",
                1,
                "The current branch audits whether the structural e is explicitly connected to the public elementary-charge surface.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_structural_charge_surface",
                "pass" if part3a_has_structural_charge_rule else "reject",
                "current pack contains structural charge surface",
                1 if part3a_has_structural_charge_rule else 0,
                "The structural Trial-2 route already introduces e through g_P and Z_P.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_public_elementary_charge_surface",
                "pass" if elementary_charge_constant_available else "reject",
                "current pack contains public elementary-charge surface",
                1 if elementary_charge_constant_available else 0,
                "The CODATA elementary charge is already fixed in the QED precision pack.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_explicit_gp_to_elementary_charge_mapping",
                "pass" if explicit_gp_to_elementary_charge_mapping_available else "reject",
                "current pack contains explicit g_P to elementary-charge mapping",
                1 if explicit_gp_to_elementary_charge_mapping_available else 0,
                "The audit finds no public statement that the structural e is the physical elementary charge entering Coulomb/QED formulas.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_explicit_si_alpha_formula_after_charge_mapping",
                "pass" if explicit_si_alpha_formula_available else "reject",
                "current pack contains explicit SI alpha formula after charge mapping",
                1 if explicit_si_alpha_formula_available else 0,
                "Without the charge-normalization statement, the SI alpha formula cannot be claimed as the honest closeout formula for the current route.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_missing_gp_to_elementary_charge_mapping",
                "pass" if dominant_blocker_is_missing_gp_to_elementary_charge_mapping else "reject",
                "dominant blocker is missing g_P to elementary-charge mapping statement",
                1 if dominant_blocker_is_missing_gp_to_elementary_charge_mapping else 0,
                "The minimal missing public surface is the explicit charge-normalization statement that identifies structural e with the physical elementary charge.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "structural_charge_surface_available": part3a_has_structural_charge_rule,
            "public_coulomb_charge_surface_available": em_doc_has_eps0_surface and em_doc_has_coulomb_q_surface,
            "public_elementary_charge_surface_available": elementary_charge_constant_available,
            "explicit_gp_to_elementary_charge_mapping_available": explicit_gp_to_elementary_charge_mapping_available,
            "explicit_si_alpha_formula_available": explicit_si_alpha_formula_available,
            "explicit_em_unit_convention_bridge_available": explicit_em_unit_convention_bridge_available,
            "dominant_blocker_is_missing_gp_to_elementary_charge_mapping": (
                dominant_blocker_is_missing_gp_to_elementary_charge_mapping
            ),
            "first_route_to_close_after_audit_or_none": NEXT_MISSING_ARTIFACT,
        },
        {
            "overall_status": "trial2_numeric_alpha_em_unit_bridge_audited",
            "advance_to_8_7_56_993": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "dimensionless_bridge_summary": dim_audit,
            "em_minimal_postulates": em_minimal["postulates"],
            "qed_alpha_target": qed_precision["alpha_precision"]["g2"],
        },
    )

    gate = payload(
        "8.7.56.993",
        "Trial-2 numeric alpha final-computation EM unit-convention bridge declaration gate",
        common_inputs,
        "Update the official gate after the EM unit-convention audit: the structural and public EM surfaces both exist, but closeout still depends on one explicit charge-normalization statement that current canon does not yet provide.",
        {
            "gate_rule": "a direct SI alpha readout cannot be an honest closeout until the structural e is explicitly identified with the physical elementary charge used in the public Coulomb/QED surface",
            "residual_rule": "the next blocker is the missing g_P-to-elementary-charge mapping statement",
        },
        [
            row(
                "trial2_numeric_alpha_em_unit_bridge_gate_complete",
                "pass",
                "EM unit-convention / charge-normalization bridge declaration gate complete",
                1,
                "The official state is updated after the charge-normalization audit.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_em_unit_bridge_audit",
                "pass" if explicit_gp_to_elementary_charge_mapping_available else "reject",
                "numeric alpha from current pack ready after EM unit bridge audit",
                1 if explicit_gp_to_elementary_charge_mapping_available else 0,
                "Without the explicit charge-normalization statement, the direct SI readout remains pre-canonical.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready_after_em_unit_bridge_audit",
                "pass" if explicit_gp_to_elementary_charge_mapping_available else "reject",
                "Trial-2 numeric alpha closeout ready after EM unit bridge audit",
                1 if explicit_gp_to_elementary_charge_mapping_available else 0,
                "Closeout remains blocked while the structural e is not explicitly identified with physical elementary charge.",
            ),
            row(
                "trial2_numeric_alpha_result_class_retained_as_precanonical_unit_incomplete_after_em_unit_bridge_audit",
                "pass" if dominant_blocker_is_missing_gp_to_elementary_charge_mapping else "reject",
                "result class retained as pre-canonical unit incomplete after EM unit bridge audit",
                1 if dominant_blocker_is_missing_gp_to_elementary_charge_mapping else 0,
                "The missing charge-normalization statement remains upstream of any final numeric pass or reject.",
            ),
            row(
                "trial2_numeric_alpha_current_blocker_is_gp_to_elementary_charge_mapping_statement",
                "pass" if dominant_blocker_is_missing_gp_to_elementary_charge_mapping else "reject",
                "current blocker is g_P to elementary-charge mapping statement",
                1 if dominant_blocker_is_missing_gp_to_elementary_charge_mapping else 0,
                "The generic EM unit-convention blocker has been narrowed to the explicit statement that maps structural e into the public elementary-charge surface.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": True,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": True,
            "trial2_numeric_alpha_raw_final_computation_value_available": bool(
                dim_gate["trial2_numeric_alpha_raw_final_computation_value_available"]
            ),
            "trial2_numeric_alpha_numeric_from_current_pack_ready": explicit_gp_to_elementary_charge_mapping_available,
            "trial2_numeric_alpha_closeout_ready": explicit_gp_to_elementary_charge_mapping_available,
            "trial2_numeric_alpha_final_computation_performed": bool(
                dim_gate["trial2_numeric_alpha_final_computation_performed"]
            ),
            "trial2_numeric_alpha_final_computation_result_class": "precanonical_unit_incomplete",
            "trial2_numeric_alpha_retry_loop_retired": bool(dim_gate["trial2_numeric_alpha_retry_loop_retired"]),
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_em_unit_bridge_gate_closed",
            "advance_to_8_7_56_994": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "em_unit_bridge_audit_summary": audit["summary"],
            "dimensionless_bridge_gate_summary": dim_gate,
            "dimensionless_bridge_route_summary": dim_route,
        },
    )

    route = payload(
        "8.7.56.994",
        "Trial-2 numeric alpha route contract one-hundred-forty-fifth refresh",
        common_inputs,
        "Refresh the next-generation contract after the EM unit-convention / charge-normalization bridge audit: keep Trial-2 numeric alpha on the precision mainline, keep the strong side on reserve, and promote the missing g_P-to-elementary-charge mapping statement as the next official blocker family.",
        {
            "next_route_rule": "the next route must determine whether current canon explicitly identifies the structural e with the public elementary charge",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_forty_fifth_refresh_complete",
                "pass",
                "route contract one-hundred-forty-fifth refresh complete",
                1,
                "The EM unit-convention bridge audit is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_gp_to_elementary_charge_mapping_statement",
                "pass" if dominant_blocker_is_missing_gp_to_elementary_charge_mapping else "reject",
                "next route selected as g_P-to-elementary-charge mapping statement",
                1 if dominant_blocker_is_missing_gp_to_elementary_charge_mapping else 0,
                "The generic EM unit-convention bridge has been narrowed to the explicit charge-normalization statement.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_em_unit_bridge_audit",
                "pass" if dim_route["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after EM unit bridge audit",
                1 if dim_route["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline despite the unresolved charge-normalization statement.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_em_unit_bridge_audit",
                "pass" if dim_route["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after EM unit bridge audit",
                1 if dim_route["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains on reserve and is not promoted by the charge-normalization audit.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": dim_route["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(dim_route["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(dim_route["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(dim_route["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(dim_route["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": bool(dim_route["unit_consistency_audit_branch_retained"]),
            "dimensionless_alpha_bridge_branch_retained": bool(
                dim_route["dimensionless_alpha_bridge_branch_retained"]
            ),
            "em_unit_convention_bridge_branch_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_forty_fifth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "gate_summary": gate["summary"],
            "dimensionless_bridge_route_summary": dim_route,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_fifth_refresh",
        route,
    )

    print("[done] 8.7.56.991-.994 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_fifth_refresh_metrics.json")
    print(f" - dominant_blocker = {NEXT_MISSING_ARTIFACT}")


# Function: run the EM unit-convention / charge-normalization bridge branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha EM unit-convention bridge branch."""
    main()


if __name__ == "__main__":
    run_cli()
