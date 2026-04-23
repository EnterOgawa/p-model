#!/usr/bin/env python3
"""Generate 8.7.56.999-.1002 Trial-2 numeric alpha mapping-literal artifacts."""

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

QED_PRECISION = OUT / "qed_vacuum_precision_metrics.json"
STATEMENT_SOURCE = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_statement_source_inventory_metrics.json"
STATEMENT_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_statement_audit_metrics.json"
STATEMENT_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_statement_declaration_gate_metrics.json"
STATEMENT_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_sixth_refresh_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal"
NEXT_ESCALATION_ROUTE = "trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping"
NEXT_ROUTE = "8.7.56.1003"


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


# Function: execute the g_P-to-elementary-charge mapping literal branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha g_P-to-elementary-charge mapping literal branch."""
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
        QED_PRECISION,
        STATEMENT_SOURCE,
        STATEMENT_AUDIT,
        STATEMENT_GATE,
        STATEMENT_ROUTE,
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
    qed_precision = read_json(QED_PRECISION)
    statement_source = read_json(STATEMENT_SOURCE)["summary"]
    statement_audit = read_json(STATEMENT_AUDIT)["summary"]
    statement_gate = read_json(STATEMENT_GATE)["summary"]
    statement_route = read_json(STATEMENT_ROUTE)["summary"]

    prior_route_active = (
        statement_gate["selected_residual_route"] == CURRENT_ROUTE
        and statement_gate["missing_v2_artifact"] == CURRENT_ROUTE
        and statement_route["selected_next_generation_route"] == CURRENT_ROUTE
    )

    advice_has_final_formula = hit(advice_text, r"\alpha = \frac{4\pi G^2 Z_P}{\hbar c}") is not None
    part1_has_weak_field_normalization = hit(part1_text, r"g_P/Z_P=4\pi G") is not None
    part2_has_h0p_background_law = hit(part2_text, r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]") is not None
    part3a_has_structural_charge_rule = hit(part3a_text, r"e=g_P/\sqrt{Z_P}") is not None
    em_doc_has_coulomb_q_surface = hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}") is not None
    elementary_charge_constant_available = "e_charge_c" in qed_precision["constants_si"]
    ai_context_has_stop_rule = "stop the mechanical descent and ask for expert guidance" in json.dumps(
        ai_context, ensure_ascii=False
    )

    explicit_mapping_literal_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, r"e_{\mathrm{phys}}=g_P/\sqrt{Z_P}"),
            hit(part3a_text, r"e_{\mathrm{phys}}=g_P/\sqrt{Z_P}"),
            hit(part5_text, r"e_{\mathrm{phys}}=g_P/\sqrt{Z_P}"),
            hit(part1_text, r"e_{\mathrm{SI}}=g_P/\sqrt{Z_P}"),
            hit(part3a_text, r"e_{\mathrm{SI}}=g_P/\sqrt{Z_P}"),
            hit(part5_text, r"e_{\mathrm{SI}}=g_P/\sqrt{Z_P}"),
            hit(part1_text, r"g_P=e\sqrt{Z_P}"),
            hit(part3a_text, r"g_P=e\sqrt{Z_P}"),
            hit(part5_text, r"g_P=e\sqrt{Z_P}"),
        )
    )

    new_public_canonical_surface_added = explicit_mapping_literal_available and (
        not statement_audit["explicit_mapping_literal_available"]
    )
    retry_triage_gate_triggered = (
        not explicit_mapping_literal_available
        and not new_public_canonical_surface_added
        and not statement_audit["explicit_mapping_literal_available"]
    )

    inventory_ready = all(
        [
            bool(statement_source["inventory_ready"]),
            bool(statement_audit["audit_ready"]),
            prior_route_active,
            advice_has_final_formula,
            part1_has_weak_field_normalization,
            part2_has_h0p_background_law,
            part3a_has_structural_charge_rule,
            em_doc_has_coulomb_q_surface,
            elementary_charge_constant_available,
            ai_context_has_stop_rule,
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
        "qed_vacuum_precision_metrics_json": display_path(QED_PRECISION),
        "mapping_statement_source_json": display_path(STATEMENT_SOURCE),
        "mapping_statement_audit_json": display_path(STATEMENT_AUDIT),
        "mapping_statement_gate_json": display_path(STATEMENT_GATE),
        "mapping_statement_route_json": display_path(STATEMENT_ROUTE),
    }

    inventory = payload(
        "8.7.56.999",
        "Trial-2 numeric alpha g_P-to-elementary-charge mapping literal source inventory",
        common_inputs,
        "Freeze the mapping-literal pack and check whether this branch contributes any new public-canonical surface beyond the already known structural e rule and the public elementary-charge surface.",
        {
            "structural_rule": "e_structural = g_P / sqrt(Z_P)",
            "public_em_rule": "Phi(r) = q / (4*pi*eps0*r), e_charge_c is fixed in the public QED pack",
            "retry_gate_rule": "if the literal branch adds no new public-canonical surface, the next route is expert advice rather than deeper wording subdivision",
        },
        [
            row(
                "trial2_numeric_alpha_mapping_literal_inventory_complete",
                "pass" if inventory_ready else "reject",
                "g_P-to-elementary-charge mapping literal inventory complete",
                1 if inventory_ready else 0,
                "This branch needs the structural e route, the public elementary-charge surface, the prior literal blocker state, and the retry triage policy in one pack.",
            ),
            row(
                "trial2_numeric_alpha_structural_charge_surface_retained_at_literal_branch",
                "pass" if part3a_has_structural_charge_rule else "reject",
                "structural charge surface retained at literal branch",
                1 if part3a_has_structural_charge_rule else 0,
                "The structural Trial-2 route still fixes e through g_P and Z_P.",
            ),
            row(
                "trial2_numeric_alpha_public_elementary_charge_surface_retained_at_literal_branch",
                "pass" if elementary_charge_constant_available else "reject",
                "public elementary-charge surface retained at literal branch",
                1 if elementary_charge_constant_available else 0,
                "The CODATA elementary charge remains available in the public QED precision pack.",
            ),
            row(
                "trial2_numeric_alpha_explicit_mapping_literal_absent",
                "pass" if not explicit_mapping_literal_available else "reject",
                "explicit mapping literal absent",
                1 if not explicit_mapping_literal_available else 0,
                "No public source currently provides a positive literal that identifies structural e with the physical elementary charge.",
            ),
            row(
                "trial2_numeric_alpha_no_new_public_canonical_surface_added_in_literal_branch",
                "pass" if not new_public_canonical_surface_added else "reject",
                "no new public-canonical surface added in literal branch",
                1 if not new_public_canonical_surface_added else 0,
                "Relative to the statement branch, the literal branch still contributes no new canonical surface.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "structural_charge_surface_available": part3a_has_structural_charge_rule,
            "public_coulomb_charge_surface_available": em_doc_has_coulomb_q_surface,
            "public_elementary_charge_surface_available": elementary_charge_constant_available,
            "explicit_mapping_literal_available": explicit_mapping_literal_available,
            "new_public_canonical_surface_added_in_literal_branch": new_public_canonical_surface_added,
            "first_route_to_close_or_none": NEXT_ESCALATION_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_mapping_literal_inventory_frozen",
            "advance_to_8_7_56_1000": inventory_ready,
            "next_required_artifacts": [CURRENT_ROUTE, NEXT_ESCALATION_ROUTE],
        },
        {
            "prior_statement_source_summary": statement_source,
            "part3a_structural_charge_rule_hit": hit(part3a_text, r"e=g_P/\sqrt{Z_P}"),
            "em_doc_coulomb_q_hit": hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}"),
            "qed_elementary_charge": qed_precision["constants_si"]["e_charge_c"],
        },
    )

    audit = payload(
        "8.7.56.1000",
        "Trial-2 numeric alpha g_P-to-elementary-charge mapping literal audit",
        common_inputs,
        "Audit whether the mapping-literal branch contributes a positive literal or any new public-canonical surface; if not, trigger the retry triage gate and stop mechanical wording descent.",
        {
            "literal_rule": "a mapping literal would provide an explicit positive formula such as e_phys = g_P / sqrt(Z_P) or g_P = e*sqrt(Z_P)",
            "triage_rule": "if the literal is still absent and no new public-canonical surface is added, the next route is expert advice rather than phrase/fragment descent",
            "audit_rule": "the literal branch is low-value if it preserves the same blocker without adding new canonical evidence",
        },
        [
            row(
                "trial2_numeric_alpha_mapping_literal_audit_complete",
                "pass",
                "g_P-to-elementary-charge mapping literal audit complete",
                1,
                "The current branch audits whether the positive identification literal actually exists and whether this branch added any new public-canonical surface.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_contains_explicit_mapping_literal",
                "pass" if explicit_mapping_literal_available else "reject",
                "current pack contains explicit mapping literal",
                1 if explicit_mapping_literal_available else 0,
                "The audit finds no explicit positive formula that equates structural e with the public elementary charge.",
            ),
            row(
                "trial2_numeric_alpha_literal_branch_added_new_public_canonical_surface",
                "pass" if new_public_canonical_surface_added else "reject",
                "literal branch added new public-canonical surface",
                1 if new_public_canonical_surface_added else 0,
                "Compared with the statement branch, this literal branch does not add new canonical evidence.",
            ),
            row(
                "trial2_numeric_alpha_retry_triage_gate_triggered_after_literal_audit",
                "pass" if retry_triage_gate_triggered else "reject",
                "retry triage gate triggered after literal audit",
                1 if retry_triage_gate_triggered else 0,
                "Because the same wording family repeated without adding new canonical surface, the next route should switch to expert advice.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "explicit_mapping_literal_available": explicit_mapping_literal_available,
            "new_public_canonical_surface_added_in_literal_branch": new_public_canonical_surface_added,
            "retry_triage_gate_triggered": retry_triage_gate_triggered,
            "first_route_to_close_after_audit_or_none": NEXT_ESCALATION_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_mapping_literal_audited",
            "advance_to_8_7_56_1001": True,
            "next_required_artifacts": [CURRENT_ROUTE, NEXT_ESCALATION_ROUTE],
        },
        {
            "prior_statement_audit_summary": statement_audit,
            "prior_statement_gate_summary": statement_gate,
            "ai_context_next_rule": ai_context["next"][-1],
        },
    )

    gate = payload(
        "8.7.56.1001",
        "Trial-2 numeric alpha g_P-to-elementary-charge mapping literal declaration gate",
        common_inputs,
        "Update the official gate after the mapping-literal audit: the literal remains absent, no new public-canonical surface was added, and the retry triage rule now promotes expert advice as the next official route.",
        {
            "gate_rule": "a direct SI alpha readout cannot be honest until one explicit literal identifies structural e with physical elementary charge",
            "stop_rule": "if the literal branch adds no new canonical surface, stop mechanical wording descent and escalate for expert advice",
        },
        [
            row(
                "trial2_numeric_alpha_mapping_literal_gate_complete",
                "pass",
                "g_P-to-elementary-charge mapping literal declaration gate complete",
                1,
                "The official state is updated after the mapping-literal audit.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_mapping_literal_audit",
                "pass" if explicit_mapping_literal_available else "reject",
                "numeric alpha from current pack ready after mapping literal audit",
                1 if explicit_mapping_literal_available else 0,
                "Without the explicit positive literal, the direct SI readout remains pre-canonical.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready_after_mapping_literal_audit",
                "pass" if explicit_mapping_literal_available else "reject",
                "Trial-2 numeric alpha closeout ready after mapping literal audit",
                1 if explicit_mapping_literal_available else 0,
                "Closeout remains blocked while the positive mapping literal is absent.",
            ),
            row(
                "trial2_numeric_alpha_stop_mechanical_wording_descent_after_mapping_literal_audit",
                "pass" if retry_triage_gate_triggered else "reject",
                "stop mechanical wording descent after mapping literal audit",
                1 if retry_triage_gate_triggered else 0,
                "The retry triage gate is now active because the literal branch introduced no new canonical surface.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": True,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": True,
            "trial2_numeric_alpha_raw_final_computation_value_available": bool(
                statement_gate["trial2_numeric_alpha_raw_final_computation_value_available"]
            ),
            "trial2_numeric_alpha_numeric_from_current_pack_ready": explicit_mapping_literal_available,
            "trial2_numeric_alpha_closeout_ready": explicit_mapping_literal_available,
            "trial2_numeric_alpha_final_computation_performed": bool(
                statement_gate["trial2_numeric_alpha_final_computation_performed"]
            ),
            "trial2_numeric_alpha_final_computation_result_class": "precanonical_unit_incomplete",
            "trial2_numeric_alpha_retry_loop_retired": bool(
                statement_gate["trial2_numeric_alpha_retry_loop_retired"]
            ),
            "trial2_numeric_alpha_retry_triage_gate_triggered": retry_triage_gate_triggered,
            "selected_residual_route": CURRENT_ROUTE,
            "missing_v2_artifact": CURRENT_ROUTE,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_mapping_literal_gate_closed",
            "advance_to_8_7_56_1002": True,
            "next_required_artifacts": [CURRENT_ROUTE, NEXT_ESCALATION_ROUTE],
        },
        {
            "mapping_literal_audit_summary": audit["summary"],
            "prior_gate_summary": statement_gate,
            "prior_route_summary": statement_route,
        },
    )

    route = payload(
        "8.7.56.1002",
        "Trial-2 numeric alpha route contract one-hundred-forty-seventh refresh",
        common_inputs,
        "Refresh the next-generation contract after the mapping-literal audit: keep Trial-2 numeric alpha on the precision mainline, keep the strong side on reserve, and switch the next official route from wording descent to expert advice.",
        {
            "next_route_rule": "the next route is expert advice on the missing g_P-to-elementary-charge mapping because the literal branch added no new public-canonical surface",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
            "triage_rule": "the retry triage gate converts low-value wording repetition into expert-advice escalation",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_forty_seventh_refresh_complete",
                "pass",
                "route contract one-hundred-forty-seventh refresh complete",
                1,
                "The mapping-literal audit is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_expert_advice_after_literal_audit",
                "pass" if retry_triage_gate_triggered else "reject",
                "next route selected as expert advice after literal audit",
                1 if retry_triage_gate_triggered else 0,
                "The retry triage gate redirects the next branch from wording descent to expert advice.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_mapping_literal_audit",
                "pass" if statement_route["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after mapping literal audit",
                1 if statement_route["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline despite the unresolved mapping literal.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_mapping_literal_audit",
                "pass" if statement_route["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after mapping literal audit",
                1 if statement_route["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains on reserve and is not promoted by the mapping-literal audit.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_ESCALATION_ROUTE,
            "strong_side_route_state": statement_route["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(statement_route["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(statement_route["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(statement_route["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(statement_route["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": bool(statement_route["unit_consistency_audit_branch_retained"]),
            "dimensionless_alpha_bridge_branch_retained": bool(
                statement_route["dimensionless_alpha_bridge_branch_retained"]
            ),
            "em_unit_convention_bridge_branch_retained": bool(
                statement_route["em_unit_convention_bridge_branch_retained"]
            ),
            "mapping_statement_branch_retained": bool(statement_route["mapping_statement_branch_retained"]),
            "mapping_literal_branch_retained": True,
            "same_pattern_retry_threshold_reached": retry_triage_gate_triggered,
            "retry_triage_gate_triggered": retry_triage_gate_triggered,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_forty_seventh_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [CURRENT_ROUTE, NEXT_ESCALATION_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": statement_route,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_seventh_refresh",
        route,
    )

    print("[done] 8.7.56.999-.1002 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_seventh_refresh_metrics.json")
    print(f" - selected_next_generation_route = {NEXT_ESCALATION_ROUTE}")


# Function: run the mapping-literal branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha mapping-literal branch."""
    main()


if __name__ == "__main__":
    run_cli()
