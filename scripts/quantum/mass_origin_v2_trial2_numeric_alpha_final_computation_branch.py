#!/usr/bin/env python3
"""Generate 8.7.56.979-.982 Trial-2 numeric alpha final-computation closeout artifacts."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"

ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_final_computation.md")
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

QED_PRECISION = OUT / "qed_vacuum_precision_metrics.json"
QBALL_FULL_COUPLED = OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"
PBG_METRICS = ROOT / "output" / "public" / "cosmology" / "cosmology_redshift_pbg_metrics.json"
PRIOR_GATE = (
    OUT
    / "mass_origin_v2_trial2_numeric_alpha_h0p_zp_background_factor_late_time_vev_linearized_field_equation_background_factor_definition_value_late_time_readout_literal_fixed_point_mapping_literal_value_symbol_fragment_declaration_gate_metrics.json"
)
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_first_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_newton_limit_h0p_zp_background_factor_late_time_vev_linearized_field_equation_background_factor_definition_value_late_time_readout_literal_fixed_point_mapping_literal_value_terminal_atom_identification"
)
CURRENT_ARTIFACT = (
    "trial2_numeric_alpha_h0p_zp_background_factor_late_time_vev_linearized_field_equation_background_factor_definition_value_late_time_readout_literal_fixed_point_mapping_literal_value_terminal_atom"
)
NEXT_ROUTE = "8.7.56.983"
NEXT_BRANCH = "8.7.56.983-.986"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_final_computation_unit_consistency_audit"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_final_computation_unit_consistency_audit"


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


# Function: extract the exact ground-state mass proxy E(beta_1) from the public vector-Q-ball ladder.

def extract_ground_state_mass_proxy(qball_full_coupled: dict) -> float:
    """Return the exact mass proxy for the public reference state M_(1,0,0,0)."""
    for candidate in qball_full_coupled["evidence"]["exact_ladder_sample_rows"]:
        if (
            candidate.get("n") == 1
            and candidate.get("k") == 0
            and candidate.get("ell") == 0
            and candidate.get("s") == 0
        ):
            return float(candidate["exact_mass_proxy"])

    raise SystemExit("[fail] missing exact ground-state row M_(1,0,0,0)")


# Function: classify the final computation result against the advice thresholds.

def classify_relative_error(relative_error: float) -> str:
    """Classify the final computation under the advice thresholds."""
    if relative_error < 0.10:
        return "numeric_pass"

    if relative_error < 0.50:
        return "numeric_constraint_watch"

    return "numeric_tension_reject"


# Function: execute the final-computation closeout branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha final-computation closeout branch."""
    for path in (
        ADVICE,
        PART1,
        PART2,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        QED_PRECISION,
        QBALL_FULL_COUPLED,
        PBG_METRICS,
        PRIOR_GATE,
        PRIOR_ROUTE,
    ):
        require(path)

    advice_text = read_text(ADVICE)
    part1_text = read_text(PART1)
    part2_text = read_text(PART2)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    qed_precision = read_json(QED_PRECISION)
    qball_full_coupled = read_json(QBALL_FULL_COUPLED)
    pbg_metrics = read_json(PBG_METRICS)
    prior_gate = read_json(PRIOR_GATE)["summary"]
    prior_route = read_json(PRIOR_ROUTE)["summary"]

    computation_formula_ready = bool(prior_gate["trial2_numeric_alpha_computation_formula_ready"])
    absolute_normalization_dictionary_ready = bool(
        prior_gate["trial2_numeric_alpha_absolute_normalization_dictionary_ready"]
    )
    h0p_bridge_pivot_retained = bool(prior_gate["h0p_bridge_pivot_retained"])
    prior_route_active = (
        prior_gate["selected_residual_route"] == CURRENT_ROUTE
        and prior_gate["missing_v2_artifact"] == CURRENT_ARTIFACT
        and prior_route["selected_next_generation_route"] == CURRENT_ROUTE
    )

    advice_has_cbg_equals_one = (
        hit(advice_text, "C_bg = 1") is not None
        or hit(advice_text, r"C_{\rm bg} = 1") is not None
    )
    advice_has_zp_rule = hit(advice_text, r"Z_P = \frac{m_0^2}{(H_0^{(P)})^2}") is not None
    advice_has_alpha_rule = hit(advice_text, r"\alpha = \frac{4\pi G^2 Z_P}{\hbar c}") is not None
    advice_has_final_formula = advice_has_zp_rule and advice_has_alpha_rule
    part1_has_electron_identification = hit(part1_text, r"m_0 = \frac{m_e}{\mathcal{E}(\beta_1)}") is not None
    part2_has_h0p_law = hit(part2_text, r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]") is not None
    part3a_has_current_alpha_route = hit(part3a_text, r"\alpha=16\pi G^2\lambda v^2/(m_0^2\hbar c)") is not None
    part5_has_current_checkpoint = hit(part5_text, "current checkpoint") is not None

    constants_si = qed_precision["constants_si"]
    G = float(constants_si["G_m3_kg_s2"])
    hbar = float(constants_si["hbar_j_s"])
    c = float(constants_si["c_m_per_s"])
    m_e_kg = float(constants_si["m_e_kg"])
    alpha_target = 1.0 / float(qed_precision["alpha_precision"]["g2"]["alpha_inv"])
    E_beta1 = extract_ground_state_mass_proxy(qball_full_coupled)
    H0P_si = float(pbg_metrics["derived"]["H0P_SI_s^-1"])

    m0_kg = m_e_kg / E_beta1
    Z_P = (m0_kg**2) / (H0P_si**2)
    alpha_pmodel = 4.0 * math.pi * (G**2) * Z_P / (hbar * c)
    alpha_ratio_to_target = alpha_pmodel / alpha_target
    relative_error = abs(alpha_pmodel - alpha_target) / alpha_target
    log10_gap_to_target = math.log10(alpha_target / alpha_pmodel)
    final_result_class = classify_relative_error(relative_error)
    pass_10pct = final_result_class == "numeric_pass"
    watch_10_to_50pct = final_result_class == "numeric_constraint_watch"
    reject_gt_50pct = final_result_class == "numeric_tension_reject"

    input_pack_ready = all(
        [
            computation_formula_ready,
            absolute_normalization_dictionary_ready,
            h0p_bridge_pivot_retained,
            prior_route_active,
            advice_has_cbg_equals_one,
            advice_has_final_formula,
            part1_has_electron_identification,
            part2_has_h0p_law,
            part3a_has_current_alpha_route,
            part5_has_current_checkpoint,
        ]
    )

    common_inputs = {
        "expert_note_markdown": display_path(ADVICE),
        "part1_markdown": display_path(PART1),
        "part2_markdown": display_path(PART2),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "qed_vacuum_precision_metrics_json": display_path(QED_PRECISION),
        "mass_origin_vector_qball_full_coupled_solver_pilot_json": display_path(QBALL_FULL_COUPLED),
        "cosmology_redshift_pbg_metrics_json": display_path(PBG_METRICS),
        "prior_gate_json": display_path(PRIOR_GATE),
        "prior_route_json": display_path(PRIOR_ROUTE),
    }

    inventory_payload = payload(
        "8.7.56.979",
        "Trial-2 numeric alpha final computation source inventory",
        common_inputs,
        "Retire the terminal-atom retry loop and freeze the final-computation input pack: C_bg=1 from the advice note, E(beta_1) from the public vector-Q-ball ground state, H0^(P) from the public redshift background law, and CODATA constants from the public QED precision pack.",
        {
            "final_computation_rule": "C_bg = 1 and Z_P = m_0^2 / (H_0^(P))^2",
            "electron_identification_rule": "m_0 = m_e / E(beta_1)",
            "alpha_rule": "alpha = 4*pi*G^2*Z_P / (hbar*c) = 4*pi*G^2*m_0^2 / ((H_0^(P))^2*hbar*c)",
        },
        [
            row(
                "trial2_numeric_alpha_final_computation_inventory_complete",
                "pass" if input_pack_ready else "reject",
                "final-computation input-pack inventory complete",
                1 if input_pack_ready else 0,
                "The terminal-atom retry loop can be retired only if the final computation note, the public ground-state proxy, the public H0^(P) value, and the public constants pack are all present together.",
            ),
            row(
                "trial2_numeric_alpha_final_computation_advice_cbg_equals_one_available",
                "pass" if advice_has_cbg_equals_one else "reject",
                "final-computation advice states C_bg = 1",
                1 if advice_has_cbg_equals_one else 0,
                "The advice note now treats C_bg as a computed value instead of a missing public token.",
            ),
            row(
                "trial2_numeric_alpha_public_ground_state_proxy_available",
                "pass",
                "public ground-state proxy E(beta_1) available",
                1,
                "The exact vector-Q-ball ground-state row M_(1,0,0,0) is already public in the full-coupled ladder metrics.",
            ),
            row(
                "trial2_numeric_alpha_public_h0p_value_available",
                "pass",
                "public H0^(P) value available",
                1,
                "The late-time redshift background law already fixes H0^(P) numerically in the public cosmology metrics.",
            ),
            row(
                "trial2_numeric_alpha_prior_terminal_atom_retry_active_before_closeout",
                "pass" if prior_route_active else "reject",
                "prior terminal-atom retry active before closeout",
                1 if prior_route_active else 0,
                "The final-computation branch closes the currently active terminal-atom retry rather than an older blocker family.",
            ),
        ],
        {
            "inventory_ready": input_pack_ready,
            "computation_formula_ready": computation_formula_ready,
            "absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "h0p_bridge_pivot_retained": h0p_bridge_pivot_retained,
            "final_computation_input_pack_ready": input_pack_ready,
            "E_beta1_available": True,
            "E_beta1_value": E_beta1,
            "H0P_si_available": True,
            "H0P_si_value": H0P_si,
            "C_bg_value_fixed": 1.0,
            "first_route_to_close_or_none": None,
        },
        {
            "overall_status": "trial2_numeric_alpha_final_computation_input_pack_frozen",
            "advance_to_8_7_56_980": input_pack_ready,
            "next_required_artifacts": [],
        },
        {
            "ai_context_current_step": ai_context["current_step"],
            "advice_cbg_hit": hit(advice_text, "C_bg = 1"),
            "advice_zp_rule_hit": hit(advice_text, r"Z_P = \frac{m_0^2}{(H_0^{(P)})^2}"),
            "advice_alpha_rule_hit": hit(advice_text, r"\alpha = \frac{4\pi G^2 Z_P}{\hbar c}"),
            "ground_state_reference_row": {
                "state": "M_(1,0,0,0)",
                "exact_mass_proxy": E_beta1,
            },
            "public_h0p_summary": pbg_metrics["derived"],
        },
    )

    audit_payload = payload(
        "8.7.56.980",
        "Trial-2 numeric alpha final computation audit",
        common_inputs,
        "Perform the final computation once and classify the result under the explicit thresholds from the advice memo.",
        {
            "m0_rule": "m_0 = m_e / E(beta_1)",
            "ZP_rule": "Z_P = m_0^2 / (H_0^(P))^2",
            "alpha_rule": "alpha = 4*pi*G^2*Z_P / (hbar*c)",
            "threshold_rule": "relative_error < 0.10 => pass, 0.10-0.50 => watch, > 0.50 => tension/reject",
        },
        [
            row(
                "trial2_numeric_alpha_final_computation_audit_complete",
                "pass",
                "final-computation audit complete",
                1,
                "The retry loop is replaced by one explicit numeric evaluation with frozen public inputs.",
            ),
            row(
                "trial2_numeric_alpha_final_computation_input_pack_ready",
                "pass" if input_pack_ready else "reject",
                "final-computation input pack ready",
                1 if input_pack_ready else 0,
                "The numeric evaluation is meaningful only if the public input pack is complete.",
            ),
            row(
                "trial2_numeric_alpha_final_computation_pass_10pct",
                "pass" if pass_10pct else "reject",
                "numeric alpha relative error below 10%",
                1 if pass_10pct else 0,
                "A pass requires agreement with the QED target at the 10% level or better.",
            ),
            row(
                "trial2_numeric_alpha_final_computation_watch_10_to_50pct",
                "pass" if watch_10_to_50pct else "reject",
                "numeric alpha relative error between 10% and 50%",
                1 if watch_10_to_50pct else 0,
                "A watch result keeps the structural pass but records numeric constraint only.",
            ),
            row(
                "trial2_numeric_alpha_final_computation_tension_gt_50pct",
                "pass" if reject_gt_50pct else "reject",
                "numeric alpha relative error above 50%",
                1 if reject_gt_50pct else 0,
                "A >50% mismatch is treated as tension for the current H0^(P)-Z_P final-computation closure.",
            ),
        ],
        {
            "audit_ready": input_pack_ready,
            "final_computation_performed": True,
            "E_beta1": E_beta1,
            "H0P_si_s^-1": H0P_si,
            "m0_kg": m0_kg,
            "Z_P": Z_P,
            "alpha_pmodel": alpha_pmodel,
            "alpha_target": alpha_target,
            "alpha_ratio_to_target": alpha_ratio_to_target,
            "relative_error": relative_error,
            "log10_gap_to_target": log10_gap_to_target,
            "result_class": final_result_class,
            "pass_10pct": pass_10pct,
            "watch_10_to_50pct": watch_10_to_50pct,
            "reject_gt_50pct": reject_gt_50pct,
            "first_route_to_close_after_audit_or_none": "trial2_numeric_alpha_final_computation_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_final_computation_audited",
            "advance_to_8_7_56_981": True,
            "next_required_artifacts": [],
        },
        {
            "constants_si": {
                "G": G,
                "hbar": hbar,
                "c": c,
                "m_e_kg": m_e_kg,
            },
            "alpha_target_source": qed_precision["alpha_precision"]["g2"],
            "pbg_model": pbg_metrics["model"],
        },
    )

    gate_payload = payload(
        "8.7.56.981",
        "Trial-2 numeric alpha final computation declaration gate",
        common_inputs,
        "Close the retry loop honestly: record that numeric alpha can now be emitted from the current public pack, but that the current H0^(P)-Z_P final-computation closure lands in numeric tension rather than pass.",
        {
            "gate_rule": "once the final computation is performed, terminal-atom / terminal-glyph / symbol-fragment retries are retired",
            "closeout_rule": "numeric_from_current_pack_ready can be true while closeout_ready remains false if the computed alpha misses the target by more than 50%",
        },
        [
            row(
                "trial2_numeric_alpha_final_computation_gate_complete",
                "pass",
                "final-computation declaration gate complete",
                1,
                "The retry loop is retired and replaced by an explicit numeric result.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_final_computation",
                "pass" if input_pack_ready else "reject",
                "numeric alpha from current pack ready after final computation",
                1 if input_pack_ready else 0,
                "The current public pack is now sufficient to emit a numeric alpha candidate.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready_after_final_computation",
                "pass" if pass_10pct else "reject",
                "Trial-2 numeric alpha closeout ready after final computation",
                1 if pass_10pct else 0,
                "Closeout requires a passing numeric result; the current computation does not satisfy that threshold.",
            ),
            row(
                "trial2_numeric_alpha_retry_loop_retired_after_final_computation",
                "pass",
                "terminal-atom retry loop retired after final computation",
                1,
                "The source-inventory subdivision loop is no longer the honest next step once the numeric result is explicit.",
            ),
            row(
                "trial2_numeric_alpha_current_h0p_zp_closeout_in_tension",
                "pass" if reject_gt_50pct else "reject",
                "current H0^(P)-Z_P closeout in numeric tension",
                1 if reject_gt_50pct else 0,
                "The computed alpha is far below the QED target under the current public frozen inputs.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": computation_formula_ready,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": input_pack_ready,
            "trial2_numeric_alpha_closeout_ready": pass_10pct,
            "trial2_numeric_alpha_final_computation_performed": True,
            "trial2_numeric_alpha_final_computation_result_class": final_result_class,
            "trial2_numeric_alpha_retry_loop_retired": True,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_final_computation_gate_closed",
            "advance_to_8_7_56_982": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": prior_gate,
            "prior_route_summary": prior_route,
        },
    )

    route_payload = payload(
        "8.7.56.982",
        "Trial-2 numeric alpha route contract one-hundred-forty-second refresh",
        common_inputs,
        "Refresh the next-generation contract after the final computation: keep Trial-2 numeric alpha on the mainline, keep the strong side on reserve, and promote discrepancy / unit-consistency interpretation as the next official route.",
        {
            "next_route_rule": "the next route audits whether the final-computation tension is a unit-consistency / normalization issue or a genuine no-go for the current H0^(P)-Z_P closure",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_forty_second_refresh_complete",
                "pass",
                "route contract one-hundred-forty-second refresh complete",
                1,
                "The next-generation contract is refreshed after the explicit numeric result is frozen.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_final_computation_unit_consistency_audit",
                "pass",
                "next route selected as final-computation unit-consistency audit",
                1,
                "The next official route no longer searches for terminal tokens; it audits the discrepancy itself.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_final_computation",
                "pass" if prior_route["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after final computation",
                1 if prior_route["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline even after the tension result.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_final_computation",
                "pass" if prior_route["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after final computation",
                1 if prior_route["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains on reserve and is not promoted by the current alpha tension result.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": prior_route["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(prior_route["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": absolute_normalization_dictionary_ready,
            "h0p_bridge_pivot_retained": h0p_bridge_pivot_retained,
            "final_computation_branch_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_forty_second_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": prior_route,
        },
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_final_computation_source_inventory", inventory_payload)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_final_computation_audit", audit_payload)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_final_computation_declaration_gate", gate_payload)
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_second_refresh", route_payload)

    print("[done] 8.7.56.979-.982 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_second_refresh_metrics.json")
    print(f" - alpha_pmodel = {alpha_pmodel:.16e}")
    print(f" - alpha_target = {alpha_target:.16e}")
    print(f" - relative_error = {relative_error:.16e}")


# Function: run the final-computation closeout branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha final-computation closeout branch."""
    main()


if __name__ == "__main__":
    run_cli()
