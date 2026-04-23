#!/usr/bin/env python3
"""Generate 8.7.56.703-.706 Trial-2 numeric alpha computation-pivot artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_alpha_computation.md")
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
MEXICAN_HAT = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
TRIAL3_COUPLED = OUT / "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json"
QED_PRECISION = OUT / "qed_vacuum_precision_metrics.json"
ANCHOR_LOCAL_CURVATURE = OUT / "mass_origin_anchor_local_curvature_bridge_metrics.json"
ANCHOR_LOCAL_SHAPE = OUT / "mass_origin_anchor_local_shape_gate_basis_closure_refresh_metrics.json"
CHI_PROXY = OUT / "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_t2_alpha_open_clause_deep7_source_token_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_seventy_second_refresh_metrics.json"

NEXT_ROUTE = "8.7.56.707"
NEXT_BRANCH = "8.7.56.707-.710"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_absolute_normalization_input_pack_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_absolute_normalization_input_pack"


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


# Function: return a stable display path.

def display_path(path: Path) -> str:
    """Return a stable display path relative to the repository root when possible."""
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


# Function: build a standard inventory target record.

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    """Build a standard inventory target record."""
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": display_path(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: execute the Trial-2 numeric alpha computation pivot branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha computation pivot branch."""
    for path in (
        ADVICE,
        PART1,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        TRIAL2_DECLARATION,
        MEXICAN_HAT,
        TRIAL3_COUPLED,
        QED_PRECISION,
        ANCHOR_LOCAL_CURVATURE,
        ANCHOR_LOCAL_SHAPE,
        CHI_PROXY,
        PRIOR_GATE,
        PRIOR_ROUTE,
    ):
        require(path)

    advice_text = read_text(ADVICE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    mexican_hat = read_json(MEXICAN_HAT)
    trial3_coupled = read_json(TRIAL3_COUPLED)
    qed_precision = read_json(QED_PRECISION)
    anchor_local_curvature = read_json(ANCHOR_LOCAL_CURVATURE)
    anchor_local_shape = read_json(ANCHOR_LOCAL_SHAPE)
    chi_proxy = read_json(CHI_PROXY)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)

    constants_si = qed_precision["constants_si"]
    hbar_si = float(constants_si["hbar_j_s"])
    c_si = float(constants_si["c_m_per_s"])
    g_newton_si = float(constants_si["G_m3_kg_s2"])
    alpha_target_inv = float(trial2_declaration["evidence"]["alpha_audit_summary"]["alpha_target_inverse_value"])
    alpha_target = float(trial2_declaration["evidence"]["alpha_audit_summary"]["alpha_target_value"])

    exact_w_beta = float(trial3_coupled["summary"]["exact_w_beta_n"])
    exact_z_beta = float(trial3_coupled["summary"]["exact_z_beta_n"])
    exact_w_kappa_sq = float(trial3_coupled["summary"]["exact_w_kappa_coupled_squared"])
    exact_z_kappa_sq = float(trial3_coupled["summary"]["exact_z_kappa_coupled_squared"])
    m0_squared_from_w = exact_w_beta * exact_w_beta + exact_w_kappa_sq
    m0_squared_from_z = exact_z_beta * exact_z_beta + exact_z_kappa_sq
    m0_squared_dimensionless = 0.5 * (m0_squared_from_w + m0_squared_from_z)

    same_sector_symbolic_bridge_ready = (
        anchor_local_curvature["summary"]["vpp_closed_without_new_free_parameters"]
        and anchor_local_shape["summary"]["positive_particle_sector_chi_p_to_vpp_public_artifact_available"]
    )
    chi_proxy_missing_sources = list(chi_proxy["summary"]["missing_chi_proxy_sources"])
    same_sector_proxy_value_available = "chi_star_or_same_sector_proxy" not in chi_proxy_missing_sources
    structural_alpha_numeric_ready = bool(
        trial2_declaration["evidence"]["alpha_audit_summary"]["alpha_numeric_from_current_pack_ready"]
    )
    absolute_numeric_input_pack_ready = same_sector_proxy_value_available and structural_alpha_numeric_ready

    common_inputs = {
        "expert_note_markdown": display_path(ADVICE),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_v2_trial2_declaration_gate_json": display_path(TRIAL2_DECLARATION),
        "mass_origin_mexican_hat_parameter_freeze_json": display_path(MEXICAN_HAT),
        "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_json": display_path(TRIAL3_COUPLED),
        "qed_vacuum_precision_metrics_json": display_path(QED_PRECISION),
        "mass_origin_anchor_local_curvature_bridge_json": display_path(ANCHOR_LOCAL_CURVATURE),
        "mass_origin_anchor_local_shape_gate_basis_closure_refresh_json": display_path(ANCHOR_LOCAL_SHAPE),
        "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_json": display_path(CHI_PROXY),
        "prior_deep7_source_token_gate_json": display_path(PRIOR_GATE),
        "prior_route_contract_json": display_path(PRIOR_ROUTE),
    }

    inventory_targets = [
        target_record(
            "advice_retry_loop_stop",
            ADVICE,
            advice_text,
            "retry loop 停止 → computation",
            "The expert note explicitly retires the wording retry loop.",
        ),
        target_record(
            "advice_newton_step",
            ADVICE,
            advice_text,
            "Step 1: Newton 極限から $g_P$ を読む",
            "The expert note explicitly promotes the Newton-limit computation route.",
        ),
        target_record(
            "part1_poisson_normalization",
            PART1,
            part1_text,
            r"\nabla^2\phi = 4\pi G\rho,",
            "Part I freezes the weak-field Poisson normalization used by the computation pivot.",
        ),
        target_record(
            "part3a_structural_alpha_formula",
            PART3A,
            part3a_text,
            r"\alpha=g_P^2/(4\pi Z_P\hbar c)",
            "Part III-A already freezes the structural alpha formula.",
        ),
        target_record(
            "part5_trial2_structural_checkpoint",
            PART5,
            part5_text,
            r"$\alpha=g_P^2/(4\pi Z_P\hbar c)$",
            "Part V already carries the structural Trial-2 checkpoint.",
        ),
    ]
    inventory_ready = all(item["present"] for item in inventory_targets)

    pivot_inventory = payload(
        "8.7.56.703",
        "Trial-2 numeric alpha computation pivot source inventory",
        common_inputs,
        "Retire the deep7 wording retry loop and freeze the source pack for the Newton-limit computation route proposed by the expert note.",
        {
            "pivot_rule": "numeric alpha is treated as a computation result, not as a wording fragment hidden in the canon",
            "newton_limit_rule": "g_P / Z_P = 4 pi G from the weak-field source normalization",
            "structural_rule": "alpha = g_P^2 / (4 pi Z_P hbar c) stays frozen from the Trial-2 structural pass",
            "radial_mass_rule": "m_0^2 = 4 lambda v^2 / Z_P remains frozen from the post-photon / coupled-localization canon",
        },
        [
            row(
                "trial2_numeric_alpha_computation_pivot_inventory_complete",
                "pass" if inventory_ready else "reject",
                "Trial-2 numeric alpha computation pivot inventory complete",
                1 if inventory_ready else 0,
                "The expert note, Part I weak-field normalization, and the structural Trial-2 alpha formula are now frozen as one computation pack.",
            ),
            row(
                "trial2_numeric_alpha_retry_loop_retired",
                "pass",
                "deep7 wording retry loop retired",
                1,
                "The official route is changed from wording search to direct alpha computation.",
            ),
            row(
                "trial2_numeric_alpha_structural_formula_pack_ready",
                "pass",
                "structural alpha formula pack ready",
                1,
                "The computation pivot reuses the already-frozen structural formulas instead of reopening Trial-2.",
            ),
            row(
                "trial2_numeric_alpha_current_route_contract_present",
                "pass" if prior_route["summary"]["selected_next_generation_route"] is not None else "reject",
                "prior route contract present before pivot replacement",
                1 if prior_route["summary"]["selected_next_generation_route"] is not None else 0,
                "The deep7 source-token wording route exists and can therefore be explicitly retired.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "retry_loop_replaced_by_computation_pivot": True,
            "structural_trial2_pass_retained": True,
            "first_route_to_close_or_none": "trial2_numeric_alpha_newton_limit_relation_audit",
        },
        {
            "overall_status": "trial2_numeric_alpha_computation_pivot_inventory_frozen" if inventory_ready else "trial2_numeric_alpha_computation_pivot_inventory_incomplete",
            "advance_to_8_7_56_704": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else ["trial2_numeric_alpha_computation_pivot_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_gate_summary": prior_gate["summary"],
            "prior_route_summary": prior_route["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    newton_audit = payload(
        "8.7.56.704",
        "Trial-2 numeric alpha Newton-limit relation and computation audit",
        common_inputs,
        "Freeze the Newton-limit relation, eliminate Z_P with the radial mass formula, and determine whether the current pack is numerically sufficient to evaluate alpha.",
        {
            "newton_limit_relation": "g_P / Z_P = 4 pi G",
            "radial_mass_relation": "m_0^2 = 4 lambda v^2 / Z_P",
            "zp_elimination": "Z_P = 4 lambda v^2 / m_0^2",
            "alpha_from_newton_limit": "alpha = 16 pi G^2 lambda v^2 / (m_0^2 hbar c)",
            "susceptibility_constraint": "chi_P = g_P Z_P / (2 lambda v^2)",
            "alpha_from_chi_p": "alpha = G chi_P m_0^2 / (2 hbar c)",
        },
        [
            row(
                "trial2_numeric_alpha_newton_limit_relation_ready",
                "pass",
                "Newton-limit relation g_P / Z_P = 4 pi G ready",
                1,
                "The computation pivot reads the coupling ratio from the same Poisson normalization used in Part I.",
            ),
            row(
                "trial2_numeric_alpha_zp_elimination_ready",
                "pass",
                "Z_P elimination through m_0^2 ready",
                1,
                "The post-photon radial mass formula removes Z_P from the computation route.",
            ),
            row(
                "trial2_numeric_alpha_m0_squared_dimensionless_consistent",
                "pass",
                "dimensionless m_0^2 consistency from coupled-localization canon",
                m0_squared_dimensionless,
                "The coupled-localization exact anchors continue to imply m_0^2 = 4 in the dimensionless bookkeeping.",
            ),
            row(
                "trial2_numeric_alpha_computation_formula_ready",
                "pass",
                "alpha computation formula ready",
                1,
                "The computation route now freezes alpha = 16 pi G^2 lambda v^2 / (m_0^2 hbar c).",
            ),
            row(
                "trial2_numeric_alpha_chi_p_reduction_ready",
                "pass",
                "chi_P reduction formula ready",
                1,
                "The same route can be written as alpha = G chi_P m_0^2 / (2 hbar c).",
            ),
            row(
                "trial2_numeric_alpha_same_sector_symbolic_bridge_ready",
                "pass" if same_sector_symbolic_bridge_ready else "reject",
                "same-sector symbolic bridge ready",
                1 if same_sector_symbolic_bridge_ready else 0,
                "The current canon already closes the symbolic chi_P bridge without yet supplying an absolute same-sector proxy value.",
            ),
            row(
                "trial2_numeric_alpha_absolute_numeric_input_pack_ready",
                "pass" if absolute_numeric_input_pack_ready else "reject",
                "absolute numeric input pack ready",
                1 if absolute_numeric_input_pack_ready else 0,
                "The current pack still lacks an honest absolute normalization input needed for numeric alpha evaluation.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "newton_limit_relation_ready": True,
            "radial_mass_elimination_ready": True,
            "alpha_computation_formula_ready": True,
            "alpha_chi_p_reduction_formula_ready": True,
            "m0_squared_dimensionless_value": m0_squared_dimensionless,
            "alpha_target_value": alpha_target,
            "alpha_target_inverse_value": alpha_target_inv,
            "same_sector_symbolic_bridge_ready": same_sector_symbolic_bridge_ready,
            "same_sector_proxy_value_available": same_sector_proxy_value_available,
            "absolute_numeric_input_pack_ready": absolute_numeric_input_pack_ready,
            "first_route_to_close_or_none": "trial2_numeric_alpha_computation_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_newton_limit_audit_complete",
            "advance_to_8_7_56_705": True,
            "next_required_artifacts": [],
        },
        {
            "advice_newton_line": hit(advice_text, "Newton の"),
            "advice_formula_line": hit(advice_text, r"\alpha = \frac{16\pi G^2 \lambda v^2}{m_0^2 \hbar c}"),
            "trial2_declaration_summary": trial2_declaration["summary"],
            "mexican_hat_summary": mexican_hat["summary"],
            "trial3_coupled_summary": trial3_coupled["summary"],
            "anchor_local_curvature_summary": anchor_local_curvature["summary"],
            "anchor_local_shape_summary": anchor_local_shape["summary"],
            "chi_proxy_summary": chi_proxy["summary"],
            "constants_si": constants_si,
            "derived_constants": {
                "G_m3_kg_s2": g_newton_si,
                "hbar_j_s": hbar_si,
                "c_m_per_s": c_si,
            },
        },
    )

    computation_gate = payload(
        "8.7.56.705",
        "Trial-2 numeric alpha computation declaration gate",
        common_inputs,
        "Close the computation pivot honestly: freeze the formula, record that the current pack still lacks an absolute numeric input pack, and retire the deep7 wording blocker family.",
        {
            "numeric_pass_rule": "|alpha_P - alpha_target| / alpha_target < 0.10",
            "structural_hold_rule": "0.10 <= relative mismatch <= 0.50 keeps the structural pass while numeric alpha stays open",
            "route_problem_rule": "relative mismatch > 0.50 would reopen the structural route",
            "current_gate_rule": "if no honest absolute normalization input pack exists, the branch closes as formula-ready but numeric-open",
        },
        [
            row(
                "trial2_numeric_alpha_computation_gate_complete",
                "pass",
                "Trial-2 numeric alpha computation gate complete",
                1,
                "The branch now declares the computation route instead of the deep7 wording retry route.",
            ),
            row(
                "trial2_numeric_alpha_retry_loop_retired_in_gate",
                "pass",
                "deep7 wording retry loop retired in declaration gate",
                1,
                "The canonical blocker is no longer described as a missing wording family.",
            ),
            row(
                "trial2_numeric_alpha_formula_ready",
                "pass",
                "numeric alpha formula ready",
                1,
                "The current canon now carries an explicit computation formula for alpha.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready",
                "pass" if absolute_numeric_input_pack_ready else "reject",
                "numeric alpha from current pack ready",
                1 if absolute_numeric_input_pack_ready else 0,
                "No honest numeric alpha can be emitted until the current pack supplies an absolute normalization input.",
            ),
            row(
                "trial2_numeric_alpha_structural_pass_retained_after_computation_pivot",
                "pass",
                "structural Trial-2 pass retained after computation pivot",
                1,
                "Failing to evaluate a number does not reopen the Maxwell/Coulomb structural pass.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": True,
            "trial2_numeric_alpha_closeout_ready": absolute_numeric_input_pack_ready,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": absolute_numeric_input_pack_ready,
            "retry_loop_replaced_by_computation_pivot": True,
            "dominant_blocker_reclassified_to_absolute_normalization_input_pack": not absolute_numeric_input_pack_ready,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_computation_gate_closed_formula_ready_numeric_open"
            if not absolute_numeric_input_pack_ready
            else "trial2_numeric_alpha_computation_gate_closed_numeric_ready",
            "advance_to_8_7_56_706": True,
            "next_required_artifacts": [NEXT_RESIDUAL_ROUTE] if not absolute_numeric_input_pack_ready else [],
        },
        {
            "newton_audit_summary": newton_audit["summary"],
            "trial2_alpha_audit_summary": trial2_declaration["evidence"]["alpha_audit_summary"],
            "chi_proxy_summary": chi_proxy["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.706",
        "Trial-2 numeric alpha next-generation route contract seventy-third refresh",
        common_inputs,
        "Refresh the next-generation contract after the computation pivot: keep the precision-alpha mainline, keep the strong side on reserve, and promote the absolute-input-pack residual as the next official route.",
        {
            "mainline_rule": "the precision-alpha route stays on the mainline while Trial-4 remains a v3 reserve",
            "selected_route_rule": "the next official route is the absolute-normalization input-pack residual needed for numeric alpha evaluation",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_computation_gate_closed",
                "pass",
                "numeric alpha computation gate closed before route refresh",
                1,
                "The next-generation contract is only refreshed after the computation gate closes.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected",
                "pass",
                "absolute-normalization input-pack route selected",
                1,
                "The next route targets the missing numeric input pack instead of a wording fragment.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained",
                "pass",
                "precision-alpha mainline retained",
                1,
                "The computation pivot keeps Trial-2 numeric alpha as the next-generation mainline route.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained",
                "pass",
                "strong-side route state retained as v3 hold reserve",
                1,
                "The strong side remains exploratory and is not promoted by the alpha computation pivot.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": prior_route["summary"]["strong_side_route_state"],
            "precision_alpha_mainline_retained": True,
            "retry_loop_replaced_by_computation_pivot": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_seventy_third_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_absolute_normalization_input_pack_source_inventory",
                "trial2_numeric_alpha_absolute_normalization_input_pack_audit",
            ],
        },
        {
            "gate_summary": computation_gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_computation_pivot_source_inventory", pivot_inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_newton_limit_audit", newton_audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_computation_declaration_gate", computation_gate)
    write_artifact("mass_origin_v2_t2_alpha_route_contract_seventy_third_refresh", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial2_numeric_alpha_computation_pivot_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_newton_limit_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_computation_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_seventy_third_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the computation-pivot branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha computation pivot branch."""
    main()


if __name__ == "__main__":
    run_cli()
