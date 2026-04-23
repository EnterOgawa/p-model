#!/usr/bin/env python3
"""Generate 8.7.56.983-.986 Trial-2 numeric alpha unit-consistency audit artifacts."""

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
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

FINAL_SOURCE = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_source_inventory_metrics.json"
FINAL_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_audit_metrics.json"
FINAL_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_declaration_gate_metrics.json"
FINAL_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_second_refresh_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_final_computation_unit_consistency_audit"
CURRENT_ARTIFACT = "trial2_numeric_alpha_final_computation_unit_consistency_audit"
NEXT_ROUTE = "8.7.56.987"
NEXT_BRANCH = "8.7.56.987-.990"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge"

G_DIMS = {"m": 3, "kg": -1, "s": -2}
HBAR_DIMS = {"kg": 1, "m": 2, "s": -1}
C_DIMS = {"m": 1, "s": -1}
MASS_DIMS = {"kg": 1}
H0P_DIMS = {"s": -1}


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


# Function: scale a dimension dictionary by an integer factor.

def scale_dims(dims: dict[str, int], factor: int) -> dict[str, int]:
    """Scale a dimension dictionary by an integer factor."""
    return {key: value * factor for key, value in dims.items()}


# Function: add two dimension dictionaries.

def add_dims(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Add two dimension dictionaries."""
    merged: dict[str, int] = {}
    for key in set(left) | set(right):
        merged[key] = left.get(key, 0) + right.get(key, 0)

    return {key: value for key, value in merged.items() if value != 0}


# Function: subtract the right dimension dictionary from the left.

def sub_dims(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Subtract the right dimension dictionary from the left."""
    return add_dims(left, scale_dims(right, -1))


# Function: format a dimension dictionary for metrics output.

def format_dims(dims: dict[str, int]) -> str:
    """Format a dimension dictionary as a compact string."""
    if not dims:
        return "dimensionless"

    parts: list[str] = []
    for key in ("kg", "m", "s"):
        exponent = dims.get(key)
        if exponent is None:
            continue

        parts.append(f"{key}^{exponent}")

    return " ".join(parts)


# Function: determine whether a dimension dictionary is dimensionless.

def is_dimensionless(dims: dict[str, int]) -> bool:
    """Return whether the dimension dictionary is empty."""
    return not dims


# Function: execute the discrepancy / unit-consistency audit branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha discrepancy / unit-consistency audit branch."""
    for path in (
        ADVICE,
        PART1,
        PART2,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        FINAL_SOURCE,
        FINAL_AUDIT,
        FINAL_GATE,
        FINAL_ROUTE,
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
    final_source = read_json(FINAL_SOURCE)["summary"]
    final_audit = read_json(FINAL_AUDIT)["summary"]
    final_gate = read_json(FINAL_GATE)["summary"]
    final_route = read_json(FINAL_ROUTE)["summary"]

    prior_route_active = (
        final_gate["selected_residual_route"] == CURRENT_ROUTE
        and final_gate["missing_v2_artifact"] == CURRENT_ARTIFACT
        and final_route["selected_next_generation_route"] == CURRENT_ROUTE
    )

    advice_has_cbg_equals_one = (
        hit(advice_text, "C_bg = 1") is not None
        or hit(advice_text, r"C_{\rm bg} = 1") is not None
    )
    advice_has_zp_rule = hit(advice_text, r"Z_P = \frac{m_0^2}{(H_0^{(P)})^2}") is not None
    advice_has_alpha_rule = hit(advice_text, r"\alpha = \frac{4\pi G^2 Z_P}{\hbar c}") is not None
    part1_has_electron_identification = hit(part1_text, r"m_0 = \frac{m_e}{\mathcal{E}(\beta_1)}") is not None
    part2_has_h0p_law = hit(part2_text, r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]") is not None
    part3a_has_final_computation_route = hit(part3a_text, r"\alpha=\frac{4\pi G^2 Z_P}{\hbar c}") is not None
    part5_has_numeric_tension_wording = hit(part5_text, "numeric tension") is not None
    status_has_next_983 = hit(status_text, "8.7.56.983") is not None
    roadmap_has_983_branch = hit(roadmap_text, "`8.7.56.983-.986`") is not None

    raw_zp_dims = sub_dims(scale_dims(MASS_DIMS, 2), scale_dims(H0P_DIMS, 2))
    raw_alpha_dims = sub_dims(
        sub_dims(add_dims(scale_dims(G_DIMS, 2), raw_zp_dims), HBAR_DIMS),
        C_DIMS,
    )

    omega0_dims = sub_dims(add_dims(MASS_DIMS, scale_dims(C_DIMS, 2)), HBAR_DIMS)
    z_p_with_mass_frequency_bridge_dims = sub_dims(scale_dims(omega0_dims, 2), scale_dims(H0P_DIMS, 2))
    alpha_with_mass_frequency_bridge_dims = sub_dims(
        sub_dims(add_dims(scale_dims(G_DIMS, 2), z_p_with_mass_frequency_bridge_dims), HBAR_DIMS),
        C_DIMS,
    )

    raw_alpha_dimensionless = is_dimensionless(raw_alpha_dims)
    mass_frequency_bridge_alone_resolves_alpha_units = is_dimensionless(alpha_with_mass_frequency_bridge_dims)
    dimensionless_alpha_bridge_required = not raw_alpha_dimensionless
    unit_consistency_inventory_ready = all(
        [
            bool(final_source["final_computation_input_pack_ready"]),
            bool(final_audit["audit_ready"]),
            bool(final_gate["trial2_numeric_alpha_final_computation_performed"]),
            prior_route_active,
            advice_has_cbg_equals_one,
            advice_has_zp_rule,
            advice_has_alpha_rule,
            part1_has_electron_identification,
            part2_has_h0p_law,
            part3a_has_final_computation_route,
            part5_has_numeric_tension_wording,
            status_has_next_983,
            roadmap_has_983_branch,
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
        "final_source_json": display_path(FINAL_SOURCE),
        "final_audit_json": display_path(FINAL_AUDIT),
        "final_gate_json": display_path(FINAL_GATE),
        "final_route_json": display_path(FINAL_ROUTE),
    }

    inventory_payload = payload(
        "8.7.56.983",
        "Trial-2 numeric alpha final-computation unit-consistency source inventory",
        common_inputs,
        "Freeze the discrepancy / unit-consistency audit pack: the expert final-computation memo, the public raw alpha candidate, and the dimensional-consistency checks needed to decide whether the candidate is an honest dimensionless fine-structure constant in SI.",
        {
            "raw_computation_rule": "alpha_candidate = 4*pi*G^2*(m_0^2/(H_0^(P))^2)/(hbar*c)",
            "unit_audit_rule": "the fine-structure constant must be dimensionless before any numeric mismatch can be interpreted physically",
        },
        [
            row(
                "trial2_numeric_alpha_unit_consistency_inventory_complete",
                "pass" if unit_consistency_inventory_ready else "reject",
                "unit-consistency audit input-pack inventory complete",
                1 if unit_consistency_inventory_ready else 0,
                "The discrepancy audit requires the final-computation memo, the raw alpha candidate, and the current route contract in one pack.",
            ),
            row(
                "trial2_numeric_alpha_raw_final_computation_value_available",
                "pass",
                "raw final-computation value available",
                1,
                "The previous branch already produced one explicit numeric alpha candidate from the current public pack.",
            ),
            row(
                "trial2_numeric_alpha_raw_alpha_candidate_dimensionless_in_si",
                "pass" if raw_alpha_dimensionless else "reject",
                "raw alpha candidate dimensionless in SI",
                1 if raw_alpha_dimensionless else 0,
                f"The direct SI readout carries dimensions {format_dims(raw_alpha_dims)} rather than being dimensionless.",
            ),
            row(
                "trial2_numeric_alpha_mass_frequency_bridge_alone_resolves_units",
                "pass" if mass_frequency_bridge_alone_resolves_alpha_units else "reject",
                "mass-frequency bridge alone resolves alpha units",
                1 if mass_frequency_bridge_alone_resolves_alpha_units else 0,
                f"Replacing m0 by m0*c^2/hbar still leaves alpha with dimensions {format_dims(alpha_with_mass_frequency_bridge_dims)}.",
            ),
            row(
                "trial2_numeric_alpha_dimensionless_alpha_bridge_required",
                "pass" if dimensionless_alpha_bridge_required else "reject",
                "dimensionless-alpha bridge required",
                1 if dimensionless_alpha_bridge_required else 0,
                "A missing unit / normalization bridge is the honest next blocker if the current alpha candidate is not dimensionless.",
            ),
        ],
        {
            "inventory_ready": unit_consistency_inventory_ready,
            "raw_final_computation_value_available": True,
            "raw_alpha_candidate_dimension_vector_si": format_dims(raw_alpha_dims),
            "mass_frequency_bridge_alpha_dimension_vector_si": format_dims(alpha_with_mass_frequency_bridge_dims),
            "dimensionless_alpha_bridge_required": dimensionless_alpha_bridge_required,
            "first_route_to_close_or_none": NEXT_MISSING_ARTIFACT,
        },
        {
            "overall_status": "trial2_numeric_alpha_unit_consistency_input_pack_frozen",
            "advance_to_8_7_56_984": unit_consistency_inventory_ready,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "ai_context_current_step": ai_context["current_step"],
            "advice_cbg_hit": hit(advice_text, r"C_{\rm bg} = 1"),
            "advice_zp_rule_hit": hit(advice_text, r"Z_P = \frac{m_0^2}{(H_0^{(P)})^2}"),
            "advice_alpha_rule_hit": hit(advice_text, r"\alpha = \frac{4\pi G^2 Z_P}{\hbar c}"),
            "raw_alpha_summary": {
                "alpha_pmodel": final_audit["alpha_pmodel"],
                "alpha_target": final_audit["alpha_target"],
                "relative_error": final_audit["relative_error"],
                "result_class": final_audit["result_class"],
            },
        },
    )

    audit_payload = payload(
        "8.7.56.984",
        "Trial-2 numeric alpha final-computation unit-consistency audit",
        common_inputs,
        "Audit whether the raw final-computation candidate can be interpreted as an honest fine-structure constant in SI, or whether a missing unit / normalization bridge remains upstream of any physical rejection.",
        {
            "raw_zp_rule": "Z_P = m_0^2 / (H_0^(P))^2",
            "raw_alpha_rule": "alpha = 4*pi*G^2*Z_P / (hbar*c)",
            "mass_frequency_bridge_probe": "omega_0 = m_0*c^2/hbar",
        },
        [
            row(
                "trial2_numeric_alpha_unit_consistency_audit_complete",
                "pass",
                "unit-consistency audit complete",
                1,
                "The current branch evaluates the raw candidate numerically and dimensionally in one place.",
            ),
            row(
                "trial2_numeric_alpha_raw_zp_dimensionless_in_si",
                "pass" if is_dimensionless(raw_zp_dims) else "reject",
                "raw Z_P candidate dimensionless in SI",
                1 if is_dimensionless(raw_zp_dims) else 0,
                f"The direct bridge gives Z_P units {format_dims(raw_zp_dims)}.",
            ),
            row(
                "trial2_numeric_alpha_raw_alpha_dimensionless_in_si",
                "pass" if raw_alpha_dimensionless else "reject",
                "raw alpha candidate dimensionless in SI",
                1 if raw_alpha_dimensionless else 0,
                f"The direct final-computation readout gives alpha units {format_dims(raw_alpha_dims)}.",
            ),
            row(
                "trial2_numeric_alpha_mass_frequency_bridge_alone_resolves_alpha_units",
                "pass" if mass_frequency_bridge_alone_resolves_alpha_units else "reject",
                "mass-frequency bridge alone resolves alpha units",
                1 if mass_frequency_bridge_alone_resolves_alpha_units else 0,
                f"Even after omega_0 = m_0*c^2/hbar, alpha carries units {format_dims(alpha_with_mass_frequency_bridge_dims)}.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_missing_dimensionless_alpha_bridge",
                "pass" if dimensionless_alpha_bridge_required else "reject",
                "dominant blocker is missing dimensionless-alpha bridge",
                1 if dimensionless_alpha_bridge_required else 0,
                "The raw numeric mismatch cannot yet be promoted to a physical no-go if the candidate is not dimensionless in SI.",
            ),
        ],
        {
            "audit_ready": unit_consistency_inventory_ready,
            "raw_alpha_candidate_dimension_vector_si": format_dims(raw_alpha_dims),
            "raw_zp_candidate_dimension_vector_si": format_dims(raw_zp_dims),
            "raw_alpha_candidate_dimensionless_in_si": raw_alpha_dimensionless,
            "mass_frequency_bridge_alpha_dimension_vector_si": format_dims(alpha_with_mass_frequency_bridge_dims),
            "mass_frequency_bridge_alone_resolves_alpha_units": mass_frequency_bridge_alone_resolves_alpha_units,
            "dimensionless_alpha_bridge_required": dimensionless_alpha_bridge_required,
            "raw_alpha_candidate_value": final_audit["alpha_pmodel"],
            "raw_alpha_target_value": final_audit["alpha_target"],
            "raw_alpha_relative_error": final_audit["relative_error"],
            "first_route_to_close_after_audit_or_none": NEXT_MISSING_ARTIFACT,
        },
        {
            "overall_status": "trial2_numeric_alpha_unit_consistency_audited",
            "advance_to_8_7_56_985": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "raw_final_computation_audit_summary": final_audit,
            "raw_zp_dims": raw_zp_dims,
            "raw_alpha_dims": raw_alpha_dims,
            "omega0_dims": omega0_dims,
            "alpha_with_mass_frequency_bridge_dims": alpha_with_mass_frequency_bridge_dims,
        },
    )

    gate_payload = payload(
        "8.7.56.985",
        "Trial-2 numeric alpha final-computation unit-consistency declaration gate",
        common_inputs,
        "Reclassify the raw final-computation result honestly: keep the diagnostic numeric value, but gate the official alpha closeout on a missing dimensionless-alpha bridge rather than a token retry or a premature physical rejection.",
        {
            "gate_rule": "a raw numeric candidate is not an honest alpha closeout if the candidate is not dimensionless in SI",
            "residual_rule": "the next blocker is an explicit dimensionless-alpha bridge / normalization rule",
        },
        [
            row(
                "trial2_numeric_alpha_unit_consistency_gate_complete",
                "pass",
                "unit-consistency declaration gate complete",
                1,
                "The official state is reclassified after the dimensional audit.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_unit_audit",
                "pass" if raw_alpha_dimensionless else "reject",
                "numeric alpha from current pack ready after unit audit",
                1 if raw_alpha_dimensionless else 0,
                "A raw number alone is insufficient if the candidate still carries SI dimensions.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready_after_unit_audit",
                "pass" if raw_alpha_dimensionless and final_audit["relative_error"] < 0.10 else "reject",
                "Trial-2 numeric alpha closeout ready after unit audit",
                1 if raw_alpha_dimensionless and final_audit["relative_error"] < 0.10 else 0,
                "Closeout requires both dimensional consistency and a passing numeric match.",
            ),
            row(
                "trial2_numeric_alpha_raw_numeric_tension_reclassified_as_precanonical_diagnostic",
                "pass" if dimensionless_alpha_bridge_required else "reject",
                "raw numeric tension reclassified as pre-canonical diagnostic",
                1 if dimensionless_alpha_bridge_required else 0,
                "The previous tension result is retained as evidence, but not yet treated as a final physical reject.",
            ),
            row(
                "trial2_numeric_alpha_current_blocker_is_dimensionless_alpha_bridge",
                "pass" if dimensionless_alpha_bridge_required else "reject",
                "current blocker is dimensionless-alpha bridge",
                1 if dimensionless_alpha_bridge_required else 0,
                "The honest next artifact is an explicit unit / normalization bridge that makes alpha dimensionless.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": bool(final_gate["trial2_numeric_alpha_computation_formula_ready"]),
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": bool(final_gate["trial2_numeric_alpha_absolute_normalization_dictionary_ready"]),
            "trial2_numeric_alpha_raw_final_computation_value_available": True,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": raw_alpha_dimensionless,
            "trial2_numeric_alpha_closeout_ready": raw_alpha_dimensionless and final_audit["relative_error"] < 0.10,
            "trial2_numeric_alpha_final_computation_performed": True,
            "trial2_numeric_alpha_final_computation_result_class": "precanonical_unit_incomplete" if dimensionless_alpha_bridge_required else final_gate["trial2_numeric_alpha_final_computation_result_class"],
            "trial2_numeric_alpha_retry_loop_retired": True,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_unit_consistency_gate_closed",
            "advance_to_8_7_56_986": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "unit_consistency_audit_summary": audit_payload["summary"],
            "raw_final_gate_summary": final_gate,
            "raw_final_route_summary": final_route,
        },
    )

    route_payload = payload(
        "8.7.56.986",
        "Trial-2 numeric alpha route contract one-hundred-forty-third refresh",
        common_inputs,
        "Refresh the next-generation contract after the unit-consistency audit: keep Trial-2 numeric alpha on the mainline, keep the strong side on reserve, and promote the dimensionless-alpha bridge as the next official blocker family.",
        {
            "next_route_rule": "the next route must determine the missing normalization / unit bridge that makes the raw candidate an honest dimensionless alpha",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_forty_third_refresh_complete",
                "pass",
                "route contract one-hundred-forty-third refresh complete",
                1,
                "The discrepancy audit has been converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_dimensionless_alpha_bridge",
                "pass" if dimensionless_alpha_bridge_required else "reject",
                "next route selected as dimensionless-alpha bridge",
                1 if dimensionless_alpha_bridge_required else 0,
                "The next route is no longer a generic unit-consistency question; it is the explicit bridge that makes alpha dimensionless.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_unit_audit",
                "pass" if final_route["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after unit audit",
                1 if final_route["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline despite the unit blocker.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_unit_audit",
                "pass" if final_route["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after unit audit",
                1 if final_route["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains on reserve and is not promoted by the current alpha audit.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": final_route["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(final_route["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(final_route["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(final_route["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(final_route["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_forty_third_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "gate_summary": gate_payload["summary"],
            "raw_final_route_summary": final_route,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_source_inventory",
        inventory_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_audit",
        audit_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_declaration_gate",
        gate_payload,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_third_refresh",
        route_payload,
    )

    print("[done] 8.7.56.983-.986 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_third_refresh_metrics.json")
    print(f" - raw_alpha_dims = {format_dims(raw_alpha_dims)}")
    print(f" - alpha_after_mass_frequency_bridge_dims = {format_dims(alpha_with_mass_frequency_bridge_dims)}")


# Function: run the discrepancy / unit-consistency audit branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha discrepancy / unit-consistency audit branch."""
    main()


if __name__ == "__main__":
    run_cli()
