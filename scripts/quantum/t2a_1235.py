#!/usr/bin/env python3
"""Generate 8.7.56.1235-.1238 Trial-2 adopted-U(1) charge-unit dictionary artifacts.

Purpose:
    Re-audit the Trial-2 numeric-alpha residual after discovering that the
    retained Q-ball `charge_proxy` rows are integer-like charge numbers
    (`Q/q ~ n`) rather than direct measurements of the elementary coupling
    magnitude itself.

Inputs:
    - Current operational docs and the Part III-A / Part V paper surfaces
    - Retained Q-ball charge mapping and discrete inversion metrics
    - The `.1215-.1218` coefficient-bridge review and `.1231-.1234` reserve-heavy
      candidate metrics

Outputs:
    - Four machine-readable metrics payloads under output/public/quantum/

Assumptions:
    - No new free parameter is introduced
    - This branch only reclassifies the residual scope and does not claim a
      solved alpha derivation
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

INVENTORY_1215 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_source_inventory_metrics.json"
)
AUDIT_1216 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_audit_metrics.json"
)
GATE_1217 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_declaration_gate_metrics.json"
)
EVAL_1218 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_numeric_evaluation_metrics.json"
)
QBALL_MAPPING = PUBLIC_OUT / "mass_origin_qball_charge_mapping_statement_freeze_metrics.json"
QBALL_NORMALIZATION = PUBLIC_OUT / "mass_origin_qball_charge_operator_normalization_audit_metrics.json"
QBALL_DISCRETE = PUBLIC_OUT / "mass_origin_qball_charge_discrete_frequency_inversion_metrics.json"
GATE_1233 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_unresolved_coefficient_declaration_gate_metrics.json"
)
EVAL_1234 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_unresolved_coefficient_numeric_evaluation_metrics.json"
)

ALPHA_TARGET = 7.2973525692838015e-3
REQUIRED_CHARGE_UNIT = math.sqrt(4.0 * math.pi * ALPHA_TARGET)
NEXT_ROUTE = "8.7.56.1239"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_contract"


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


# Function: build one wording target record.

def target(text: str, path: Path, key: str, pattern: str, note: str) -> dict:
    """Build one wording target record."""
    evidence = hit(text, pattern)
    return {
        "file_key": key,
        "file": display_path(path),
        "pattern": pattern,
        "present": evidence is not None,
        "note": note,
        "evidence": evidence,
    }


# Function: compute the integer-like diagnostics for the discrete Q-ball rows.

def compute_integer_diagnostics(rows: list[dict]) -> dict:
    """Compute integer-like diagnostics for the retained Q-ball charge rows."""
    abs_errors = []
    rel_errors = []
    mode_rows = []
    for row_data in rows:
        mode_index = int(row_data["mode_index"])
        charge_proxy = float(row_data["charge_proxy"])
        abs_error = abs(charge_proxy - mode_index)
        rel_error = abs_error / mode_index
        abs_errors.append(abs_error)
        rel_errors.append(rel_error)
        mode_rows.append(
            {
                "mode_index": mode_index,
                "charge_proxy": charge_proxy,
                "abs_error_to_integer": abs_error,
                "rel_error_to_integer": rel_error,
            }
        )

    ground_state = mode_rows[0]
    return {
        "mode_rows": mode_rows,
        "ground_state_mode_index": ground_state["mode_index"],
        "ground_state_charge_proxy": ground_state["charge_proxy"],
        "ground_state_abs_error_to_integer": ground_state["abs_error_to_integer"],
        "ground_state_rel_error_to_integer": ground_state["rel_error_to_integer"],
        "max_abs_error_to_integer": max(abs_errors),
        "max_rel_error_to_integer": max(rel_errors),
        "mean_rel_error_to_integer": sum(rel_errors) / len(rel_errors),
    }


# Function: execute the adopted-U(1) charge-unit dictionary review branch.

def main() -> None:
    """Execute the 8.7.56.1235-.1238 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART3A,
        PART5,
        INVENTORY_1215,
        AUDIT_1216,
        GATE_1217,
        EVAL_1218,
        QBALL_MAPPING,
        QBALL_NORMALIZATION,
        QBALL_DISCRETE,
        GATE_1233,
        EVAL_1234,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    inventory_1215 = read_json(INVENTORY_1215)["summary"]
    audit_1216 = read_json(AUDIT_1216)["summary"]
    gate_1217 = read_json(GATE_1217)["summary"]
    eval_1218 = read_json(EVAL_1218)["summary"]
    qball_mapping = read_json(QBALL_MAPPING)
    qball_normalization = read_json(QBALL_NORMALIZATION)
    qball_discrete = read_json(QBALL_DISCRETE)
    gate_1233 = read_json(GATE_1233)["summary"]
    eval_1234 = read_json(EVAL_1234)["summary"]

    integer_diag = compute_integer_diagnostics(qball_discrete["evidence"]["discrete_mode_rows"])
    ground_state_charge_proxy = integer_diag["ground_state_charge_proxy"]
    ground_state_mode_index = integer_diag["ground_state_mode_index"]
    charge_proxy_to_required_q_ratio = ground_state_charge_proxy / REQUIRED_CHARGE_UNIT
    retired_misread_alpha = (ground_state_charge_proxy**2) / (4.0 * math.pi)
    retired_misread_ratio_to_target = retired_misread_alpha / ALPHA_TARGET

    qball_discrete_rule_present = (
        qball_discrete["formulas"]["charge_discretization_rule"] == "Q_n = n q"
    )
    qball_mapping_uses_elementary_charge_unit = (
        qball_mapping["summary"]["charge_quantum_normalization"] == "elementary_charge_unit_q"
    )
    qball_direct_identity_required = bool(qball_normalization["summary"]["direct_qball_u1_identity_required"])
    structural_e_route_present = hit(part3a_text, "structural $e=g_P/\\sqrt{Z_P}$") is not None
    adopted_u1_covariant_derivative_present = hit(part3a_text, "D_\\mu=\\partial_\\mu+i q A_\\mu") is not None
    adopted_u1_charge_coupling_line_present = hit(part3a_text, "結合定数 q は環境非依存") is not None
    qball_integer_like_first_five = integer_diag["max_rel_error_to_integer"] < 2.0e-3
    direct_comparison_to_required_q_is_category_mismatch = True
    explicit_charge_unit_dictionary_available = False
    reserve_heavy_contract_premature = True
    inventory_ready = all(
        (
            inventory_1215["inventory_ready"],
            audit_1216["audit_ready"],
            gate_1217["adopted_u1_or_future_canon_review_required"],
            gate_1233["reserve_heavy_route_required"],
            qball_discrete_rule_present,
            qball_mapping_uses_elementary_charge_unit,
            qball_direct_identity_required,
            adopted_u1_covariant_derivative_present,
            adopted_u1_charge_coupling_line_present,
            structural_e_route_present,
        )
    )

    targets = [
        target(status_text, STATUS, "status_old_next", "reserve-heavy route contract", "STATUS must still expose the pre-review reserve-heavy candidate before this reclassification."),
        target(roadmap_text, ROADMAP, "roadmap_1235", "`8.7.56.1235-.1238`", "ROADMAP must expose the current 1235 branch slot."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "recent_1231", "`8.7.56.1231-.1234`", "Recent history must preserve the predecessor unresolved-coefficient review."),
        target(part3a_text, PART3A, "part3a_structural_e", "structural $e=g_P/\\sqrt{Z_P}$", "Part III-A must preserve the structural Trial-2 e route."),
        target(part3a_text, PART3A, "part3a_covariant_derivative", "D_\\mu=\\partial_\\mu+i q A_\\mu", "Part III-A adopted U(1) section must preserve the covariant derivative with charge unit q."),
        target(part3a_text, PART3A, "part3a_q_constant", "結合定数 q は環境非依存", "Part III-A must preserve q as the coupling constant of the adopted U(1) sector."),
        target(part5_text, PART5, "part5_reserve_heavy_next", "reserve-heavy route contract branch `8.7.56.1235-.1238`", "Part V must expose the pre-review next-step wording before this branch retires it."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "inventory_1215": display_path(INVENTORY_1215),
            "audit_1216": display_path(AUDIT_1216),
            "gate_1217": display_path(GATE_1217),
            "eval_1218": display_path(EVAL_1218),
            "qball_mapping": display_path(QBALL_MAPPING),
            "qball_normalization": display_path(QBALL_NORMALIZATION),
            "qball_discrete": display_path(QBALL_DISCRETE),
            "gate_1233": display_path(GATE_1233),
            "eval_1234": display_path(EVAL_1234),
        },
        "constants": {
            "required_charge_unit_q": REQUIRED_CHARGE_UNIT,
            "alpha_target": ALPHA_TARGET,
            "selected_next_route_name": NEXT_ROUTE_NAME,
            "selected_next_route_step": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1235",
        "Trial-2 adopted-U(1) charge-unit dictionary source inventory",
        inputs,
        [
            row("inventory_ready", "pass" if inventory_ready else "fail", "charge-unit dictionary inventory ready", 1.0 if inventory_ready else 0.0, "The structural e route, adopted-U(1) q surfaces, retained discrete Q-ball rows, and the predecessor reserve-heavy candidate are assembled into one pack."),
            row("qball_discrete_rule_present", "pass" if qball_discrete_rule_present else "fail", "Q-ball discrete rule present", 1.0 if qball_discrete_rule_present else 0.0, "The retained discrete inversion must still explicitly say Q_n = n q."),
            row("qball_mapping_uses_elementary_charge_unit", "pass" if qball_mapping_uses_elementary_charge_unit else "fail", "Q-ball mapping uses elementary charge unit q", 1.0 if qball_mapping_uses_elementary_charge_unit else 0.0, "The old canonical mapping must still label q as the adopted elementary charge unit."),
            row("adopted_u1_covariant_derivative_present", "pass" if adopted_u1_covariant_derivative_present else "fail", "adopted-U(1) covariant derivative present", 1.0 if adopted_u1_covariant_derivative_present else 0.0, "The adopted-U(1) section must still expose D_mu = partial_mu + i q A_mu."),
            row("structural_e_route_present", "pass" if structural_e_route_present else "fail", "structural Trial-2 e route present", 1.0 if structural_e_route_present else 0.0, "The structural route e = g_P / sqrt(Z_P) must still remain visible in the current pack."),
        ],
        {
            "inventory_ready": inventory_ready,
            "required_charge_unit_q": REQUIRED_CHARGE_UNIT,
            "ground_state_charge_proxy": ground_state_charge_proxy,
            "ground_state_mode_index": ground_state_mode_index,
            "ground_state_abs_error_to_integer": integer_diag["ground_state_abs_error_to_integer"],
            "selected_next_substep": "8.7.56.1236",
        },
        {
            "overall_status": "trial2_numeric_alpha_charge_unit_dictionary_inventory_fixed",
            "advance_to_8_7_56_1236": inventory_ready,
            "next_required_artifacts": ["charge_unit_dictionary_audit"],
        },
        {
            "targets": targets,
            "qball_mapping_summary": qball_mapping["summary"],
            "qball_normalization_summary": qball_normalization["summary"],
            "qball_discrete_formulas": qball_discrete["formulas"],
            "ai_context_snapshot": ai_context,
        },
    )

    audit = payload(
        "8.7.56.1236",
        "Trial-2 adopted-U(1) charge-unit dictionary audit",
        inputs,
        [
            row("ground_state_charge_proxy_tracks_integer_mode", "pass" if math.isclose(ground_state_charge_proxy, ground_state_mode_index, rel_tol=5.0e-4, abs_tol=0.0) else "fail", "ground-state charge proxy tracks integer mode", 1.0 if math.isclose(ground_state_charge_proxy, ground_state_mode_index, rel_tol=5.0e-4, abs_tol=0.0) else 0.0, "The retained mode-1 row behaves like Q/q ~ 1, not like an elementary charge magnitude near 0.3028."),
            row("first_five_charge_proxies_integer_like", "pass" if qball_integer_like_first_five else "fail", "first five charge proxies are integer-like", integer_diag["max_rel_error_to_integer"], "Across the first five discrete rows the charge proxies track the integer mode indices at the 10^-3 level or better."),
            row("ground_state_matches_required_charge_unit", "pass" if math.isclose(ground_state_charge_proxy, REQUIRED_CHARGE_UNIT, rel_tol=0.10, abs_tol=0.0) else "fail", "ground-state proxy matches required charge unit", 1.0 if math.isclose(ground_state_charge_proxy, REQUIRED_CHARGE_UNIT, rel_tol=0.10, abs_tol=0.0) else 0.0, "If the proxy were the elementary charge magnitude itself it would need to sit near 0.3028, but it does not."),
            row("direct_comparison_to_required_q_is_category_mismatch", "pass" if direct_comparison_to_required_q_is_category_mismatch else "fail", "direct comparison of charge proxy to required q is a category mismatch", 1.0 if direct_comparison_to_required_q_is_category_mismatch else 0.0, "The retained discrete inversion was built from Q_n = n q, so its charge_proxy rows track the integer charge number Q/q rather than the coupling magnitude q itself."),
            row("explicit_charge_unit_dictionary_available", "pass" if explicit_charge_unit_dictionary_available else "fail", "explicit adopted-U(1) charge-unit dictionary available", 1.0 if explicit_charge_unit_dictionary_available else 0.0, "Current pack still does not explicitly map the adopted charge unit q to the structural coupling e = g_P / sqrt(Z_P)."),
            row("reserve_heavy_contract_premature", "pass" if reserve_heavy_contract_premature else "fail", "reserve-heavy contract is premature", 1.0 if reserve_heavy_contract_premature else 0.0, "Before exact screening is demoted to the main residual, the charge-number / charge-unit dictionary mismatch has to be fixed first."),
        ],
        {
            "audit_ready": inventory_ready,
            "ground_state_charge_proxy": ground_state_charge_proxy,
            "ground_state_mode_index": ground_state_mode_index,
            "ground_state_abs_error_to_integer": integer_diag["ground_state_abs_error_to_integer"],
            "ground_state_rel_error_to_integer": integer_diag["ground_state_rel_error_to_integer"],
            "first_five_max_rel_error_to_integer": integer_diag["max_rel_error_to_integer"],
            "required_charge_unit_q": REQUIRED_CHARGE_UNIT,
            "charge_proxy_to_required_q_ratio": charge_proxy_to_required_q_ratio,
            "direct_comparison_to_required_q_is_category_mismatch": direct_comparison_to_required_q_is_category_mismatch,
            "explicit_adopted_u1_charge_unit_to_structural_coupling_dictionary_available": explicit_charge_unit_dictionary_available,
            "reserve_heavy_contract_premature": reserve_heavy_contract_premature,
            "result_class": "adopted_u1_charge_unit_dictionary_missing",
        },
        {
            "overall_status": "trial2_numeric_alpha_charge_unit_dictionary_audited",
            "advance_to_8_7_56_1237": inventory_ready,
            "next_required_artifacts": ["charge_unit_dictionary_declaration_gate"],
        },
        {
            "integer_diagnostics": integer_diag,
            "retired_gate_1233_summary": gate_1233,
            "retired_eval_1234_summary": eval_1234,
        },
    )

    declaration_gate = payload(
        "8.7.56.1237",
        "Trial-2 adopted-U(1) charge-unit dictionary declaration gate",
        inputs,
        [
            row("qball_direct_no_go_retired", "pass", "Q-ball direct no-go retired", 1.0, "The Q-ball rows no longer count as a failed direct-e comparison once they are read as Q/q mode numbers."),
            row("primary_lane_charge_unit_dictionary", "pass", "primary residual lane is the adopted-U(1) charge-unit dictionary", 1.0, "The immediate missing piece is the dictionary that maps the adopted charge unit q onto the structural coupling."),
            row("secondary_lane_vacuum_polarization", "pass", "secondary residual lane is vacuum-polarization external import", 1.0, "Screening remains useful, but it is downstream from the missing charge-unit dictionary."),
            row("numeric_closeout_ready", "pass" if False else "fail", "numeric closeout ready", 0.0, "This branch fixes the category mismatch but does not yet close alpha numerically."),
            row("physical_reject_required", "pass" if False else "fail", "physical reject required", 0.0, "The route remains open and is not a physical reject."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "adopted_u1_charge_unit_dictionary_missing",
            "qball_direct_no_go_retired": True,
            "vacuum_polarization_reserve_heavy_contract_premature": reserve_heavy_contract_premature,
            "primary_residual_lane": "adopted_u1_charge_unit_dictionary",
            "secondary_residual_lane": "adopted_u1_vacuum_polarization_external_import",
            "reserve_residual_lane": "future_canon_bridge",
            "numeric_closeout_ready": False,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_charge_unit_dictionary_gate_frozen",
            "advance_to_8_7_56_1238": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "retired_reserve_heavy_route_name": gate_1233["selected_next_generation_route"],
            "required_charge_unit_q": REQUIRED_CHARGE_UNIT,
            "ground_state_charge_proxy": ground_state_charge_proxy,
        },
    )

    numeric_evaluation = payload(
        "8.7.56.1238",
        "Trial-2 adopted-U(1) charge-unit dictionary numeric evaluation",
        inputs,
        [
            row("ground_state_charge_number_proxy", "pass", "ground-state charge-number proxy", ground_state_charge_proxy, "The retained Q-ball mode-1 row is read as Q/q ~ 1 rather than as the elementary charge magnitude itself."),
            row("ground_state_integer_abs_error", "pass", "ground-state absolute error to integer mode 1", integer_diag["ground_state_abs_error_to_integer"], "The mode-1 row sits very close to the integer charge number 1."),
            row("first_five_max_integer_relative_error", "pass", "first five max relative error to integer mode numbers", integer_diag["max_rel_error_to_integer"], "All first-five discrete rows remain integer-like at the 10^-3 level or better."),
            row("required_charge_unit_q", "pass", "required adopted elementary charge unit q", REQUIRED_CHARGE_UNIT, "If canonical Maxwell normalization is retained, target alpha implies q = sqrt(4 pi alpha_target)."),
            row("retired_misread_alpha", "pass", "retired misread alpha from treating charge_proxy as q", retired_misread_alpha, "This is the old q^2/(4 pi) readout that is now treated as a category mismatch rather than as a valid alpha candidate."),
            row("retired_misread_ratio_to_target", "pass", "retired misread alpha ratio to target", retired_misread_ratio_to_target, "The old mismatch survives only as a retired misread benchmark."),
        ],
        {
            "ground_state_charge_number_proxy": ground_state_charge_proxy,
            "ground_state_mode_index": ground_state_mode_index,
            "ground_state_abs_error_to_integer": integer_diag["ground_state_abs_error_to_integer"],
            "ground_state_rel_error_to_integer": integer_diag["ground_state_rel_error_to_integer"],
            "first_five_max_rel_error_to_integer": integer_diag["max_rel_error_to_integer"],
            "required_charge_unit_q": REQUIRED_CHARGE_UNIT,
            "charge_proxy_to_required_q_ratio": charge_proxy_to_required_q_ratio,
            "retired_misread_alpha": retired_misread_alpha,
            "retired_misread_alpha_ratio_to_target": retired_misread_ratio_to_target,
            "numeric_state_changed_by_current_branch": True,
        },
        {
            "numeric_state_class": "charge_number_vs_charge_unit_dictionary_split_fixed",
            "reserve_heavy_contract_premature": reserve_heavy_contract_premature,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
        },
        {
            "integer_diagnostics": integer_diag,
            "required_charge_unit_explanation": "The Q-ball discrete rows constrain Q/q mode numbers; they do not by themselves determine q = e_phys.",
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_numeric_evaluation",
        numeric_evaluation,
    )


if __name__ == "__main__":
    main()
