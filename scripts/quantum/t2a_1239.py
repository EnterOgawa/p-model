#!/usr/bin/env python3
"""Generate 8.7.56.1239-.1242 Trial-2 adopted-U(1) dictionary-contract artifacts.

Purpose:
    Freeze the current official contract after `.1235-.1238` split the retained
    Q-ball `charge_proxy` rows from the adopted elementary charge unit `q`.
    This branch asks whether the present pack already provides a direct
    `q = e` dictionary, or whether a separate charge-field translation is still
    required to connect adopted-U(1) `q` with structural
    `e = g_P / sqrt(Z_P)`.

Inputs:
    - Current operational docs and the Part I / Part III-A / Part V paper
      surfaces
    - The retained Q-ball mapping and discrete inversion metrics
    - The prior `.1235-.1238` charge-unit dictionary review metrics
    - The earlier `.1039-.1042` implicit field-normalization translation audit

Outputs:
    - Four machine-readable metrics payloads under output/public/quantum/

Assumptions:
    - No new free parameter is introduced
    - This branch only formalizes the dictionary contract and does not claim a
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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_TRANSLATION_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "current_canon_reconciliation_audit_metrics.json"
)
QBALL_MAPPING = PUBLIC_OUT / "mass_origin_qball_charge_mapping_statement_freeze_metrics.json"
QBALL_DISCRETE = PUBLIC_OUT / "mass_origin_qball_charge_discrete_frequency_inversion_metrics.json"
AUDIT_1236 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_audit_metrics.json"
)
GATE_1237 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_declaration_gate_metrics.json"
)
EVAL_1238 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_numeric_evaluation_metrics.json"
)

ALPHA_TARGET = 7.2973525692838015e-3
CURRENT_ACTION_LEVEL_E = 1.0
CURRENT_ACTION_LEVEL_ALPHA = CURRENT_ACTION_LEVEL_E**2 / (4.0 * math.pi)
REQUIRED_CHARGE_UNIT = math.sqrt(4.0 * math.pi * ALPHA_TARGET)
NEXT_ROUTE = "8.7.56.1243"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_adopted_u1_charge_field_translation_review"


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


# Function: locate the first matching line for one pattern list.

def hit_any(text: str, patterns: tuple[str, ...]) -> dict | None:
    """Return the first matching line for one pattern list."""
    for pattern in patterns:
        evidence = hit(text, pattern)
        if evidence is not None:
            return evidence

    return None


# Function: locate the first matching line across multiple files.

def cross_hit(files: tuple[tuple[str, Path, str], ...], patterns: tuple[str, ...]) -> dict | None:
    """Return the first matching line across multiple files."""
    for file_key, path, text in files:
        evidence = hit_any(text, patterns)
        if evidence is not None:
            return {
                "file_key": file_key,
                "file": display_path(path),
                "pattern": evidence["pattern"],
                "line": evidence["line"],
                "text": evidence["text"],
            }

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


# Function: execute the adopted-U(1) dictionary-contract branch.

def main() -> None:
    """Execute the 8.7.56.1239-.1242 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART1,
        PART3A,
        PART5,
        PRIOR_TRANSLATION_AUDIT,
        QBALL_MAPPING,
        QBALL_DISCRETE,
        AUDIT_1236,
        GATE_1237,
        EVAL_1238,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    prior_translation_summary = read_json(PRIOR_TRANSLATION_AUDIT)["summary"]
    qball_mapping = read_json(QBALL_MAPPING)
    qball_discrete = read_json(QBALL_DISCRETE)
    audit_1236 = read_json(AUDIT_1236)["summary"]
    gate_1237 = read_json(GATE_1237)["summary"]
    eval_1238 = read_json(EVAL_1238)["summary"]

    ground_state_charge_proxy = float(audit_1236["ground_state_charge_proxy"])
    direct_q_equals_e_evidence = cross_hit(
        (
            ("part1", PART1, part1_text),
            ("part3a", PART3A, part3a_text),
            ("part5", PART5, part5_text),
        ),
        ("$q=e$", "$q = e$", "q=e", "q = e", "q\\leftrightarrow e", "q ↔ e"),
    )
    explicit_charge_field_translation_evidence = cross_hit(
        (
            ("part1", PART1, part1_text),
            ("part3a", PART3A, part3a_text),
            ("part5", PART5, part5_text),
        ),
        ("A_{\\rm charge}", "A_{\\rm can}", "A_charge", "A_can"),
    )

    targets = [
        target(part1_text, PART1, "part1_canonical_photon_field", "A_\\mu=\\delta P_\\mu^T/\\sqrt{Z_P}", "Part I must preserve the canonical photon-branch normalization."),
        target(part3a_text, PART3A, "part3a_structural_charge_route", "e=g_P/\\sqrt{Z_P}", "Part III-A must preserve the structural Trial-2 charge route."),
        target(part3a_text, PART3A, "part3a_covariant_derivative", "D_\\mu=\\partial_\\mu+i q A_\\mu", "Part III-A must preserve the adopted-U(1) covariant derivative with charge unit q."),
        target(part3a_text, PART3A, "part3a_independent_connection", "独立接続", "Part III-A must still state that the adopted-U(1) connection is independent at the origin-analysis level."),
        target(part3a_text, PART3A, "part3a_a_reject_b_adopt", "**A棄却、B採用**", "Part III-A must preserve the A-reject/B-adopt judgment."),
        target(part3a_text, PART3A, "part3a_separate_follow_through", "separate follow-through", "Part III-A must keep local U(1) origin and numeric alpha normalization as separate follow-through items."),
        target(part5_text, PART5, "part5_next_step_1239", "8.7.56.1239-.1242", "Part V must preserve that this branch is the current official next step."),
    ]

    canonical_photon_field_present = targets[0]["present"]
    structural_charge_route_present = targets[1]["present"]
    adopted_u1_covariant_derivative_present = targets[2]["present"]
    independent_connection_required_present = targets[3]["present"]
    a_reject_b_adopt_present = targets[4]["present"]
    separate_follow_through_present = targets[5]["present"]
    part5_current_step_present = targets[6]["present"]
    qball_discrete_rule_present = qball_discrete["formulas"]["charge_discretization_rule"] == "Q_n = n q"
    qball_charge_quantum_normalization_present = (
        qball_mapping["summary"]["charge_quantum_normalization"] == "elementary_charge_unit_q"
    )
    prior_field_normalization_translation_supported = bool(
        prior_translation_summary["implicit_field_normalization_translation_supported"]
    )
    direct_q_equals_e_literal_available = direct_q_equals_e_evidence is not None
    explicit_charge_field_translation_available = explicit_charge_field_translation_evidence is not None
    same_symbol_a_reuse_without_dictionary = (
        canonical_photon_field_present
        and adopted_u1_covariant_derivative_present
        and not direct_q_equals_e_literal_available
        and not explicit_charge_field_translation_available
    )
    translation_dictionary_required_inferred = (
        independent_connection_required_present
        and a_reject_b_adopt_present
        and structural_charge_route_present
        and canonical_photon_field_present
        and adopted_u1_covariant_derivative_present
        and qball_charge_quantum_normalization_present
        and not direct_q_equals_e_literal_available
    )
    inventory_ready = all(
        (
            canonical_photon_field_present,
            structural_charge_route_present,
            adopted_u1_covariant_derivative_present,
            independent_connection_required_present,
            a_reject_b_adopt_present,
            separate_follow_through_present,
            part5_current_step_present,
            qball_discrete_rule_present,
            qball_charge_quantum_normalization_present,
            prior_field_normalization_translation_supported,
        )
    )
    vacuum_polarization_secondary_lane_retained = (
        gate_1237["secondary_residual_lane"] == "adopted_u1_vacuum_polarization_external_import"
    )
    current_canon_completes_charge_unit_dictionary = (
        direct_q_equals_e_literal_available or explicit_charge_field_translation_available
    )

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
        },
        "prior_metrics": {
            "prior_translation_audit_1040": display_path(PRIOR_TRANSLATION_AUDIT),
            "qball_mapping": display_path(QBALL_MAPPING),
            "qball_discrete": display_path(QBALL_DISCRETE),
            "audit_1236": display_path(AUDIT_1236),
            "gate_1237": display_path(GATE_1237),
            "eval_1238": display_path(EVAL_1238),
        },
        "constants": {
            "current_action_level_e": CURRENT_ACTION_LEVEL_E,
            "current_action_level_alpha": CURRENT_ACTION_LEVEL_ALPHA,
            "required_charge_unit_q": REQUIRED_CHARGE_UNIT,
            "ground_state_charge_number_proxy": ground_state_charge_proxy,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1239",
        "Trial-2 numeric alpha adopted-U(1) charge-unit dictionary contract source inventory",
        inputs,
        [
            row("dictionary_contract_inventory_ready", "pass" if inventory_ready else "reject", "dictionary-contract inventory ready", 1 if inventory_ready else 0, "The contract branch is ready only if the structural e route, adopted-U(1) q route, Q-ball charge rule, and canonical photon field are all visible in one pack."),
            row("canonical_photon_field_present", "pass" if canonical_photon_field_present else "reject", "canonical photon field present", 1 if canonical_photon_field_present else 0, "Part I must preserve A_mu = delta P_mu^T / sqrt(Z_P)."),
            row("structural_charge_route_present", "pass" if structural_charge_route_present else "reject", "structural charge route present", 1 if structural_charge_route_present else 0, "Part III-A must preserve e = g_P / sqrt(Z_P)."),
            row("adopted_u1_covariant_derivative_present", "pass" if adopted_u1_covariant_derivative_present else "reject", "adopted-U(1) covariant derivative present", 1 if adopted_u1_covariant_derivative_present else 0, "Part III-A must preserve D_mu = partial_mu + i q A_mu."),
            row("independent_connection_required_present", "pass" if independent_connection_required_present else "reject", "independent connection requirement present", 1 if independent_connection_required_present else 0, "Part III-A must still say the adopted-U(1) connection is independent at the origin-analysis level."),
            row("qball_charge_discretization_rule_present", "pass" if qball_discrete_rule_present else "reject", "Q-ball charge discretization rule present", 1 if qball_discrete_rule_present else 0, "The retained discrete inversion must still say Q_n = n q."),
            row("qball_charge_quantum_normalization_present", "pass" if qball_charge_quantum_normalization_present else "reject", "Q-ball charge quantum normalization present", 1 if qball_charge_quantum_normalization_present else 0, "The retained Q-ball mapping must still fix q as the adopted elementary charge unit."),
            row("prior_field_normalization_translation_supported", "pass" if prior_field_normalization_translation_supported else "reject", "prior field-normalization translation supported", 1 if prior_field_normalization_translation_supported else 0, "Earlier current-canon reconciliation already allowed an implicit normalization-translation reading on the photon side."),
        ],
        {
            "inventory_ready": inventory_ready,
            "canonical_photon_field_present": canonical_photon_field_present,
            "structural_charge_route_present": structural_charge_route_present,
            "adopted_u1_covariant_derivative_present": adopted_u1_covariant_derivative_present,
            "independent_connection_required_present": independent_connection_required_present,
            "qball_charge_quantum_normalization_present": qball_charge_quantum_normalization_present,
            "prior_field_normalization_translation_supported": prior_field_normalization_translation_supported,
            "selected_next_substep": "8.7.56.1240",
        },
        {
            "overall_status": "trial2_numeric_alpha_charge_unit_dictionary_contract_inventory_fixed",
            "advance_to_8_7_56_1240": inventory_ready,
            "next_required_artifacts": ["charge_unit_dictionary_contract_audit"],
        },
        {
            "targets": targets,
            "prior_translation_summary": prior_translation_summary,
            "prior_1237_summary": gate_1237,
            "prior_1238_summary": eval_1238,
            "status_hits": {
                "status_next_1239": hit(status_text, "8.7.56.1239"),
                "roadmap_branch_1239": hit(roadmap_text, "`8.7.56.1239-.1242`"),
                "work_history_1235_entry": hit(work_history_recent_text, "8.7.56.1235-.1238"),
            },
        },
    )

    audit = payload(
        "8.7.56.1240",
        "Trial-2 numeric alpha adopted-U(1) charge-unit dictionary contract audit",
        inputs,
        [
            row("direct_q_equals_structural_e_literal_available", "pass" if direct_q_equals_e_literal_available else "reject", "direct q = e literal available", 1 if direct_q_equals_e_literal_available else 0, "No current public-canonical surface writes q = e directly."),
            row("explicit_charge_field_translation_available", "pass" if explicit_charge_field_translation_available else "reject", "explicit charge-field translation available", 1 if explicit_charge_field_translation_available else 0, "No current public-canonical surface writes an explicit A_charge / A_can translation."),
            row("same_symbol_a_reuse_without_dictionary", "pass" if same_symbol_a_reuse_without_dictionary else "reject", "same-symbol A reuse without explicit dictionary", 1 if same_symbol_a_reuse_without_dictionary else 0, "This is an inference from public surfaces: adopted-U(1) and structural photon sections both reuse A_mu, but no normalization dictionary is written."),
            row("translation_dictionary_required_inferred", "pass" if translation_dictionary_required_inferred else "reject", "translation dictionary required inferred", 1 if translation_dictionary_required_inferred else 0, "This is an inference from public surfaces: independent adopted-U(1) connection plus separate canonical photon normalization means a dictionary is still needed to connect q to structural e."),
            row("current_canon_completes_charge_unit_dictionary", "pass" if current_canon_completes_charge_unit_dictionary else "reject", "current canon completes charge-unit dictionary", 1 if current_canon_completes_charge_unit_dictionary else 0, "The current pack still does not complete the q-to-e dictionary."),
            row("vacuum_polarization_secondary_lane_retained", "pass" if vacuum_polarization_secondary_lane_retained else "reject", "vacuum-polarization secondary lane retained", 1 if vacuum_polarization_secondary_lane_retained else 0, "The adopted-U(1) vacuum-polarization analog stays secondary while the dictionary gap remains primary."),
        ],
        {
            "audit_ready": inventory_ready,
            "direct_q_equals_structural_e_literal_available": direct_q_equals_e_literal_available,
            "explicit_charge_field_translation_available": explicit_charge_field_translation_available,
            "same_symbol_a_reuse_without_dictionary": same_symbol_a_reuse_without_dictionary,
            "translation_dictionary_required_inferred": translation_dictionary_required_inferred,
            "current_canon_completes_charge_unit_dictionary": current_canon_completes_charge_unit_dictionary,
            "result_class": "adopted_u1_charge_unit_dictionary_contract_incomplete",
        },
        {
            "overall_status": "trial2_numeric_alpha_charge_unit_dictionary_contract_audit_completed",
            "advance_to_8_7_56_1241": inventory_ready,
            "next_required_artifacts": ["charge_unit_dictionary_contract_declaration_gate"],
        },
        {
            "inventory_summary": inventory["summary"],
            "direct_q_equals_e_evidence": direct_q_equals_e_evidence,
            "explicit_charge_field_translation_evidence": explicit_charge_field_translation_evidence,
        },
    )

    declaration_gate = payload(
        "8.7.56.1241",
        "Trial-2 numeric alpha adopted-U(1) charge-unit dictionary contract declaration gate",
        inputs,
        [
            row("charge_unit_dictionary_contract_completed", "pass" if inventory_ready else "reject", "charge-unit dictionary contract completed", 1 if inventory_ready else 0, "This branch completed the contract-level classification even though the dictionary itself remains open."),
            row("primary_residual_lane_fixed", "pass", "primary residual lane fixed", 1.0, "The primary residual lane remains the adopted-U(1) charge-unit dictionary."),
            row("secondary_vacuum_polarization_lane_retained", "pass" if vacuum_polarization_secondary_lane_retained else "reject", "secondary vacuum-polarization lane retained", 1 if vacuum_polarization_secondary_lane_retained else 0, "Vacuum-polarization stays a secondary carry-over lane."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "The route remains open because the structural charge route still stands."),
            row("closeout_ready", "reject", "closeout ready", 0.0, "Closeout is not ready while the q-to-e dictionary remains incomplete."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "adopted_u1_charge_unit_dictionary_missing",
            "charge_unit_dictionary_contract_completed": True,
            "direct_q_equals_structural_e_literal_available": direct_q_equals_e_literal_available,
            "explicit_charge_field_translation_available": explicit_charge_field_translation_available,
            "translation_dictionary_required_inferred": translation_dictionary_required_inferred,
            "primary_residual_lane": "adopted_u1_charge_unit_dictionary",
            "secondary_residual_lane": "adopted_u1_vacuum_polarization_external_import",
            "reserve_residual_lane": "future_canon_bridge",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_charge_unit_dictionary_contract_declared",
            "advance_to_8_7_56_1242": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"], "prior_1237_summary": gate_1237},
    )

    evaluation = payload(
        "8.7.56.1242",
        "Trial-2 numeric alpha adopted-U(1) charge-unit dictionary contract numeric evaluation",
        inputs,
        [
            row("current_action_level_e_fixed", "pass", "current action-level e fixed", CURRENT_ACTION_LEVEL_E, "The structural action-level route still fixes e = 1."),
            row("current_action_level_alpha_fixed", "pass", "current action-level alpha fixed", CURRENT_ACTION_LEVEL_ALPHA, "The structural action-level route still fixes alpha = 1/(4 pi)."),
            row("required_charge_unit_q_fixed", "pass", "required charge unit q fixed", REQUIRED_CHARGE_UNIT, "Observed alpha still requires q = sqrt(4 pi alpha_target) = 0.30282212087175264."),
            row("ground_state_charge_number_proxy_fixed", "pass", "ground-state charge-number proxy fixed", ground_state_charge_proxy, "The retained ground-state Q-ball row remains a Q/q proxy near integer mode 1."),
            row("numeric_state_changed_by_contract", "reject", "numeric state changed by dictionary contract", 0.0, "This branch only formalizes the dictionary contract; it does not change the numeric alpha state."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "adopted_u1_charge_unit_dictionary_missing",
            "e_current_canon_action_level": CURRENT_ACTION_LEVEL_E,
            "alpha_current_canon_action_level": CURRENT_ACTION_LEVEL_ALPHA,
            "required_charge_unit_q": REQUIRED_CHARGE_UNIT,
            "ground_state_charge_number_proxy": ground_state_charge_proxy,
            "direct_q_equals_structural_e_literal_available": direct_q_equals_e_literal_available,
            "explicit_charge_field_translation_available": explicit_charge_field_translation_available,
            "translation_dictionary_required_inferred": translation_dictionary_required_inferred,
            "numeric_state_changed_by_current_branch": False,
            "numeric_state_class": "dictionary_contract_fixed_numeric_state_unchanged",
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_charge_unit_dictionary_contract_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"declaration_gate_summary": declaration_gate["summary"], "prior_1238_summary": eval_1238},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_contract_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_charge_unit_dictionary_contract_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1239-.1242 artifacts generated")
    print(f"[required_charge_unit_q] {REQUIRED_CHARGE_UNIT:.16f}")
    print(f"[translation_dictionary_required_inferred] {int(translation_dictionary_required_inferred)}")
    print(f"[selected_next_route] {NEXT_ROUTE_NAME}")


if __name__ == "__main__":
    main()
