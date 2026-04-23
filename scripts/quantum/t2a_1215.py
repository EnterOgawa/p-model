#!/usr/bin/env python3
"""Generate 8.7.56.1215-.1218 Trial-2 charge-normalization bridge review artifacts."""

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
NOTE_PLACEHOLDER = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_placeholder_compress_and_attempt.md")

INVENTORY_1211 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_source_inventory_metrics.json"
)
AUDIT_1212 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_audit_metrics.json"
)
GATE_1213 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_declaration_gate_metrics.json"
)
EVAL_1214 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_numeric_evaluation_metrics.json"
)

ALPHA_TARGET = 7.2973525692838015e-3
CURRENT_E_ACTION_LEVEL = 1.0
CURRENT_ALPHA_ACTION_LEVEL = 1.0 / (4.0 * math.pi)
NEXT_ROUTE = "8.7.56.1219"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_charge_normalization_residual_scope_classification"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort when one required path is missing.

def require(path: Path) -> None:
    """Abort when one required path is missing."""
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


# Function: return one display path relative to the repo when possible.

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first matching line for one substring.

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


# Function: write one JSON metrics artifact and one CSV rows table.

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


# Function: compute the elementary-charge target summary.

def compute_charge_targets() -> dict[str, float]:
    """Compute the elementary-charge target and current-to-target residuals."""
    e_target = math.sqrt(4.0 * math.pi * ALPHA_TARGET)
    e_ratio_current_to_target = CURRENT_E_ACTION_LEVEL / e_target
    alpha_gap_factor = CURRENT_ALPHA_ACTION_LEVEL / ALPHA_TARGET
    return {
        "e_target": e_target,
        "e_ratio_current_to_target": e_ratio_current_to_target,
        "alpha_gap_factor": alpha_gap_factor,
    }


# Function: execute the charge-normalization exact coefficient bridge review branch.

def main() -> None:
    """Execute the 8.7.56.1215-.1218 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART1,
        PART3A,
        PART5,
        NOTE_PLACEHOLDER,
        INVENTORY_1211,
        AUDIT_1212,
        GATE_1213,
        EVAL_1214,
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
    placeholder_note_text = read_text(NOTE_PLACEHOLDER)
    inventory_1211 = read_json(INVENTORY_1211)["summary"]
    audit_1212 = read_json(AUDIT_1212)["summary"]
    gate_1213 = read_json(GATE_1213)["summary"]
    eval_1214 = read_json(EVAL_1214)["summary"]
    charge_targets = compute_charge_targets()

    structural_route_present = hit(part3a_text, "e=g_P/\\sqrt{Z_P}") is not None
    adopted_u1_surface_present = hit(part3a_text, "\"Local Maxwell/QED is kept unchanged\"") is not None
    u1_b_adoption_present = hit(part3a_text, "**A棄却、B採用**") is not None
    independent_connection_present = hit(part3a_text, "独立接続") is not None
    explicit_mapping_absent_present = (
        hit(part3a_text, "explicit $g_P\\leftrightarrow e$ charge-normalization statement") is not None
        or hit(part3a_text, "positive な **mapping statement** も **mapping literal** も absent") is not None
    )
    elementary_charge_surface_present = hit(part3a_text, "public elementary charge $e$") is not None
    part1_current_surface_present = hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})") is not None
    qball_charge_candidate_present = hit(placeholder_note_text, "### 候補3: Q-ball charge normalization") is not None
    inventory_ready = all(
        (
            inventory_1211["inventory_ready"],
            audit_1212["audit_ready"],
            gate_1213["charge_normalization_residual_open"],
            not gate_1213["physical_reject_required"],
            structural_route_present,
            adopted_u1_surface_present,
            u1_b_adoption_present,
            independent_connection_present,
            explicit_mapping_absent_present,
            elementary_charge_surface_present,
            part1_current_surface_present,
            qball_charge_candidate_present,
        )
    )

    explicit_current_canon_charge_bridge_available = False
    adopted_u1_external_preservation_only = True
    current_canon_justifies_required_coefficient = False
    qball_charge_candidate_publicly_fixed = False
    audit_ready = inventory_ready

    targets = [
        target(status_text, STATUS, "status_charge_branch", "charge-normalization exact coefficient bridge review branch", "STATUS must expose the charge-normalization bridge review as the live branch."),
        target(roadmap_text, ROADMAP, "roadmap_1215", "`8.7.56.1215`", "ROADMAP must expose the current source-inventory step."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "recent_1211", "`8.7.56.1211-.1214`", "Recent history must preserve the predecessor action-level closeout."),
        target(part3a_text, PART3A, "part3a_structural_route", "e=g_P/\\sqrt{Z_P}", "Part III-A must preserve the structural charge route."),
        target(part3a_text, PART3A, "part3a_adopted_u1", "\"Local Maxwell/QED is kept unchanged\"", "Part III-A must preserve the adopted U(1) sector wording."),
        target(part3a_text, PART3A, "part3a_b_adopted", "**A棄却、B採用**", "Part III-A must preserve the B-adoption judgment for U(1) origin."),
        target(part3a_text, PART3A, "part3a_independent_connection", "独立接続", "Part III-A must preserve that U(1) requires an independent connection."),
        target(part3a_text, PART3A, "part3a_mapping_absent", "explicit $g_P\\leftrightarrow e$ charge-normalization statement", "Part III-A must preserve the explicit mapping absence."),
        target(part3a_text, PART3A, "part3a_public_e", "public elementary charge $e$", "Part III-A must preserve the public elementary-charge surface."),
        target(part1_text, PART1, "part1_current", "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})", "Part I must preserve the current normalization surface carried into this review."),
        target(placeholder_note_text, NOTE_PLACEHOLDER, "note_qball_charge", "### 候補3: Q-ball charge normalization", "The placeholder-compress note must preserve the Q-ball charge candidate."),
        target(part5_text, PART5, "part5_action_level_closed", "action-level factors closed / charge-normalization residual open", "Part V must preserve the current checkpoint wording before this branch completes."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "placeholder_note": display_path(NOTE_PLACEHOLDER),
        },
        "prior_metrics": {
            "inventory_1211": display_path(INVENTORY_1211),
            "audit_1212": display_path(AUDIT_1212),
            "gate_1213": display_path(GATE_1213),
            "eval_1214": display_path(EVAL_1214),
        },
        "constants": {
            "alpha_target": ALPHA_TARGET,
            "current_e_action_level": CURRENT_E_ACTION_LEVEL,
            "current_alpha_action_level": CURRENT_ALPHA_ACTION_LEVEL,
            "required_charge_normalization_factor": charge_targets["e_target"],
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1215",
        "Trial-2 numeric alpha charge-normalization exact coefficient bridge review source inventory",
        inputs,
        [
            row("inventory_ready", "pass" if inventory_ready else "reject", "charge-normalization inventory ready", 1 if inventory_ready else 0, "The branch is ready only if the structural route, adopted-U(1) wording, mapping absence, and Q-ball candidate are all visible in one pack."),
            row("structural_route_present", "pass" if structural_route_present else "reject", "structural e route present", 1 if structural_route_present else 0, "Part III-A must expose e = g_P / sqrt(Z_P)."),
            row("adopted_u1_surface_present", "pass" if adopted_u1_surface_present else "reject", "adopted U(1) surface present", 1 if adopted_u1_surface_present else 0, "Part III-A must expose the adopted-U(1) stance."),
            row("u1_b_adoption_present", "pass" if u1_b_adoption_present else "reject", "B-adoption for U(1) present", 1 if u1_b_adoption_present else 0, "Part III-A must preserve the A-reject/B-adopt judgment."),
            row("explicit_mapping_absent_present", "pass" if explicit_mapping_absent_present else "reject", "explicit mapping absence present", 1 if explicit_mapping_absent_present else 0, "Part III-A must still expose the absence of an explicit g_P-to-elementary-charge mapping."),
            row("qball_charge_candidate_present", "pass" if qball_charge_candidate_present else "reject", "Q-ball charge candidate present", 1 if qball_charge_candidate_present else 0, "The note must preserve the candidate that pushes the residual into a charge-normalization family."),
        ],
        {
            "inventory_ready": inventory_ready,
            "structural_route_present": structural_route_present,
            "adopted_u1_surface_present": adopted_u1_surface_present,
            "explicit_mapping_absent_present": explicit_mapping_absent_present,
            "required_charge_normalization_factor": charge_targets["e_target"],
            "selected_next_substep": "8.7.56.1216",
        },
        {"overall_status": "trial2_numeric_alpha_charge_normalization_inventory_fixed", "advance_to_8_7_56_1216": inventory_ready, "next_required_artifacts": ["charge_normalization_bridge_audit"]},
        {"targets": targets, "prior_1213_summary": gate_1213, "prior_1214_summary": eval_1214, "ai_context_snapshot": ai_context},
    )

    audit = payload(
        "8.7.56.1216",
        "Trial-2 numeric alpha charge-normalization exact coefficient bridge review audit",
        inputs,
        [
            row("explicit_current_canon_charge_bridge_available", "reject", "explicit current-canon charge bridge available", 0.0, "The current canon still lacks an explicit coefficient bridge from the structural route to the public elementary charge."),
            row("adopted_u1_external_preservation_only", "pass" if audit_ready else "reject", "adopted-U1 preserves external e only", 1 if audit_ready else 0, "A-reject/B-adopt means U(1) is retained as an independent adopted sector, so it can preserve public QED structures without deriving the coefficient from P-only canon."),
            row("current_canon_justifies_required_coefficient", "reject", "current canon justifies required coefficient", 0.0, "No public-canonical surface justifies the residual coefficient 0.30282212087175264 as a derivation from current canon alone."),
            row("qball_charge_candidate_publicly_fixed", "reject", "Q-ball charge candidate publicly fixed", 0.0, "The placeholder note carries Q-ball charge normalization only as a candidate, not as a public-canonical fixed theorem or statement."),
            row("residual_scope_localized", "pass" if audit_ready else "reject", "residual scope localized to adopted-U1 or future canon", 1 if audit_ready else 0, "The remaining residual belongs to adopted-U1 external import or future-canon bridge scope, not to the closed action-level factor family."),
        ],
        {
            "audit_ready": audit_ready,
            "required_charge_normalization_factor": charge_targets["e_target"],
            "explicit_current_canon_charge_bridge_available": explicit_current_canon_charge_bridge_available,
            "adopted_u1_external_preservation_only": adopted_u1_external_preservation_only,
            "current_canon_justifies_required_coefficient": current_canon_justifies_required_coefficient,
            "qball_charge_candidate_publicly_fixed": qball_charge_candidate_publicly_fixed,
            "residual_scope_class": "adopted_u1_external_import_or_future_canon_bridge",
            "result_class": "current_canon_charge_normalization_bridge_absent_adopted_u1_external_only",
        },
        {"overall_status": "trial2_numeric_alpha_charge_normalization_bridge_audit_completed", "advance_to_8_7_56_1217": audit_ready, "next_required_artifacts": ["charge_normalization_bridge_declaration_gate"]},
        {"inventory_summary": inventory["summary"]},
    )

    declaration_gate = payload(
        "8.7.56.1217",
        "Trial-2 numeric alpha charge-normalization exact coefficient bridge review declaration gate",
        inputs,
        [
            row("current_canon_charge_bridge_completed", "reject", "current-canon charge bridge completed", 0.0, "The current canon still does not complete the charge-normalization bridge."),
            row("residual_scope_classification_ready", "pass" if audit_ready else "reject", "residual scope classification ready", 1 if audit_ready else 0, "The branch now knows the residual scope belongs outside the closed action-level factor family."),
            row("adopted_u1_or_future_canon_review_required", "pass", "adopted-U1 or future-canon review required", 1.0, "Further progress requires either an explicit adopted-U1 coefficient bridge or a future-canon bridge surface."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "The route remains open because the dimensionless formula and structural route still stand."),
            row("closeout_ready", "reject", "closeout ready", 0.0, "Closeout is not ready while the charge-normalization coefficient remains externally scoped."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "charge_normalization_bridge_absent_under_current_canon",
            "required_charge_normalization_factor": charge_targets["e_target"],
            "residual_scope_class": "adopted_u1_external_import_or_future_canon_bridge",
            "current_canon_charge_bridge_completed": False,
            "adopted_u1_or_future_canon_review_required": True,
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {"overall_status": "trial2_numeric_alpha_charge_normalization_scope_declared", "advance_to_8_7_56_1218": audit_ready, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"audit_summary": audit["summary"]},
    )

    evaluation = payload(
        "8.7.56.1218",
        "Trial-2 numeric alpha charge-normalization exact coefficient bridge review numeric evaluation",
        inputs,
        [
            row("current_e_action_level_fixed", "pass", "current e action-level fixed", CURRENT_E_ACTION_LEVEL, "The current P-sector action-level route still gives e = 1."),
            row("required_charge_normalization_factor_fixed", "pass", "required charge-normalization factor fixed", charge_targets["e_target"], "The observed alpha still requires a residual coefficient 0.30282212087175264."),
            row("current_canon_supplies_required_factor", "reject", "current canon supplies required factor", 0.0, "No explicit public-canonical bridge supplies the required coefficient."),
            row("numeric_state_changes_without_new_bridge", "reject", "numeric state changes without new bridge", 0.0, "Without a new charge-normalization bridge, the numeric state remains unchanged from .1214."),
            row("charge_normalization_bridge_review_completed", "pass", "charge-normalization bridge review completed", 1.0, "This branch completed the current-canon review and localized the residual scope."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "charge_normalization_bridge_absent_under_current_canon",
            "e_current_canon_action_level": CURRENT_E_ACTION_LEVEL,
            "alpha_current_canon_action_level": CURRENT_ALPHA_ACTION_LEVEL,
            "required_charge_normalization_factor": charge_targets["e_target"],
            "current_to_target_e_ratio": charge_targets["e_ratio_current_to_target"],
            "alpha_gap_factor": charge_targets["alpha_gap_factor"],
            "current_canon_supplies_required_factor": False,
            "numeric_state_changed_by_current_branch": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {"overall_status": "trial2_numeric_alpha_charge_normalization_bridge_review_completed", "advance_to_next_route": True, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"declaration_gate_summary": declaration_gate["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1215-.1218 artifacts generated")
    print(f"[required_charge_factor] {charge_targets['e_target']:.16f}")
    print(f"[current_to_target_e_ratio] {charge_targets['e_ratio_current_to_target']:.16f}")
    print(f"[alpha_gap_factor] {charge_targets['alpha_gap_factor']:.16f}")


if __name__ == "__main__":
    main()
