#!/usr/bin/env python3
"""Generate 8.7.56.1219-.1222 Trial-2 residual-scope classification artifacts."""

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

NOTE_QBALL = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_qball_noether_charge.md")
NOTE_PLACEHOLDER = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_placeholder_compress_and_attempt.md")

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

ALPHA_TARGET = 7.2973525692838015e-3
REQUIRED_E = math.sqrt(4.0 * math.pi * ALPHA_TARGET)
ACTION_LEVEL_E = 1.0
ACTION_LEVEL_ALPHA = 1.0 / (4.0 * math.pi)
NEXT_ROUTE = "8.7.56.1223"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_adopted_u1_external_import_primary_lane_contract"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: require one input path to exist before continuing.

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


# Function: return the ground-state charge row from the retained Q-ball artifact.

def ground_state_charge_row(qball_discrete: dict) -> dict:
    """Return the mode_index=1 row from the Q-ball discrete inversion artifact."""
    for row_data in qball_discrete["evidence"]["discrete_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return row_data

    raise SystemExit("[fail] missing Q-ball ground-state charge row")


# Function: execute the residual-scope classification branch.

def main() -> None:
    """Execute the 8.7.56.1219-.1222 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART3A,
        PART5,
        NOTE_QBALL,
        NOTE_PLACEHOLDER,
        INVENTORY_1215,
        AUDIT_1216,
        GATE_1217,
        EVAL_1218,
        QBALL_MAPPING,
        QBALL_NORMALIZATION,
        QBALL_DISCRETE,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    qball_note_text = read_text(NOTE_QBALL)
    placeholder_note_text = read_text(NOTE_PLACEHOLDER)

    inventory_1215 = read_json(INVENTORY_1215)["summary"]
    audit_1216 = read_json(AUDIT_1216)["summary"]
    gate_1217 = read_json(GATE_1217)["summary"]
    eval_1218 = read_json(EVAL_1218)["summary"]
    qball_mapping = read_json(QBALL_MAPPING)
    qball_normalization = read_json(QBALL_NORMALIZATION)
    qball_discrete = read_json(QBALL_DISCRETE)
    ground_state = ground_state_charge_row(qball_discrete)

    qball_ground_state_charge_proxy = float(ground_state["charge_proxy"])
    qball_ground_state_energy_proxy = float(ground_state["energy_proxy"])
    qball_alpha_candidate = (qball_ground_state_charge_proxy**2) / (4.0 * math.pi)
    qball_charge_ratio_to_required = qball_ground_state_charge_proxy / REQUIRED_E
    qball_alpha_ratio_to_target = qball_alpha_candidate / ALPHA_TARGET
    qball_charge_relative_error = abs(qball_ground_state_charge_proxy - REQUIRED_E) / REQUIRED_E
    qball_action_level_charge_delta = abs(qball_ground_state_charge_proxy - ACTION_LEVEL_E)

    qball_note_available = hit(qball_note_text, "# Trial-2 numeric α: Q-ball Noether charge candidate") is not None
    qball_noether_formula_present = hit(qball_note_text, "j^\\mu = i(P^{*\\nu}\\partial^\\mu P_\\nu - P^\\nu\\partial^\\mu P_\\nu^*)") is not None
    qball_factor_task_present = hit(qball_note_text, "q_{(1,0,0,0)} \\stackrel{?}{=} 0.30282") is not None
    qball_mapping_statement_available = (
        qball_mapping["summary"]["u1_charge_quantization_to_qball_charge_mapping"] == "available"
    )
    qball_direct_identity_required = bool(qball_normalization["summary"]["direct_qball_u1_identity_required"])
    qball_normalization_freedom_available = bool(
        qball_normalization["summary"]["charge_operator_normalization_freedom_available"]
    )
    qball_ground_state_charge_row_available = True
    qball_ground_state_energy_matches_beta1 = math.isclose(
        qball_ground_state_energy_proxy,
        1.0191595901506567,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    inventory_ready = all(
        (
            inventory_1215["inventory_ready"],
            audit_1216["audit_ready"],
            gate_1217["adopted_u1_or_future_canon_review_required"],
            qball_note_available,
            qball_noether_formula_present,
            qball_factor_task_present,
            qball_mapping_statement_available,
            qball_direct_identity_required,
            qball_ground_state_charge_row_available,
            qball_ground_state_energy_matches_beta1,
        )
    )

    qball_ground_state_charge_matches_required_factor = math.isclose(
        qball_ground_state_charge_proxy,
        REQUIRED_E,
        rel_tol=0.10,
        abs_tol=0.0,
    )
    qball_ground_state_charge_matches_action_level = math.isclose(
        qball_ground_state_charge_proxy,
        ACTION_LEVEL_E,
        rel_tol=5.0e-4,
        abs_tol=0.0,
    )
    qball_note_supports_independent_qball_lane = False
    primary_residual_lane = "adopted_u1_external_import"
    secondary_residual_lane = "future_canon_bridge"
    reserve_residual_lane = "qball_noether_charge_candidate"

    targets = [
        target(status_text, STATUS, "status_1219", "charge-normalization residual scope classification branch", "STATUS must expose the current residual-scope classification branch."),
        target(roadmap_text, ROADMAP, "roadmap_1219", "`8.7.56.1219`", "ROADMAP must expose the current source-inventory step."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "recent_1215", "`8.7.56.1215-.1218`", "Recent history must preserve the predecessor bridge-review branch."),
        target(part3a_text, PART3A, "part3a_adopted_u1", "U(1) を独立に採用し", "Part III-A must preserve the adopted-U(1) wording."),
        target(part3a_text, PART3A, "part3a_charge_quantization", "観測上の離散性を**採用条件として固定", "Part III-A must preserve adopted charge quantization."),
        target(part3a_text, PART3A, "part3a_beta1", "\\mathcal{E}(\\beta_1)=1.0191595901506567", "Part III-A must preserve the ground-state proxy value used by electron identification."),
        target(part5_text, PART5, "part5_current_state", "charge-normalization bridge absent under current canon", "Part V must preserve the pre-classification current state."),
        target(qball_note_text, NOTE_QBALL, "note_qball_header", "# Trial-2 numeric α: Q-ball Noether charge candidate", "The new note must preserve the Q-ball Noether-charge route."),
        target(qball_note_text, NOTE_QBALL, "note_qball_formula", "j^\\mu = i(P^{*\\nu}\\partial^\\mu P_\\nu - P^\\nu\\partial^\\mu P_\\nu^*)", "The new note must expose the Noether-current formula."),
        target(placeholder_note_text, NOTE_PLACEHOLDER, "placeholder_factor", "required $e$ factor", "The placeholder-compress note must preserve the target coefficient 0.30282."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "qball_note": display_path(NOTE_QBALL),
            "placeholder_note": display_path(NOTE_PLACEHOLDER),
        },
        "prior_metrics": {
            "inventory_1215": display_path(INVENTORY_1215),
            "audit_1216": display_path(AUDIT_1216),
            "gate_1217": display_path(GATE_1217),
            "eval_1218": display_path(EVAL_1218),
            "qball_mapping": display_path(QBALL_MAPPING),
            "qball_normalization": display_path(QBALL_NORMALIZATION),
            "qball_discrete": display_path(QBALL_DISCRETE),
        },
        "constants": {
            "required_charge_normalization_factor": REQUIRED_E,
            "alpha_target": ALPHA_TARGET,
            "action_level_e": ACTION_LEVEL_E,
            "action_level_alpha": ACTION_LEVEL_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1219",
        "Trial-2 numeric alpha charge-normalization residual scope classification source inventory",
        inputs,
        [
            row("inventory_ready", "pass" if inventory_ready else "reject", "residual-scope inventory ready", 1 if inventory_ready else 0, "The branch is ready only if the new Q-ball Noether-charge note, the old Q-ball/U(1) identity artifacts, and the .1215-.1218 bridge-review pack are visible together."),
            row("qball_note_available", "pass" if qball_note_available else "reject", "Q-ball Noether-charge note available", 1 if qball_note_available else 0, "The new note must be present."),
            row("qball_mapping_statement_available", "pass" if qball_mapping_statement_available else "reject", "Q-ball/U(1) mapping statement available", 1 if qball_mapping_statement_available else 0, "The old public artifact must already freeze Q-ball Noether charge coincides with adopted U(1) charge."),
            row("qball_direct_identity_required", "pass" if qball_direct_identity_required else "reject", "direct Q-ball/U(1) identity required", 1 if qball_direct_identity_required else 0, "The old normalization audit must already remove multiplicative freedom."),
            row("qball_ground_state_charge_row_available", "pass", "ground-state charge row available", 1.0, "The old discrete inversion artifact must expose the mode_index=1 charge row."),
            row("qball_ground_state_energy_matches_beta1", "pass" if qball_ground_state_energy_matches_beta1 else "reject", "ground-state energy matches beta1 proxy", 1 if qball_ground_state_energy_matches_beta1 else 0, "The old Q-ball ground-state energy must match the Trial-2 electron-identification proxy E(beta1)."),
        ],
        {
            "inventory_ready": inventory_ready,
            "required_charge_normalization_factor": REQUIRED_E,
            "qball_ground_state_charge_proxy": qball_ground_state_charge_proxy,
            "qball_ground_state_energy_proxy": qball_ground_state_energy_proxy,
            "qball_direct_identity_required": qball_direct_identity_required,
            "qball_normalization_freedom_available": qball_normalization_freedom_available,
            "selected_next_substep": "8.7.56.1220",
        },
        {
            "overall_status": "trial2_numeric_alpha_residual_scope_inventory_fixed",
            "advance_to_8_7_56_1220": inventory_ready,
            "next_required_artifacts": ["charge_normalization_residual_scope_classification_audit"],
        },
        {
            "targets": targets,
            "prior_1217_summary": gate_1217,
            "prior_1218_summary": eval_1218,
            "qball_mapping_summary": qball_mapping["summary"],
            "qball_normalization_summary": qball_normalization["summary"],
            "qball_ground_state_row": ground_state,
            "ai_context_snapshot": ai_context,
        },
    )

    audit = payload(
        "8.7.56.1220",
        "Trial-2 numeric alpha charge-normalization residual scope classification audit",
        inputs,
        [
            row("qball_ground_state_charge_matches_required_factor", "pass" if qball_ground_state_charge_matches_required_factor else "reject", "Q-ball ground-state charge matches required factor", 1 if qball_ground_state_charge_matches_required_factor else 0, "The new note only closes the gap if the ground-state Noether charge is close to 0.30282."),
            row("qball_ground_state_charge_matches_action_level", "pass" if qball_ground_state_charge_matches_action_level else "reject", "Q-ball ground-state charge matches current action-level candidate", 1 if qball_ground_state_charge_matches_action_level else 0, "The retained Q-ball ground-state charge is effectively unity under the old direct identity."),
            row("qball_normalization_freedom_available", "pass" if qball_normalization_freedom_available else "reject", "Q-ball normalization freedom available", 1 if qball_normalization_freedom_available else 0, "The old public audit already froze that no multiplicative normalization freedom remains."),
            row("qball_note_supports_independent_qball_lane", "pass" if qball_note_supports_independent_qball_lane else "reject", "Q-ball note supports an independent residual lane", 1 if qball_note_supports_independent_qball_lane else 0, "Without normalization freedom, the Q-ball note is evidence inside adopted-U(1) external import rather than a separate independent lane."),
            row("primary_lane_adopted_u1_external_import", "pass", "primary residual lane is adopted-U1 external import", 1.0, "The old Q-ball/U(1) identity plus no-freedom audit push the residual into the adopted external charge unit, not into a new internal theorem."),
            row("secondary_lane_future_canon_bridge", "pass", "secondary residual lane is future-canon bridge", 1.0, "A future-canon bridge remains the secondary route if one wants an internal derivation rather than adopted import."),
        ],
        {
            "audit_ready": inventory_ready,
            "required_charge_normalization_factor": REQUIRED_E,
            "qball_ground_state_charge_proxy": qball_ground_state_charge_proxy,
            "qball_ground_state_energy_proxy": qball_ground_state_energy_proxy,
            "qball_ground_state_charge_matches_required_factor": qball_ground_state_charge_matches_required_factor,
            "qball_ground_state_charge_matches_action_level": qball_ground_state_charge_matches_action_level,
            "qball_normalization_freedom_available": qball_normalization_freedom_available,
            "qball_note_supports_independent_qball_lane": qball_note_supports_independent_qball_lane,
            "primary_residual_lane": primary_residual_lane,
            "secondary_residual_lane": secondary_residual_lane,
            "reserve_residual_lane": reserve_residual_lane,
            "result_class": "adopted_u1_external_import_primary_future_canon_secondary_qball_reserve",
        },
        {
            "overall_status": "trial2_numeric_alpha_residual_scope_classified",
            "advance_to_8_7_56_1221": inventory_ready,
            "next_required_artifacts": ["charge_normalization_residual_scope_classification_declaration_gate"],
        },
        {
            "inventory_summary": inventory["summary"],
            "qball_discrete_summary": qball_discrete["summary"],
            "qball_ground_state_row": ground_state,
        },
    )

    declaration_gate = payload(
        "8.7.56.1221",
        "Trial-2 numeric alpha charge-normalization residual scope classification declaration gate",
        inputs,
        [
            row("adopted_u1_external_import_primary_lane_ready", "pass", "adopted-U1 external-import primary lane ready", 1.0, "The residual is now classified with a primary lane."),
            row("future_canon_bridge_secondary_lane_ready", "pass", "future-canon bridge secondary lane ready", 1.0, "Future canon remains the internal-derivation backup lane."),
            row("qball_reserve_lane_retained", "pass", "Q-ball reserve lane retained", 1.0, "The new note is retained as computation-side evidence, but not as an independent primary lane."),
            row("numeric_closeout_ready", "reject", "numeric closeout ready", 0.0, "The Q-ball Noether-charge readout does not close the 0.30282 gap under the retained direct identity."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "The route remains structurally alive and is not a physical reject."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "charge_normalization_residual_scope_classified",
            "primary_residual_lane": primary_residual_lane,
            "secondary_residual_lane": secondary_residual_lane,
            "reserve_residual_lane": reserve_residual_lane,
            "current_canon_internal_derivation_complete": False,
            "numeric_closeout_ready": False,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_residual_lane_gate_frozen",
            "advance_to_8_7_56_1222": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"]},
    )

    evaluation = payload(
        "8.7.56.1222",
        "Trial-2 numeric alpha charge-normalization residual scope classification numeric evaluation",
        inputs,
        [
            row("qball_ground_state_charge_proxy", "inventory", "Q-ball ground-state charge proxy", qball_ground_state_charge_proxy, "The retained Q-ball discrete inversion already fixes the mode_index=1 charge near unity."),
            row("qball_alpha_candidate", "inventory", "Q-ball alpha candidate", qball_alpha_candidate, "The Noether-charge readout implies alpha = q_1^2 / (4*pi) under the retained direct identity."),
            row("qball_charge_ratio_to_required", "inventory", "Q-ball charge ratio to required factor", qball_charge_ratio_to_required, "Values above one mean the Q-ball charge stays too large to explain the 0.30282 factor."),
            row("qball_alpha_ratio_to_target", "inventory", "Q-ball alpha ratio to target", qball_alpha_ratio_to_target, "This remains near the current 1/(4*pi) mismatch rather than closing to 1/137."),
            row("numeric_state_changed_by_current_branch", "reject", "numeric state changed by current branch", 0.0, "The new note changes scope classification, not the numeric state itself."),
            row("ground_state_charge_action_level_delta", "inventory", "ground-state charge delta from unity", qball_action_level_charge_delta, "The old Q-ball readout is very close to the current action-level candidate e = 1."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "adopted_u1_external_import_primary_future_canon_secondary_qball_reserve",
            "qball_ground_state_charge_proxy": qball_ground_state_charge_proxy,
            "qball_ground_state_energy_proxy": qball_ground_state_energy_proxy,
            "qball_alpha_candidate": qball_alpha_candidate,
            "required_charge_normalization_factor": REQUIRED_E,
            "qball_charge_ratio_to_required": qball_charge_ratio_to_required,
            "qball_charge_relative_error": qball_charge_relative_error,
            "qball_alpha_ratio_to_target": qball_alpha_ratio_to_target,
            "numeric_state_changed_by_current_branch": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_residual_scope_classification_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"declaration_gate_summary": declaration_gate["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1219-.1222 artifacts generated")
    print(f"[qball_ground_state_charge_proxy] {qball_ground_state_charge_proxy:.16f}")
    print(f"[qball_alpha_candidate] {qball_alpha_candidate:.16f}")
    print(f"[qball_charge_ratio_to_required] {qball_charge_ratio_to_required:.16f}")


if __name__ == "__main__":
    main()
