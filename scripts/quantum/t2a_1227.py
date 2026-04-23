#!/usr/bin/env python3
"""Generate 8.7.56.1227-.1230 Trial-2 vacuum-polarization charge/spin ledger artifacts."""

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

NOTE_VACUUM = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_vacuum_polarization_analog.md")
QBALL_MAPPING = PUBLIC_OUT / "mass_origin_qball_charge_mapping_statement_freeze_metrics.json"
QBALL_NORMALIZATION = PUBLIC_OUT / "mass_origin_qball_charge_operator_normalization_audit_metrics.json"

INVENTORY_1223 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_analog_source_inventory_metrics.json"
)
AUDIT_1224 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_analog_audit_metrics.json"
)
GATE_1225 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_analog_declaration_gate_metrics.json"
)
EVAL_1226 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_analog_numeric_evaluation_metrics.json"
)

ALPHA_TARGET = 7.2973525692838015e-3
ALPHA_BARE = 1.0 / (4.0 * math.pi)
INV_ALPHA_BARE = 4.0 * math.pi
NEXT_ROUTE = "8.7.56.1231"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_adopted_u1_vacuum_polarization_unresolved_coefficient_review"

MUON_RATIO = 206.0
TAU_RATIO = 3478.0
PROTON_RATIO = 1836.0


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


# Function: compute the minimal fixed-ledger alpha candidates.

def compute_minimal_ledger() -> dict[str, float]:
    """Compute the minimal fixed-ledger and proton-augmented screening passes."""
    l_mu = math.log(MUON_RATIO * MUON_RATIO)
    l_tau = math.log(TAU_RATIO * TAU_RATIO)
    l_p = math.log(PROTON_RATIO * PROTON_RATIO)

    minimal_inv_alpha = INV_ALPHA_BARE + (l_mu + l_tau) / (3.0 * math.pi)
    minimal_alpha = 1.0 / minimal_inv_alpha
    proton_augmented_inv_alpha = minimal_inv_alpha + l_p / (3.0 * math.pi)
    proton_augmented_alpha = 1.0 / proton_augmented_inv_alpha

    return {
        "minimal_fixed_ledger_inv_alpha": minimal_inv_alpha,
        "minimal_fixed_ledger_alpha": minimal_alpha,
        "minimal_fixed_ledger_ratio_to_target": minimal_alpha / ALPHA_TARGET,
        "proton_augmented_inv_alpha": proton_augmented_inv_alpha,
        "proton_augmented_alpha": proton_augmented_alpha,
        "proton_augmented_ratio_to_target": proton_augmented_alpha / ALPHA_TARGET,
    }


# Function: execute the charge/spin ledger review branch.

def main() -> None:
    """Execute the 8.7.56.1227-.1230 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART3A,
        PART5,
        NOTE_VACUUM,
        QBALL_MAPPING,
        QBALL_NORMALIZATION,
        INVENTORY_1223,
        AUDIT_1224,
        GATE_1225,
        EVAL_1226,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    vacuum_note_text = read_text(NOTE_VACUUM)

    qball_mapping = read_json(QBALL_MAPPING)
    qball_normalization = read_json(QBALL_NORMALIZATION)
    inventory_1223 = read_json(INVENTORY_1223)["summary"]
    audit_1224 = read_json(AUDIT_1224)["summary"]
    gate_1225 = read_json(GATE_1225)["summary"]
    eval_1226 = read_json(EVAL_1226)["summary"]
    ledger = compute_minimal_ledger()

    inventory_ready = all(
        (
            inventory_1223["inventory_ready"],
            audit_1224["vacuum_polarization_analog_admissible_as_external_import"],
            gate_1225["vacuum_polarization_analog_external_import_admissible"],
            qball_normalization["summary"]["direct_qball_u1_identity_required"],
            hit(vacuum_note_text, "| electron | 1 | 0 | 0 |") is not None,
            hit(vacuum_note_text, "| electron, muon, tau | 1/2 | $+4/3$") is not None,
            hit(vacuum_note_text, "| W | 1 | $-7$") is not None,
            hit(vacuum_note_text, "| radial mode | 0 | $+1/3$") is not None,
        )
    )

    electron_shell_rule = "exclude_zero_log_at_on_shell_reference"
    charged_fermion_rule = "include_muon_tau"
    composite_proton_rule = "reserve_effective_hadronic_candidate"
    neutral_state_rule = "exclude_observed_neutral_states_under_external_import"
    neutral_rule_requires_external_judgment = True
    w_vector_rule = "reserve_vector_coefficient_unfixed"
    radial_scalar_rule = "reserve_scalar_charge_coupling_unfixed"

    minimum_fixed_ledger_ready = True
    full_exact_ledger_ready = False
    mainline_value_retained = True
    minimal_ledger_improves_over_bare = ledger["minimal_fixed_ledger_alpha"] < ALPHA_BARE
    minimal_ledger_close_to_target = False
    proton_augmented_close_to_target = False
    lower_bound_pass_supports_immediate_breakthrough = False
    unresolved_items_dominate_remaining_gap = True
    audit_ready = inventory_ready

    targets = [
        target(status_text, STATUS, "status_current_branch", "vacuum-polarization analog charge/spin ledger review", "STATUS must expose the charge/spin ledger review as the current branch."),
        target(roadmap_text, ROADMAP, "roadmap_1227", "`8.7.56.1227-.1230`", "ROADMAP must expose the current 1227 branch."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "recent_1223", "`8.7.56.1223-.1226`", "Recent history must preserve the predecessor admissibility review."),
        target(vacuum_note_text, NOTE_VACUUM, "note_on_shell", "q^2 \\to m_e^2", "The note must preserve the on-shell electron reference point."),
        target(vacuum_note_text, NOTE_VACUUM, "note_electron_row", "| electron | 1 | 0 | 0 |", "The note must preserve the zero-log electron row."),
        target(vacuum_note_text, NOTE_VACUUM, "note_spin_fermion", "| electron, muon, tau | 1/2 | $+4/3$", "The note must preserve the charged-fermion spin rule."),
        target(vacuum_note_text, NOTE_VACUUM, "note_spin_vector", "| W | 1 | $-7$", "The note must preserve the charged-vector placeholder rule."),
        target(vacuum_note_text, NOTE_VACUUM, "note_spin_scalar", "| radial mode | 0 | $+1/3$", "The note must preserve the radial-scalar placeholder rule."),
        target(part3a_text, PART3A, "part3a_adopted_u1", "\"Local Maxwell/QED is kept unchanged\"", "Part III-A must preserve the adopted-U(1) stance."),
        target(part5_text, PART5, "part5_current_step", "current official next step is Trial-2 numeric $\\alpha$ adopted-U(1) vacuum-polarization analog charge/spin ledger review branch `8.7.56.1227-.1230`", "Part V must expose the current 1227 branch before this branch completes."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "vacuum_note": display_path(NOTE_VACUUM),
            "qball_mapping": display_path(QBALL_MAPPING),
            "qball_normalization": display_path(QBALL_NORMALIZATION),
            "inventory_1223": display_path(INVENTORY_1223),
            "audit_1224": display_path(AUDIT_1224),
            "gate_1225": display_path(GATE_1225),
            "eval_1226": display_path(EVAL_1226),
        },
        "predecessor_summary": {
            "vacuum_polarization_analog_external_import_admissible": gate_1225[
                "vacuum_polarization_analog_external_import_admissible"
            ],
            "closest_rough_candidate_alpha": eval_1226["closest_rough_candidate_alpha"],
            "closest_rough_candidate_ratio_to_target": eval_1226["closest_rough_candidate_ratio_to_target"],
        },
        "ledger_rules": {
            "electron_shell_rule": electron_shell_rule,
            "charged_fermion_rule": charged_fermion_rule,
            "composite_proton_rule": composite_proton_rule,
            "neutral_state_rule": neutral_state_rule,
            "neutral_rule_requires_external_judgment": neutral_rule_requires_external_judgment,
            "w_vector_rule": w_vector_rule,
            "radial_scalar_rule": radial_scalar_rule,
        },
    }

    rows_inventory = [
        row("inventory_ready", "pass" if inventory_ready else "fail", "charge/spin ledger inventory ready", 1.0 if inventory_ready else 0.0, "The predecessor admissibility review, Q-ball charge identity pack, and the note's counting tables are assembled into one ledger pack."),
        row("qball_direct_identity_required", "pass" if qball_normalization["summary"]["direct_qball_u1_identity_required"] else "fail", "direct Q-ball/U(1) identity required", 1.0 if qball_normalization["summary"]["direct_qball_u1_identity_required"] else 0.0, "The retained public pack leaves no independent multiplicative charge normalization freedom."),
        row("note_spin_table_present", "pass" if hit(vacuum_note_text, "spin factor の影響") is not None else "fail", "vacuum-polarization spin table present", 1.0 if hit(vacuum_note_text, "spin factor の影響") is not None else 0.0, "The note exposes the fermion/vector/scalar loop placeholders needed for this ledger review."),
    ]

    summary_inventory = {
        "inventory_ready": inventory_ready,
        "qball_direct_identity_required": qball_normalization["summary"]["direct_qball_u1_identity_required"],
        "charge_quantum_normalization": qball_mapping["summary"]["charge_quantum_normalization"],
        "electron_shell_rule": electron_shell_rule,
        "charged_fermion_rule": charged_fermion_rule,
    }

    decision_inventory = {
        "source_inventory_step_completed": inventory_ready,
        "candidate_mechanism_class": "vacuum_polarization_charge_spin_ledger_under_adopted_u1",
    }

    evidence_inventory = {
        "targets": targets,
        "canonical_statement": qball_mapping["formulas"]["canonical_statement"],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_source_inventory",
        payload(
            "8.7.56.1227",
            "Trial-2 adopted-U(1) vacuum-polarization charge/spin ledger source inventory",
            inputs,
            rows_inventory,
            summary_inventory,
            decision_inventory,
            evidence_inventory,
        ),
    )

    rows_audit = [
        row("audit_ready", "pass" if audit_ready else "fail", "charge/spin ledger audit ready", 1.0 if audit_ready else 0.0, "The note's state and spin placeholders can now be audited against the retained U(1) identity pack."),
        row("electron_shell_zero_log_fixed", "pass", "electron shell zero-log rule fixed", 1.0, "At q^2 = m_e^2 the electron shell is fixed to zero contribution by the note itself."),
        row("charged_fermion_rule_fixed", "pass", "charged-fermion rule fixed", 1.0, "Muon and tau can be carried as the minimal charged-fermion ledger under the note's table."),
        row("composite_proton_rule_fixed", "pass", "composite proton rule fixed as reserve", 1.0, "The proton is retained only as an effective hadronic add-on candidate rather than part of the minimal exact ledger."),
        row("neutral_state_rule_requires_external_judgment", "pass" if neutral_rule_requires_external_judgment else "fail", "neutral-state rule requires external observed-charge judgment", 1.0 if neutral_rule_requires_external_judgment else 0.0, "Neutron and Z cannot be kept in the minimal observed-U(1) ledger without conflicting with their observed-neutral labels."),
        row("w_vector_rule_unfixed", "pass", "W vector coefficient remains unfixed", 1.0, "The note supplies a rough sign placeholder, but the current pack does not yet fix the exact charged-vector coefficient for this import route."),
        row("radial_scalar_rule_unfixed", "pass", "radial scalar coefficient remains unfixed", 1.0, "The radial-mode placeholder lacks a fixed adopted-U(1) charge/coupling coefficient in the current pack."),
        row("minimum_fixed_ledger_ready", "pass" if minimum_fixed_ledger_ready else "fail", "minimum fixed ledger ready", 1.0 if minimum_fixed_ledger_ready else 0.0, "A lower-bound pass can proceed with electron zero-log plus muon/tau charged-fermion screening only."),
        row("full_exact_ledger_ready", "pass" if full_exact_ledger_ready else "fail", "full exact ledger ready", 1.0 if full_exact_ledger_ready else 0.0, "Composite, neutral, vector, and scalar items are not fixed enough for a full exact screening run."),
    ]

    summary_audit = {
        "audit_ready": audit_ready,
        "electron_shell_rule": electron_shell_rule,
        "charged_fermion_rule": charged_fermion_rule,
        "composite_proton_rule": composite_proton_rule,
        "neutral_state_rule": neutral_state_rule,
        "neutral_rule_requires_external_judgment": neutral_rule_requires_external_judgment,
        "w_vector_rule": w_vector_rule,
        "radial_scalar_rule": radial_scalar_rule,
        "minimum_fixed_ledger_ready": minimum_fixed_ledger_ready,
        "full_exact_ledger_ready": full_exact_ledger_ready,
        "mainline_value_retained": mainline_value_retained,
    }

    decision_audit = {
        "ledger_classification": "minimal_lower_bound_ledger_fixed_full_ledger_unresolved",
        "exact_screening_ready_now": full_exact_ledger_ready,
        "lower_bound_pass_ready": minimum_fixed_ledger_ready,
    }

    evidence_audit = {
        "qball_normalization_summary": qball_normalization["summary"],
        "rough_candidate_reference": {
            "closest_rough_candidate_alpha": eval_1226["closest_rough_candidate_alpha"],
            "closest_rough_candidate_ratio_to_target": eval_1226["closest_rough_candidate_ratio_to_target"],
        },
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_audit",
        payload(
            "8.7.56.1228",
            "Trial-2 adopted-U(1) vacuum-polarization charge/spin ledger audit",
            inputs,
            rows_audit,
            summary_audit,
            decision_audit,
            evidence_audit,
        ),
    )

    rows_gate = [
        row("problem_classification_fixed", "pass", "trial2 numeric alpha ledger classification fixed", 1.0, "The ledger branch fixes what is included, excluded, or reserved without overclaiming a solved screening formula."),
        row("minimum_lower_bound_pass_ready", "pass" if minimum_fixed_ledger_ready else "fail", "minimum lower-bound pass ready", 1.0 if minimum_fixed_ledger_ready else 0.0, "The branch can proceed to a lower-bound evaluation with the fixed minimal ledger."),
        row("full_exact_run_not_ready", "pass" if not full_exact_ledger_ready else "fail", "full exact run not ready", 1.0 if not full_exact_ledger_ready else 0.0, "Vector, scalar, and composite coefficient questions still block a full exact screening run."),
    ]

    summary_gate = {
        "trial2_numeric_alpha_problem_classification": "adopted_u1_vacuum_polarization_charge_spin_ledger_fixed_lower_bound_ready",
        "minimum_fixed_ledger_ready": minimum_fixed_ledger_ready,
        "full_exact_ledger_ready": full_exact_ledger_ready,
        "vector_scalar_composite_review_required": True,
        "numeric_closeout_ready": False,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }

    decision_gate = {
        "next_route_name": NEXT_ROUTE_NAME,
        "next_route_step": NEXT_ROUTE,
        "continue_mainline_value": mainline_value_retained,
    }

    evidence_gate = {
        "minimal_fixed_ledger_alpha": ledger["minimal_fixed_ledger_alpha"],
        "minimal_fixed_ledger_ratio_to_target": ledger["minimal_fixed_ledger_ratio_to_target"],
        "proton_augmented_alpha": ledger["proton_augmented_alpha"],
        "proton_augmented_ratio_to_target": ledger["proton_augmented_ratio_to_target"],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_declaration_gate",
        payload(
            "8.7.56.1229",
            "Trial-2 adopted-U(1) vacuum-polarization charge/spin ledger declaration gate",
            inputs,
            rows_gate,
            summary_gate,
            decision_gate,
            evidence_gate,
        ),
    )

    rows_eval = [
        row("alpha_bare", "pass", "bare alpha candidate", ALPHA_BARE, "The action-level bare coupling remains 1/(4π)."),
        row("minimal_fixed_ledger_alpha", "pass", "minimal fixed-ledger alpha candidate", ledger["minimal_fixed_ledger_alpha"], "This lower-bound pass uses electron-shell zero plus muon/tau charged-fermion screening only."),
        row("minimal_fixed_ledger_ratio_to_target", "pass", "minimal fixed-ledger ratio to target", ledger["minimal_fixed_ledger_ratio_to_target"], "The lower-bound pass remains far above the target alpha."),
        row("proton_augmented_alpha", "pass", "proton-augmented alpha candidate", ledger["proton_augmented_alpha"], "Adding the proton as an effective composite candidate improves the value slightly but still remains far above target."),
        row("proton_augmented_ratio_to_target", "pass", "proton-augmented ratio to target", ledger["proton_augmented_ratio_to_target"], "Even the proton-augmented ledger is not close to 1/137."),
    ]

    summary_eval = {
        "alpha_target": ALPHA_TARGET,
        "alpha_bare": ALPHA_BARE,
        "minimal_fixed_ledger_alpha": ledger["minimal_fixed_ledger_alpha"],
        "minimal_fixed_ledger_ratio_to_target": ledger["minimal_fixed_ledger_ratio_to_target"],
        "proton_augmented_alpha": ledger["proton_augmented_alpha"],
        "proton_augmented_ratio_to_target": ledger["proton_augmented_ratio_to_target"],
        "minimal_ledger_improves_over_bare": minimal_ledger_improves_over_bare,
        "minimal_ledger_close_to_target": minimal_ledger_close_to_target,
        "proton_augmented_close_to_target": proton_augmented_close_to_target,
        "lower_bound_pass_supports_immediate_breakthrough": lower_bound_pass_supports_immediate_breakthrough,
        "unresolved_items_dominate_remaining_gap": unresolved_items_dominate_remaining_gap,
    }

    decision_eval = {
        "numeric_state_class": "minimal_lower_bound_not_close_unresolved_items_dominate",
        "continue_to_unresolved_coefficient_review": True,
        "demote_to_reserve_now": False,
    }

    evidence_eval = {
        "minimal_fixed_ledger_states": [
            "electron_shell_zero_log",
            "muon_charged_fermion",
            "tau_charged_fermion",
        ],
        "reserved_items": [
            "proton_effective_hadronic_candidate",
            "neutron_and_Z_observed_neutral_judgment",
            "W_vector_coefficient",
            "radial_scalar_coefficient",
        ],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_numeric_evaluation",
        payload(
            "8.7.56.1230",
            "Trial-2 adopted-U(1) vacuum-polarization charge/spin ledger numeric evaluation",
            inputs,
            rows_eval,
            summary_eval,
            decision_eval,
            evidence_eval,
        ),
    )


if __name__ == "__main__":
    main()
