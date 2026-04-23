#!/usr/bin/env python3
"""Generate 8.7.56.1223-.1226 Trial-2 adopted-U(1) vacuum-polarization analog artifacts."""

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

INVENTORY_1219 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_source_inventory_metrics.json"
)
AUDIT_1220 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_audit_metrics.json"
)
GATE_1221 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_declaration_gate_metrics.json"
)
EVAL_1222 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_numeric_evaluation_metrics.json"
)

ALPHA_TARGET = 7.2973525692838015e-3
ALPHA_BARE = 1.0 / (4.0 * math.pi)
INV_ALPHA_BARE = 4.0 * math.pi
NEXT_ROUTE = "8.7.56.1227"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_review"

MASS_RATIOS = {
    "muon": 206.0,
    "proton": 1836.0,
    "neutron": 1839.0,
    "tau": 3478.0,
    "W": 157274.0,
    "Z": 178446.0,
}


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


# Function: evaluate one screening candidate from one list of logarithms.

def alpha_from_logs(log_terms: list[float]) -> tuple[float, float]:
    """Return 1/alpha and alpha for one positive-log screening pass."""
    inv_alpha = INV_ALPHA_BARE + (1.0 / (3.0 * math.pi)) * sum(log_terms)
    return inv_alpha, 1.0 / inv_alpha


# Function: compute the rough vacuum-polarization candidates carried by the note.

def compute_candidates() -> dict[str, float | str]:
    """Compute raw, charge-clean, and spin-rough candidates for the note review."""
    logs = {name: math.log(ratio * ratio) for name, ratio in MASS_RATIOS.items()}

    note_raw_terms = [logs[name] for name in ("muon", "proton", "neutron", "tau", "W", "Z")]
    note_raw_inv_alpha, note_raw_alpha = alpha_from_logs(note_raw_terms)

    charge_clean_terms = [logs[name] for name in ("muon", "proton", "tau", "W")]
    charge_clean_inv_alpha, charge_clean_alpha = alpha_from_logs(charge_clean_terms)

    fermion_piece = (4.0 / 3.0) * (logs["muon"] + logs["proton"] + logs["tau"])
    vector_piece = -7.0 * logs["W"]
    spin_rough_inv_alpha = INV_ALPHA_BARE + (1.0 / (3.0 * math.pi)) * (fermion_piece + vector_piece)
    spin_rough_alpha = 1.0 / spin_rough_inv_alpha

    candidates = {
        "alpha_bare": ALPHA_BARE,
        "note_raw_identified_inv_alpha": note_raw_inv_alpha,
        "note_raw_identified_alpha": note_raw_alpha,
        "note_raw_identified_ratio_to_target": note_raw_alpha / ALPHA_TARGET,
        "charge_clean_identified_inv_alpha": charge_clean_inv_alpha,
        "charge_clean_identified_alpha": charge_clean_alpha,
        "charge_clean_identified_ratio_to_target": charge_clean_alpha / ALPHA_TARGET,
        "spin_rough_charge_clean_inv_alpha": spin_rough_inv_alpha,
        "spin_rough_charge_clean_alpha": spin_rough_alpha,
        "spin_rough_charge_clean_ratio_to_target": spin_rough_alpha / ALPHA_TARGET,
        "rough_pass_spread_factor": max(note_raw_alpha, charge_clean_alpha, spin_rough_alpha)
        / min(note_raw_alpha, charge_clean_alpha, spin_rough_alpha),
    }

    closest_label = "note_raw_identified_pass"
    closest_alpha = note_raw_alpha
    closest_ratio = note_raw_alpha / ALPHA_TARGET

    for label, alpha in (
        ("charge_clean_identified_pass", charge_clean_alpha),
        ("spin_rough_charge_clean_pass", spin_rough_alpha),
    ):
        ratio = alpha / ALPHA_TARGET
        if abs(alpha - ALPHA_TARGET) < abs(closest_alpha - ALPHA_TARGET):
            closest_label = label
            closest_alpha = alpha
            closest_ratio = ratio

    candidates["closest_rough_candidate_label"] = closest_label
    candidates["closest_rough_candidate_alpha"] = closest_alpha
    candidates["closest_rough_candidate_ratio_to_target"] = closest_ratio
    return candidates


# Function: execute the vacuum-polarization analog review branch.

def main() -> None:
    """Execute the 8.7.56.1223-.1226 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART3A,
        PART5,
        NOTE_VACUUM,
        INVENTORY_1219,
        AUDIT_1220,
        GATE_1221,
        EVAL_1222,
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

    inventory_1219 = read_json(INVENTORY_1219)["summary"]
    audit_1220 = read_json(AUDIT_1220)["summary"]
    gate_1221 = read_json(GATE_1221)["summary"]
    eval_1222 = read_json(EVAL_1222)["summary"]
    candidates = compute_candidates()

    adopted_u1_primary_lane_retained = (
        audit_1220["primary_residual_lane"] == "adopted_u1_external_import"
        and gate_1221["selected_next_generation_route"]
        == "trial2_numeric_alpha_adopted_u1_external_import_primary_lane_contract"
    )
    structural_route_present = hit(part3a_text, "e=g_P/\\sqrt{Z_P}") is not None
    adopted_u1_surface_present = hit(part3a_text, "\"Local Maxwell/QED is kept unchanged\"") is not None
    explicit_mapping_absent_present = hit(part3a_text, "charge-normalization bridge absent under current canon") is not None
    note_bare_surface_present = hit(vacuum_note_text, "bare α = 1/(4π) = 0.0796") is not None
    note_formula_present = hit(vacuum_note_text, "\\frac{1}{\\alpha_{\\rm phys}(q^2)}") is not None
    note_spin_table_present = hit(vacuum_note_text, "spin factor の影響") is not None

    inventory_ready = all(
        (
            inventory_1219["inventory_ready"],
            adopted_u1_primary_lane_retained,
            structural_route_present,
            adopted_u1_surface_present,
            explicit_mapping_absent_present,
            note_bare_surface_present,
            note_formula_present,
            note_spin_table_present,
        )
    )

    direct_current_canon_loop_bridge_available = False
    vacuum_polarization_analog_admissible_as_external_import = True
    breakthrough_confirmed = False
    note_raw_improves_over_bare = candidates["note_raw_identified_alpha"] < ALPHA_BARE
    charge_clean_improves_over_bare = candidates["charge_clean_identified_alpha"] < ALPHA_BARE
    rough_passes_close_target = False
    rough_passes_consistent = candidates["rough_pass_spread_factor"] < 2.0
    computation_branch_has_merit = True
    charge_spin_ledger_required = True
    audit_ready = inventory_ready

    targets = [
        target(status_text, STATUS, "status_residual_scope", "adopted-U(1) external import primary", "STATUS must preserve the adopted-U(1) primary lane before this review runs."),
        target(roadmap_text, ROADMAP, "roadmap_1223", "`8.7.56.1223-.1226`", "ROADMAP must expose the current 1223 branch."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "recent_1219", "`8.7.56.1219-.1222`", "Recent history must preserve the predecessor residual-scope classification."),
        target(part3a_text, PART3A, "part3a_structural_route", "e=g_P/\\sqrt{Z_P}", "Part III-A must preserve the structural charge route."),
        target(part3a_text, PART3A, "part3a_adopted_u1", "\"Local Maxwell/QED is kept unchanged\"", "Part III-A must preserve the adopted-U(1) stance."),
        target(part3a_text, PART3A, "part3a_mapping_absent", "charge-normalization bridge absent under current canon", "Part III-A must preserve the current-canon absence wording."),
        target(vacuum_note_text, NOTE_VACUUM, "note_bare_alpha", "bare α = 1/(4π) = 0.0796", "The note must preserve the bare-coupling premise."),
        target(vacuum_note_text, NOTE_VACUUM, "note_formula", "\\frac{1}{\\alpha_{\\rm phys}(q^2)}", "The note must preserve the one-loop screening formula."),
        target(vacuum_note_text, NOTE_VACUUM, "note_spin", "spin factor の影響", "The note must preserve the spin-factor correction step."),
        target(part5_text, PART5, "part5_current_step", "current official next step is Trial-2 numeric $\\alpha$ adopted-U(1) external-import primary-lane contract branch `8.7.56.1223-.1226`", "Part V must expose the current 1223 branch before this review completes."),
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
            "inventory_1219": display_path(INVENTORY_1219),
            "audit_1220": display_path(AUDIT_1220),
            "gate_1221": display_path(GATE_1221),
            "eval_1222": display_path(EVAL_1222),
        },
        "predecessor_summary": {
            "primary_residual_lane": audit_1220["primary_residual_lane"],
            "secondary_residual_lane": audit_1220["secondary_residual_lane"],
            "reserve_residual_lane": audit_1220["reserve_residual_lane"],
            "required_charge_factor": eval_1222["qball_charge_ratio_to_required"],
        },
        "candidate_ledger": {
            "mass_ratios_to_electron": MASS_RATIOS,
            "alpha_bare": ALPHA_BARE,
            "alpha_target": ALPHA_TARGET,
            "raw_note_candidate": {
                "state_list": ["muon", "proton", "neutron", "tau", "W", "Z"],
                "alpha": candidates["note_raw_identified_alpha"],
                "inv_alpha": candidates["note_raw_identified_inv_alpha"],
            },
            "charge_clean_candidate": {
                "state_list": ["muon", "proton", "tau", "W"],
                "alpha": candidates["charge_clean_identified_alpha"],
                "inv_alpha": candidates["charge_clean_identified_inv_alpha"],
            },
            "spin_rough_candidate": {
                "fermion_states": ["muon", "proton", "tau"],
                "vector_state": "W",
                "alpha": candidates["spin_rough_charge_clean_alpha"],
                "inv_alpha": candidates["spin_rough_charge_clean_inv_alpha"],
            },
        },
    }

    rows_inventory = [
        row("inventory_ready", "pass" if inventory_ready else "fail", "vacuum-polarization analog inventory ready", 1.0 if inventory_ready else 0.0, "The adopted-U(1) primary lane, the predecessor residual classification, and the new note are collected into one computation pack."),
        row("adopted_u1_primary_lane_retained", "pass" if adopted_u1_primary_lane_retained else "fail", "adopted-U(1) primary lane retained", 1.0 if adopted_u1_primary_lane_retained else 0.0, "The predecessor branch still fixes adopted-U(1) external import as the primary lane."),
        row("note_formula_present", "pass" if note_formula_present else "fail", "vacuum-polarization formula present in note", 1.0 if note_formula_present else 0.0, "The note surfaces a standard one-loop screening formula that can be tested computation-first."),
    ]

    summary_inventory = {
        "inventory_ready": inventory_ready,
        "adopted_u1_primary_lane_retained": adopted_u1_primary_lane_retained,
        "structural_route_present": structural_route_present,
        "adopted_u1_surface_present": adopted_u1_surface_present,
        "explicit_mapping_absent_present": explicit_mapping_absent_present,
        "vacuum_polarization_note_present": note_bare_surface_present and note_formula_present,
    }

    decision_inventory = {
        "source_inventory_step_completed": inventory_ready,
        "candidate_mechanism_class": "adopted_u1_external_import_vacuum_polarization_analog_candidate",
    }

    evidence_inventory = {
        "targets": targets,
        "current_step": ai_context["current_step"],
        "vacuum_note_mechanisms": [
            "radial_mode_loop",
            "qball_spectrum_charged_loop",
            "finite_size_charge_distribution",
        ],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_analog_source_inventory",
        payload(
            "8.7.56.1223",
            "Trial-2 adopted-U(1) vacuum-polarization analog source inventory",
            inputs,
            rows_inventory,
            summary_inventory,
            decision_inventory,
            evidence_inventory,
        ),
    )

    rows_audit = [
        row("audit_ready", "pass" if audit_ready else "fail", "vacuum-polarization analog audit ready", 1.0 if audit_ready else 0.0, "The predecessor residual-lane classification and the new computation note can be audited together."),
        row("external_import_admissible", "pass" if vacuum_polarization_analog_admissible_as_external_import else "fail", "vacuum-polarization analog admissible as external import", 1.0 if vacuum_polarization_analog_admissible_as_external_import else 0.0, "The note concretizes the already-primary adopted-U(1) external-import lane."),
        row("current_canon_loop_bridge_available", "pass" if direct_current_canon_loop_bridge_available else "fail", "direct current-canon loop bridge available", 1.0 if direct_current_canon_loop_bridge_available else 0.0, "Current canon still does not explicitly license the standard one-loop screening formula as an internal derivation."),
        row("note_raw_improves_over_bare", "pass" if note_raw_improves_over_bare else "fail", "note raw identified pass improves over bare alpha", candidates["note_raw_identified_alpha"], "The note's own identified-particle pass reduces alpha from 1/(4π), but remains far above target."),
        row("charge_clean_improves_over_bare", "pass" if charge_clean_improves_over_bare else "fail", "charge-clean identified pass improves over bare alpha", candidates["charge_clean_identified_alpha"], "Excluding neutral states still leaves alpha above target, although it remains smaller than bare alpha."),
        row("rough_passes_close_target", "pass" if rough_passes_close_target else "fail", "rough passes close the target alpha", candidates["closest_rough_candidate_ratio_to_target"], "Even the closest rough pass remains multiple times larger than the target alpha."),
        row("rough_passes_consistent", "pass" if rough_passes_consistent else "fail", "rough passes are numerically consistent", candidates["rough_pass_spread_factor"], "Charge-clean and spin-rough choices spread the candidate widely, so the branch needs a stricter charge/spin ledger before any exact claim."),
        row("breakthrough_confirmed", "pass" if breakthrough_confirmed else "fail", "vacuum-polarization analog breakthrough confirmed", 1.0 if breakthrough_confirmed else 0.0, "This branch does not confirm a breakthrough."),
    ]

    summary_audit = {
        "audit_ready": audit_ready,
        "vacuum_polarization_analog_admissible_as_external_import": vacuum_polarization_analog_admissible_as_external_import,
        "direct_current_canon_loop_bridge_available": direct_current_canon_loop_bridge_available,
        "note_raw_improves_over_bare": note_raw_improves_over_bare,
        "charge_clean_improves_over_bare": charge_clean_improves_over_bare,
        "rough_passes_close_target": rough_passes_close_target,
        "rough_passes_consistent": rough_passes_consistent,
        "computation_branch_has_merit": computation_branch_has_merit,
        "charge_spin_ledger_required": charge_spin_ledger_required,
        "breakthrough_confirmed": breakthrough_confirmed,
    }

    decision_audit = {
        "primary_lane_status": "admissible_external_import_computation_candidate",
        "candidate_ready_for_exact_computation": charge_spin_ledger_required,
        "breakthrough_state": "not_confirmed",
    }

    evidence_audit = {
        "candidate_values": candidates,
        "predecessor_gate": {
            "current_canon_internal_derivation_complete": gate_1221["current_canon_internal_derivation_complete"],
            "numeric_closeout_ready": gate_1221["numeric_closeout_ready"],
            "physical_reject_required": gate_1221["physical_reject_required"],
        },
        "neutral_state_warning": [
            "neutron",
            "Z",
        ],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_analog_audit",
        payload(
            "8.7.56.1224",
            "Trial-2 adopted-U(1) vacuum-polarization analog audit",
            inputs,
            rows_audit,
            summary_audit,
            decision_audit,
            evidence_audit,
        ),
    )

    rows_gate = [
        row("problem_classification_fixed", "pass", "trial2 numeric alpha problem classification fixed", 1.0, "The branch fixes the new classification without overclaiming current-canon internal derivation."),
        row("external_import_candidate_admissible", "pass" if vacuum_polarization_analog_admissible_as_external_import else "fail", "vacuum-polarization analog external-import candidate admissible", 1.0 if vacuum_polarization_analog_admissible_as_external_import else 0.0, "The note is retained as an admissible computation-side version of the already-primary adopted-U(1) lane."),
        row("breakthrough_not_confirmed", "pass" if not breakthrough_confirmed else "fail", "breakthrough not confirmed", 1.0 if not breakthrough_confirmed else 0.0, "The rough passes do not justify a solved alpha derivation."),
    ]

    summary_gate = {
        "trial2_numeric_alpha_problem_classification": "adopted_u1_external_import_vacuum_polarization_analog_reviewed",
        "vacuum_polarization_analog_external_import_admissible": vacuum_polarization_analog_admissible_as_external_import,
        "current_canon_internal_derivation_complete": False,
        "numeric_closeout_ready": False,
        "physical_reject_required": False,
        "rough_screening_computation_supports_immediate_breakthrough": breakthrough_confirmed,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }

    decision_gate = {
        "next_route_name": NEXT_ROUTE_NAME,
        "next_route_step": NEXT_ROUTE,
        "carry_order": {
            "primary": "adopted_u1_external_import_vacuum_polarization_analog_candidate",
            "secondary": "future_canon_bridge",
            "reserve": "qball_noether_charge_candidate",
        },
    }

    evidence_gate = {
        "closest_rough_candidate_label": candidates["closest_rough_candidate_label"],
        "closest_rough_candidate_alpha": candidates["closest_rough_candidate_alpha"],
        "closest_rough_candidate_ratio_to_target": candidates["closest_rough_candidate_ratio_to_target"],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_analog_declaration_gate",
        payload(
            "8.7.56.1225",
            "Trial-2 adopted-U(1) vacuum-polarization analog declaration gate",
            inputs,
            rows_gate,
            summary_gate,
            decision_gate,
            evidence_gate,
        ),
    )

    rows_eval = [
        row("alpha_bare", "pass", "bare alpha candidate", ALPHA_BARE, "The action-level bare coupling remains 1/(4π)."),
        row("note_raw_identified_alpha", "pass", "note raw identified-pass alpha", candidates["note_raw_identified_alpha"], "This is the note's direct identified-particle screening pass."),
        row("charge_clean_identified_alpha", "pass", "charge-clean identified-pass alpha", candidates["charge_clean_identified_alpha"], "This excludes clearly neutral states from the first-pass ledger."),
        row("spin_rough_charge_clean_alpha", "pass", "spin-rough charge-clean alpha", candidates["spin_rough_charge_clean_alpha"], "This adds the note's spin-sign structure at rough level and shows strong instability."),
        row("closest_rough_candidate_ratio_to_target", "pass", "closest rough candidate ratio to target", candidates["closest_rough_candidate_ratio_to_target"], "Even the closest rough pass remains well above the target alpha."),
    ]

    summary_eval = {
        "alpha_target": ALPHA_TARGET,
        "alpha_bare": ALPHA_BARE,
        "note_raw_identified_alpha_candidate": candidates["note_raw_identified_alpha"],
        "note_raw_identified_ratio_to_target": candidates["note_raw_identified_ratio_to_target"],
        "charge_clean_identified_alpha_candidate": candidates["charge_clean_identified_alpha"],
        "charge_clean_identified_ratio_to_target": candidates["charge_clean_identified_ratio_to_target"],
        "spin_rough_charge_clean_alpha_candidate": candidates["spin_rough_charge_clean_alpha"],
        "spin_rough_charge_clean_ratio_to_target": candidates["spin_rough_charge_clean_ratio_to_target"],
        "closest_rough_candidate_label": candidates["closest_rough_candidate_label"],
        "closest_rough_candidate_alpha": candidates["closest_rough_candidate_alpha"],
        "closest_rough_candidate_ratio_to_target": candidates["closest_rough_candidate_ratio_to_target"],
        "rough_pass_spread_factor": candidates["rough_pass_spread_factor"],
        "current_branch_confirms_breakthrough": breakthrough_confirmed,
        "current_canon_numeric_state_changed": False,
        "external_import_candidate_numeric_state_available": True,
    }

    decision_eval = {
        "best_rough_pass_is_not_close_enough": True,
        "exact_charge_spin_ledger_required": True,
        "numeric_state_class": "admissible_external_import_candidate_not_yet_close",
    }

    evidence_eval = {
        "ratio_reference": {
            "note_raw_ratio_to_target": candidates["note_raw_identified_ratio_to_target"],
            "charge_clean_ratio_to_target": candidates["charge_clean_identified_ratio_to_target"],
            "spin_rough_ratio_to_target": candidates["spin_rough_charge_clean_ratio_to_target"],
        },
        "predecessor_alpha_candidates": {
            "action_level_alpha": ALPHA_BARE,
            "qball_ground_state_alpha": eval_1222["qball_alpha_candidate"],
        },
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_analog_numeric_evaluation",
        payload(
            "8.7.56.1226",
            "Trial-2 adopted-U(1) vacuum-polarization analog numeric evaluation",
            inputs,
            rows_eval,
            summary_eval,
            decision_eval,
            evidence_eval,
        ),
    )


if __name__ == "__main__":
    main()
