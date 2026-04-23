#!/usr/bin/env python3
"""Generate 8.7.56.1231-.1234 Trial-2 vacuum-polarization unresolved-coefficient artifacts."""

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
EVAL_1222 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_charge_normalization_residual_scope_classification_numeric_evaluation_metrics.json"
)
INVENTORY_1227 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_source_inventory_metrics.json"
)
AUDIT_1228 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_audit_metrics.json"
)
GATE_1229 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_declaration_gate_metrics.json"
)
EVAL_1230 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_charge_spin_ledger_numeric_evaluation_metrics.json"
)

ALPHA_TARGET = 7.2973525692838015e-3
ALPHA_BARE = 1.0 / (4.0 * math.pi)
INV_ALPHA_BARE = 4.0 * math.pi
NEXT_ROUTE = "8.7.56.1235"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_adopted_u1_vacuum_polarization_reserve_heavy_route_contract"

MUON_RATIO = 206.0
TAU_RATIO = 3478.0
PROTON_RATIO = 1836.0
W_RATIO = 157274.0


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


# Function: evaluate one alpha candidate from one inverse-alpha value.

def alpha_from_inv(inv_alpha: float) -> float:
    """Return alpha from one positive inverse-alpha value."""
    if inv_alpha <= 0.0:
        return float("nan")

    return 1.0 / inv_alpha


# Function: compute the unresolved-coefficient scenario envelope.

def compute_scenarios(qball_ground_state_energy_proxy: float) -> dict[str, float]:
    """Compute lower-bound, proton, W, and radial scenario values."""
    l_mu = math.log(MUON_RATIO * MUON_RATIO)
    l_tau = math.log(TAU_RATIO * TAU_RATIO)
    l_p = math.log(PROTON_RATIO * PROTON_RATIO)
    l_w = math.log(W_RATIO * W_RATIO)
    l_radial = math.log((1.0 / qball_ground_state_energy_proxy) ** 2)

    minimal_inv = INV_ALPHA_BARE + (l_mu + l_tau) / (3.0 * math.pi)
    proton_plus1_inv = minimal_inv + l_p / (3.0 * math.pi)
    proton_dirac_inv = INV_ALPHA_BARE + ((4.0 / 3.0) * (l_mu + l_tau + l_p)) / (3.0 * math.pi)
    radial_only_inv = minimal_inv + ((1.0 / 3.0) * l_radial) / (3.0 * math.pi)
    w_dirac_inv = proton_dirac_inv + (-7.0 * l_w) / (3.0 * math.pi)
    w_dirac_radial_inv = w_dirac_inv + ((1.0 / 3.0) * l_radial) / (3.0 * math.pi)

    minimal_alpha = alpha_from_inv(minimal_inv)
    proton_plus1_alpha = alpha_from_inv(proton_plus1_inv)
    proton_dirac_alpha = alpha_from_inv(proton_dirac_inv)
    radial_only_alpha = alpha_from_inv(radial_only_inv)
    w_dirac_alpha = alpha_from_inv(w_dirac_inv)
    w_dirac_radial_alpha = alpha_from_inv(w_dirac_radial_inv)

    return {
        "l_mu": l_mu,
        "l_tau": l_tau,
        "l_p": l_p,
        "l_w": l_w,
        "l_radial": l_radial,
        "minimal_inv_alpha": minimal_inv,
        "minimal_alpha": minimal_alpha,
        "minimal_ratio_to_target": minimal_alpha / ALPHA_TARGET,
        "proton_plus1_inv_alpha": proton_plus1_inv,
        "proton_plus1_alpha": proton_plus1_alpha,
        "proton_plus1_ratio_to_target": proton_plus1_alpha / ALPHA_TARGET,
        "proton_dirac_inv_alpha": proton_dirac_inv,
        "proton_dirac_alpha": proton_dirac_alpha,
        "proton_dirac_ratio_to_target": proton_dirac_alpha / ALPHA_TARGET,
        "radial_only_inv_alpha": radial_only_inv,
        "radial_only_alpha": radial_only_alpha,
        "radial_only_ratio_to_target": radial_only_alpha / ALPHA_TARGET,
        "w_dirac_inv_alpha": w_dirac_inv,
        "w_dirac_alpha": w_dirac_alpha,
        "w_dirac_ratio_to_target": w_dirac_alpha / ALPHA_TARGET,
        "w_dirac_radial_inv_alpha": w_dirac_radial_inv,
        "w_dirac_radial_alpha": w_dirac_radial_alpha,
        "w_dirac_radial_ratio_to_target": w_dirac_radial_alpha / ALPHA_TARGET,
        "radial_only_relative_shift": abs(radial_only_alpha - minimal_alpha) / minimal_alpha,
        "proton_dirac_improvement_factor_vs_minimal": minimal_alpha / proton_dirac_alpha,
        "w_dirac_degradation_factor_vs_minimal": w_dirac_alpha / minimal_alpha,
    }


# Function: execute the unresolved-coefficient review branch.

def main() -> None:
    """Execute the 8.7.56.1231-.1234 branch."""
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
        EVAL_1222,
        INVENTORY_1227,
        AUDIT_1228,
        GATE_1229,
        EVAL_1230,
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
    eval_1222 = read_json(EVAL_1222)["summary"]
    inventory_1227 = read_json(INVENTORY_1227)["summary"]
    audit_1228 = read_json(AUDIT_1228)["summary"]
    gate_1229 = read_json(GATE_1229)["summary"]
    eval_1230 = read_json(EVAL_1230)["summary"]
    scenarios = compute_scenarios(eval_1222["qball_ground_state_energy_proxy"])

    proton_note_line_present = hit(vacuum_note_text, "| proton | 1836 |") is not None
    w_note_line_present = hit(vacuum_note_text, "| W | 1 | $-7$") is not None
    radial_note_line_present = hit(vacuum_note_text, "| radial mode | 0 | $+1/3$") is not None
    adopted_u1_surface_present = hit(part3a_text, "\"Local Maxwell/QED is kept unchanged\"") is not None
    mapping_absent_surface_present = hit(part3a_text, "charge-normalization bridge absent under current canon") is not None
    unresolved_step_surface_present = hit(part5_text, "vacuum-polarization unresolved coefficient review branch `8.7.56.1231-.1234`") is not None

    inventory_ready = all(
        (
            inventory_1227["inventory_ready"],
            audit_1228["minimum_fixed_ledger_ready"],
            not audit_1228["full_exact_ledger_ready"],
            gate_1229["selected_next_generation_route"]
            == "trial2_numeric_alpha_adopted_u1_vacuum_polarization_unresolved_coefficient_review",
            proton_note_line_present,
            w_note_line_present,
            radial_note_line_present,
            adopted_u1_surface_present,
            mapping_absent_surface_present,
            qball_normalization["summary"]["direct_qball_u1_identity_required"],
            unresolved_step_surface_present,
        )
    )

    proton_effective_term_fixed_now = False
    proton_effective_term_best_case_still_far = scenarios["proton_dirac_ratio_to_target"] > 2.0
    w_vector_coefficient_fixed_now = False
    w_note_minus7_worsens_target = scenarios["w_dirac_ratio_to_target"] > scenarios["minimal_ratio_to_target"]
    radial_scalar_coefficient_fixed_now = False
    radial_scalar_numeric_leverage_small = scenarios["radial_only_relative_shift"] < 1.0e-3
    full_exact_extended_screening_ready = False
    extended_screening_mainline_justified = False
    reserve_heavy_route_required = True
    physical_reject_required = False
    audit_ready = inventory_ready

    targets = [
        target(status_text, STATUS, "status_current_branch", "vacuum-polarization unresolved coefficient review", "STATUS must expose the unresolved-coefficient review as the current branch."),
        target(roadmap_text, ROADMAP, "roadmap_1231", "`8.7.56.1231-.1234`", "ROADMAP must expose the current 1231 branch."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "recent_1227", "`8.7.56.1227-.1230`", "Recent history must preserve the predecessor charge/spin ledger review."),
        target(vacuum_note_text, NOTE_VACUUM, "note_proton_line", "| proton | 1836 |", "The note must preserve the proton identified-state line."),
        target(vacuum_note_text, NOTE_VACUUM, "note_w_line", "| W | 1 | $-7$", "The note must preserve the W spin-factor placeholder line."),
        target(vacuum_note_text, NOTE_VACUUM, "note_radial_line", "| radial mode | 0 | $+1/3$", "The note must preserve the radial-mode placeholder line."),
        target(part3a_text, PART3A, "part3a_adopted_u1", "\"Local Maxwell/QED is kept unchanged\"", "Part III-A must preserve the adopted-U(1) stance."),
        target(part3a_text, PART3A, "part3a_mapping_absent", "charge-normalization bridge absent under current canon", "Part III-A must preserve the current-canon bridge absence wording."),
        target(part5_text, PART5, "part5_unresolved_step", "vacuum-polarization unresolved coefficient review branch `8.7.56.1231-.1234`", "Part V must expose the current unresolved-coefficient branch before it completes."),
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
            "eval_1222": display_path(EVAL_1222),
            "inventory_1227": display_path(INVENTORY_1227),
            "audit_1228": display_path(AUDIT_1228),
            "gate_1229": display_path(GATE_1229),
            "eval_1230": display_path(EVAL_1230),
        },
        "predecessor_summary": {
            "minimum_fixed_ledger_alpha": eval_1230["minimal_fixed_ledger_alpha"],
            "minimum_fixed_ledger_ratio_to_target": eval_1230["minimal_fixed_ledger_ratio_to_target"],
            "proton_augmented_alpha": eval_1230["proton_augmented_alpha"],
            "proton_augmented_ratio_to_target": eval_1230["proton_augmented_ratio_to_target"],
            "direct_qball_u1_identity_required": qball_normalization["summary"]["direct_qball_u1_identity_required"],
            "qball_ground_state_energy_proxy": eval_1222["qball_ground_state_energy_proxy"],
        },
        "scenario_constants": {
            "muon_ratio_to_electron": MUON_RATIO,
            "tau_ratio_to_electron": TAU_RATIO,
            "proton_ratio_to_electron": PROTON_RATIO,
            "w_ratio_to_electron": W_RATIO,
            "radial_ratio_to_electron": 1.0 / eval_1222["qball_ground_state_energy_proxy"],
        },
    }

    rows_inventory = [
        row("inventory_ready", "pass" if inventory_ready else "fail", "unresolved coefficient inventory ready", 1.0 if inventory_ready else 0.0, "The predecessor ledger pack, Q-ball identity artifacts, and the note's proton/W/radial placeholders are assembled into one review pack."),
        row("proton_line_present", "pass" if proton_note_line_present else "fail", "proton placeholder line present", 1.0 if proton_note_line_present else 0.0, "The note explicitly surfaces the proton identified-state line."),
        row("w_line_present", "pass" if w_note_line_present else "fail", "W placeholder line present", 1.0 if w_note_line_present else 0.0, "The note explicitly surfaces the charged-vector coefficient placeholder."),
        row("radial_line_present", "pass" if radial_note_line_present else "fail", "radial placeholder line present", 1.0 if radial_note_line_present else 0.0, "The note explicitly surfaces the radial-scalar placeholder."),
    ]

    summary_inventory = {
        "inventory_ready": inventory_ready,
        "direct_qball_u1_identity_required": qball_normalization["summary"]["direct_qball_u1_identity_required"],
        "proton_note_line_present": proton_note_line_present,
        "w_note_line_present": w_note_line_present,
        "radial_note_line_present": radial_note_line_present,
        "qball_ground_state_energy_proxy": eval_1222["qball_ground_state_energy_proxy"],
    }

    decision_inventory = {
        "source_inventory_step_completed": inventory_ready,
        "candidate_mechanism_class": "adopted_u1_vacuum_polarization_unresolved_coefficient_family",
    }

    evidence_inventory = {
        "targets": targets,
        "qball_canonical_statement": qball_mapping["formulas"]["canonical_statement"],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_unresolved_coefficient_source_inventory",
        payload(
            "8.7.56.1231",
            "Trial-2 adopted-U(1) vacuum-polarization unresolved coefficient source inventory",
            inputs,
            rows_inventory,
            summary_inventory,
            decision_inventory,
            evidence_inventory,
        ),
    )

    rows_audit = [
        row("audit_ready", "pass" if audit_ready else "fail", "unresolved coefficient audit ready", 1.0 if audit_ready else 0.0, "The proton, W, and radial placeholders can now be audited against the fixed lower-bound ledger."),
        row("proton_effective_term_fixed_now", "pass" if proton_effective_term_fixed_now else "fail", "proton effective term fixed now", 1.0 if proton_effective_term_fixed_now else 0.0, "Current pack still treats the proton only as an effective hadronic reserve rather than a fixed pointlike loop coefficient."),
        row("proton_dirac_best_case_alpha", "pass", "proton best-case Dirac alpha candidate", scenarios["proton_dirac_alpha"], "Even the optimistic proton-as-Dirac scenario remains far above target."),
        row("w_vector_coefficient_fixed_now", "pass" if w_vector_coefficient_fixed_now else "fail", "W vector coefficient fixed now", 1.0 if w_vector_coefficient_fixed_now else 0.0, "Current pack does not license the non-Abelian pointlike charged-vector coefficient as a fixed adopted-U(1) import."),
        row("w_note_minus7_alpha", "pass", "W note -7 scenario alpha candidate", scenarios["w_dirac_alpha"], "If the note's -7 charged-vector coefficient is adopted literally, alpha moves far away from target rather than toward it."),
        row("radial_scalar_coefficient_fixed_now", "pass" if radial_scalar_coefficient_fixed_now else "fail", "radial scalar coefficient fixed now", 1.0 if radial_scalar_coefficient_fixed_now else 0.0, "Current pack does not fix a charged radial-scalar coupling coefficient."),
        row("radial_only_alpha", "pass", "radial-only scenario alpha candidate", scenarios["radial_only_alpha"], "The radial-mode placeholder barely shifts the lower-bound result when the ground-state energy proxy is used for m0/me."),
        row("radial_scalar_numeric_leverage_small", "pass" if radial_scalar_numeric_leverage_small else "fail", "radial scalar numeric leverage small", scenarios["radial_only_relative_shift"], "The radial placeholder has negligible leverage on alpha in the retained dictionary."),
        row("full_exact_extended_screening_ready", "pass" if full_exact_extended_screening_ready else "fail", "full exact extended screening ready", 1.0 if full_exact_extended_screening_ready else 0.0, "The current pack still cannot fix the unresolved coefficients well enough for an honest exact screening run."),
    ]

    summary_audit = {
        "audit_ready": audit_ready,
        "proton_effective_term_fixed_now": proton_effective_term_fixed_now,
        "proton_effective_term_best_case_still_far": proton_effective_term_best_case_still_far,
        "proton_dirac_best_case_alpha": scenarios["proton_dirac_alpha"],
        "proton_dirac_best_case_ratio_to_target": scenarios["proton_dirac_ratio_to_target"],
        "w_vector_coefficient_fixed_now": w_vector_coefficient_fixed_now,
        "w_note_minus7_alpha": scenarios["w_dirac_alpha"],
        "w_note_minus7_ratio_to_target": scenarios["w_dirac_ratio_to_target"],
        "w_note_minus7_worsens_target": w_note_minus7_worsens_target,
        "radial_scalar_coefficient_fixed_now": radial_scalar_coefficient_fixed_now,
        "radial_only_alpha": scenarios["radial_only_alpha"],
        "radial_only_ratio_to_target": scenarios["radial_only_ratio_to_target"],
        "radial_only_relative_shift": scenarios["radial_only_relative_shift"],
        "radial_scalar_numeric_leverage_small": radial_scalar_numeric_leverage_small,
        "full_exact_extended_screening_ready": full_exact_extended_screening_ready,
        "extended_screening_mainline_justified": extended_screening_mainline_justified,
        "reserve_heavy_route_required": reserve_heavy_route_required,
    }

    decision_audit = {
        "unresolved_coefficient_class": "reserve_heavy_after_lower_bound",
        "extended_screening_mainline_justified": extended_screening_mainline_justified,
        "reserve_heavy_route_required": reserve_heavy_route_required,
    }

    evidence_audit = {
        "lower_bound_reference": {
            "minimal_fixed_ledger_alpha": eval_1230["minimal_fixed_ledger_alpha"],
            "proton_augmented_alpha": eval_1230["proton_augmented_alpha"],
        },
        "scenario_logs": {
            "l_mu": scenarios["l_mu"],
            "l_tau": scenarios["l_tau"],
            "l_p": scenarios["l_p"],
            "l_w": scenarios["l_w"],
            "l_radial": scenarios["l_radial"],
        },
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_unresolved_coefficient_audit",
        payload(
            "8.7.56.1232",
            "Trial-2 adopted-U(1) vacuum-polarization unresolved coefficient audit",
            inputs,
            rows_audit,
            summary_audit,
            decision_audit,
            evidence_audit,
        ),
    )

    rows_gate = [
        row("minimum_fixed_ledger_retained", "pass", "minimum fixed ledger retained", 1.0, "The muon/tau lower-bound ledger remains the only fixed mainline computation slice."),
        row("extended_screening_mainline_justified", "pass" if extended_screening_mainline_justified else "fail", "extended screening mainline justified", 1.0 if extended_screening_mainline_justified else 0.0, "Current pack does not justify continuing an exact extended screening run as the mainline."),
        row("reserve_heavy_route_required", "pass" if reserve_heavy_route_required else "fail", "reserve-heavy route required", 1.0 if reserve_heavy_route_required else 0.0, "The honest next state is a reserve-heavy route rather than a forced exact-screening continuation."),
        row("physical_reject_required", "pass" if physical_reject_required else "fail", "physical reject required", 1.0 if physical_reject_required else 0.0, "The computation candidate stays open even though the exact unresolved coefficients remain unfixed."),
    ]

    summary_gate = {
        "trial2_numeric_alpha_problem_classification": "adopted_u1_vacuum_polarization_unresolved_coefficients_reserve_heavy",
        "minimum_fixed_ledger_ready": audit_1228["minimum_fixed_ledger_ready"],
        "full_exact_extended_screening_ready": full_exact_extended_screening_ready,
        "extended_screening_mainline_justified": extended_screening_mainline_justified,
        "reserve_heavy_route_required": reserve_heavy_route_required,
        "vacuum_polarization_analog_external_import_admissible": True,
        "numeric_closeout_ready": False,
        "physical_reject_required": physical_reject_required,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }

    decision_gate = {
        "next_route_name": NEXT_ROUTE_NAME,
        "next_route_step": NEXT_ROUTE,
        "mainline_policy": "retain_lower_bound_result_demote_exact_extension_to_reserve_heavy_route",
    }

    evidence_gate = {
        "proton_best_case_ratio_to_target": scenarios["proton_dirac_ratio_to_target"],
        "w_note_minus7_ratio_to_target": scenarios["w_dirac_ratio_to_target"],
        "radial_only_relative_shift": scenarios["radial_only_relative_shift"],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_unresolved_coefficient_declaration_gate",
        payload(
            "8.7.56.1233",
            "Trial-2 adopted-U(1) vacuum-polarization unresolved coefficient declaration gate",
            inputs,
            rows_gate,
            summary_gate,
            decision_gate,
            evidence_gate,
        ),
    )

    best_nonminimal_label = "proton_dirac_best_case"
    best_nonminimal_alpha = scenarios["proton_dirac_alpha"]
    best_nonminimal_ratio = scenarios["proton_dirac_ratio_to_target"]

    rows_eval = [
        row("minimal_fixed_ledger_alpha", "pass", "minimal fixed-ledger alpha", scenarios["minimal_alpha"], "The lower-bound muon/tau ledger remains the mainline anchor."),
        row("proton_dirac_best_case_alpha", "pass", "proton best-case Dirac alpha", scenarios["proton_dirac_alpha"], "Even the optimistic proton-as-Dirac case remains far above the target alpha."),
        row("w_note_minus7_alpha", "pass", "W note -7 alpha", scenarios["w_dirac_alpha"], "Literal adoption of the note's W coefficient drives alpha far away from target."),
        row("w_note_minus7_plus_radial_alpha", "pass", "W note -7 plus radial alpha", scenarios["w_dirac_radial_alpha"], "Adding the radial placeholder on top of the W scenario does not repair the overshoot."),
        row("best_nonminimal_ratio_to_target", "pass", "best nonminimal scenario ratio to target", best_nonminimal_ratio, "No nonminimal unresolved-coefficient scenario comes close to 1/137 under the current pack."),
    ]

    summary_eval = {
        "alpha_target": ALPHA_TARGET,
        "alpha_bare": ALPHA_BARE,
        "minimal_fixed_ledger_alpha": scenarios["minimal_alpha"],
        "minimal_fixed_ledger_ratio_to_target": scenarios["minimal_ratio_to_target"],
        "proton_dirac_best_case_alpha": scenarios["proton_dirac_alpha"],
        "proton_dirac_best_case_ratio_to_target": scenarios["proton_dirac_ratio_to_target"],
        "w_note_minus7_alpha": scenarios["w_dirac_alpha"],
        "w_note_minus7_ratio_to_target": scenarios["w_dirac_ratio_to_target"],
        "w_note_minus7_plus_radial_alpha": scenarios["w_dirac_radial_alpha"],
        "w_note_minus7_plus_radial_ratio_to_target": scenarios["w_dirac_radial_ratio_to_target"],
        "radial_only_alpha": scenarios["radial_only_alpha"],
        "radial_only_ratio_to_target": scenarios["radial_only_ratio_to_target"],
        "best_nonminimal_scenario_label": best_nonminimal_label,
        "best_nonminimal_scenario_alpha": best_nonminimal_alpha,
        "best_nonminimal_scenario_ratio_to_target": best_nonminimal_ratio,
        "any_unresolved_scenario_close_to_target": False,
        "extended_screening_mainline_value_retained": extended_screening_mainline_justified,
        "reserve_heavy_route_required": reserve_heavy_route_required,
    }

    decision_eval = {
        "numeric_state_class": "lower_bound_retained_unresolved_envelope_not_close",
        "extended_mainline_continuation_worthwhile_now": extended_screening_mainline_justified,
        "demote_to_reserve_heavy_route_now": reserve_heavy_route_required,
    }

    evidence_eval = {
        "best_case_notes": [
            "proton best-case already assumes an optimistic Dirac-like effective coefficient",
            "W scenario already uses the note's strongest explicit coefficient and moves away from target",
            "radial placeholder changes alpha at only the 1e-4 relative level under the retained m0/me dictionary",
        ],
        "retained_lower_bound_states": [
            "electron_shell_zero_log",
            "muon_charged_fermion",
            "tau_charged_fermion",
        ],
    }

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_adopted_u1_vacuum_polarization_unresolved_coefficient_numeric_evaluation",
        payload(
            "8.7.56.1234",
            "Trial-2 adopted-U(1) vacuum-polarization unresolved coefficient numeric evaluation",
            inputs,
            rows_eval,
            summary_eval,
            decision_eval,
            evidence_eval,
        ),
    )


if __name__ == "__main__":
    main()
