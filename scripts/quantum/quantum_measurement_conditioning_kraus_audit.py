#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_measurement_conditioning_kraus_audit.py

Step 8.7.50.5:
Recover the effective Lueders/Kraus update rule from detector-environment
decoherence plus conditioning on a macroscopic detector record.

Inputs:
  - output/public/quantum/quantum_measurement_dynamic_collapse_simulation_metrics.json
  - output/public/quantum/quantum_measurement_pointer_basis_grounding_audit_metrics.json

Outputs:
  - output/public/quantum/quantum_measurement_conditioning_kraus_audit_metrics.json
  - output/public/quantum/quantum_measurement_conditioning_kraus_confusion.csv

Assumptions:
  - The pointer basis from step 8.7.50.4 is already grounded as stable detector
    P-background modes.
  - The environment trace suppresses off-diagonal coherence according to the
    already-frozen dynamic-collapse metrics; no new microscopic parameters are
    introduced here.
  - Finite detector response is represented by a classical confusion matrix on
    the pointer sectors, which is sufficient to recover diagonal Kraus
    operators in the pointer basis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

# Guard: add the repository root once so local packages resolve predictably.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# Class: store one row of the detector-response confusion table.
@dataclass(frozen=True)
class ConfusionRow:
    scenario: str
    record_label: str
    branch_label: str
    prior_branch_probability: float
    response_probability: float
    joint_probability: float
    posterior_branch_probability_given_record: float


# Function: return the current UTC timestamp in ISO 8601 form.

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: render a path relative to the repository root when possible.

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# Function: read a UTF-8 JSON file into a Python dictionary.

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# Function: return the binary response probability for a given record/branch pair.

def _response_probability(*, record_label: str, branch_label: str, epsilon: float) -> float:
    same_label = record_label == branch_label
    return float(1.0 - epsilon if same_label else epsilon)


# Function: build confusion-table rows for one detector-response scenario.

def _scenario_rows(
    *,
    scenario: str,
    epsilon: float,
    prior_plus: float,
    prior_minus: float,
) -> List[ConfusionRow]:
    priors = {"+": prior_plus, "-": prior_minus}
    rows: List[ConfusionRow] = []

    for record_label in ("+", "-"):
        record_probability = 0.0
        branch_joint: Dict[str, float] = {}

        for branch_label, prior in priors.items():
            response_probability = _response_probability(
                record_label=record_label,
                branch_label=branch_label,
                epsilon=epsilon,
            )
            joint_probability = prior * response_probability
            branch_joint[branch_label] = joint_probability
            record_probability += joint_probability

        for branch_label, prior in priors.items():
            response_probability = _response_probability(
                record_label=record_label,
                branch_label=branch_label,
                epsilon=epsilon,
            )
            joint_probability = branch_joint[branch_label]
            posterior = joint_probability / record_probability if record_probability > 0.0 else float("nan")
            rows.append(
                ConfusionRow(
                    scenario=scenario,
                    record_label=record_label,
                    branch_label=branch_label,
                    prior_branch_probability=float(prior),
                    response_probability=float(response_probability),
                    joint_probability=float(joint_probability),
                    posterior_branch_probability_given_record=float(posterior),
                )
            )

    return rows


# Function: summarize the diagonal Kraus/POVM data implied by one response matrix.

def _kraus_summary(*, epsilon: float, rows: List[ConfusionRow]) -> Dict[str, Any]:
    by_record_branch = {(row.record_label, row.branch_label): row for row in rows}
    complete_plus = _response_probability(record_label="+", branch_label="+", epsilon=epsilon) + _response_probability(
        record_label="-",
        branch_label="+",
        epsilon=epsilon,
    )
    complete_minus = _response_probability(record_label="+", branch_label="-", epsilon=epsilon) + _response_probability(
        record_label="-",
        branch_label="-",
        epsilon=epsilon,
    )
    completeness_error = max(abs(complete_plus - 1.0), abs(complete_minus - 1.0))

    return {
        "epsilon_response": float(epsilon),
        "response_matrix": {
            "record_plus_given_branch_plus": float(by_record_branch[("+", "+")].response_probability),
            "record_plus_given_branch_minus": float(by_record_branch[("+", "-")].response_probability),
            "record_minus_given_branch_plus": float(by_record_branch[("-", "+")].response_probability),
            "record_minus_given_branch_minus": float(by_record_branch[("-", "-")].response_probability),
        },
        "k_plus_diag": [float(math.sqrt(1.0 - epsilon)), float(math.sqrt(epsilon))],
        "k_minus_diag": [float(math.sqrt(epsilon)), float(math.sqrt(1.0 - epsilon))],
        "povm_plus_diag": [float(1.0 - epsilon), float(epsilon)],
        "povm_minus_diag": [float(epsilon), float(1.0 - epsilon)],
        "posterior_branch_plus_given_plus_record": float(by_record_branch[("+", "+")].posterior_branch_probability_given_record),
        "posterior_branch_minus_given_minus_record": float(by_record_branch[("-", "-")].posterior_branch_probability_given_record),
        "completeness_error_abs_max": float(completeness_error),
    }


# Function: combine frozen measurement artifacts into the C2 audit payload.

def build_payload(
    *,
    sim_metrics: Dict[str, Any],
    pointer_metrics: Dict[str, Any],
    sim_metrics_path: Path,
    pointer_metrics_path: Path,
) -> Dict[str, Any]:
    sim_summary = sim_metrics.get("summary") if isinstance(sim_metrics.get("summary"), dict) else {}
    pointer_summary = pointer_metrics.get("summary") if isinstance(pointer_metrics.get("summary"), dict) else {}
    pointer_decision = pointer_metrics.get("decision") if isinstance(pointer_metrics.get("decision"), dict) else {}

    pointer_status = str(pointer_decision.get("c1_pointer_basis_status", "unknown"))
    coherence_suppression_ratio = float(sim_summary.get("coherence_suppression_ratio"))
    final_coherence_median = float(sim_summary.get("final_coherence_median"))
    pointer_consensus_fraction = float(sim_summary.get("pointer_consensus_fraction"))
    branch_stable_fraction = float(sim_summary.get("branch_stable_fraction"))
    branch_plus_fraction = float(sim_summary.get("branch_plus_fraction"))
    branch_minus_fraction = float(sim_summary.get("branch_minus_fraction"))
    max_static_sign_error = float(pointer_summary.get("max_static_sign_error_upper_bound"))

    resolved_branch_weight = branch_plus_fraction + branch_minus_fraction

    # Guard: reject degenerate inputs before normalizing resolved branch weights.
    if resolved_branch_weight <= 0.0:
        raise ValueError("resolved branch weight must be positive")

    prior_plus = branch_plus_fraction / resolved_branch_weight
    prior_minus = branch_minus_fraction / resolved_branch_weight
    epsilon_consensus = max(0.0, 1.0 - pointer_consensus_fraction)
    epsilon_instability = max(0.0, 1.0 - branch_stable_fraction)
    epsilon_nominal = max(max_static_sign_error, epsilon_consensus)
    epsilon_upper_bound = max(epsilon_nominal, epsilon_instability)

    nominal_rows = _scenario_rows(
        scenario="nominal",
        epsilon=epsilon_nominal,
        prior_plus=prior_plus,
        prior_minus=prior_minus,
    )
    upper_rows = _scenario_rows(
        scenario="upper_bound",
        epsilon=epsilon_upper_bound,
        prior_plus=prior_plus,
        prior_minus=prior_minus,
    )

    nominal_summary = _kraus_summary(epsilon=epsilon_nominal, rows=nominal_rows)
    upper_summary = _kraus_summary(epsilon=epsilon_upper_bound, rows=upper_rows)

    passes = {
        "pointer_basis_closed": pointer_status == "closed",
        "offdiagonal_suppression_strong_enough": bool(coherence_suppression_ratio <= 5.0e-2),
        "nominal_detector_response_near_luders": bool(epsilon_nominal <= 5.0e-2),
        "finite_response_kraus_is_cptp": bool(upper_summary["completeness_error_abs_max"] <= 1.0e-12 and epsilon_upper_bound <= 1.0e-1),
    }
    all_pass = all(passes.values())

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.50.5", "name": "Measurement conditioning and Kraus recovery"},
        "inputs": {
            "dynamic_collapse_metrics_json": _rel(sim_metrics_path),
            "pointer_basis_grounding_metrics_json": _rel(pointer_metrics_path),
        },
        "intent": "Recover the diagonal Lueders/Kraus update as the effective consequence of detector-environment decoherence plus conditioning on a macroscopic detector record.",
        "assumptions": [
            "The pointer basis from step 8.7.50.4 is already fixed as stable detector P-background modes.",
            "The environment trace suppresses off-diagonal terms with the measured coherence-suppression ratio from the frozen dynamic-collapse simulation.",
            "Finite detector response can be represented by a classical confusion matrix on the pointer sectors without introducing new P-model free parameters.",
        ],
        "formulas": {
            "decohered_state": "rho_SD^(dec) = sum_m Pi_m rho Pi_m otimes rho_D^(m) + O(r_coh), r_coh := |rho_01(t_read)| / |rho_01(0)|",
            "conditioning_map": "I_r(rho) = Tr_DE[(I otimes F_r otimes I) U (rho otimes rho_D,0 otimes rho_E,0) U^dagger] = sum_m r_{r|m} Pi_m rho Pi_m + O(r_coh)",
            "diagonal_kraus": "M_r = sum_m sqrt(r_{r|m}) Pi_m, E_r = M_r^dagger M_r, sum_r E_r = I",
            "luders_limit": "If r_{r|m} -> delta_{rm}, then rho_r -> Pi_r rho Pi_r / Tr(Pi_r rho).",
            "binary_response": "For r in {+,-}, M_+ = sqrt(1-eps) Pi_+ + sqrt(eps) Pi_-, M_- = sqrt(eps) Pi_+ + sqrt(1-eps) Pi_-",
        },
        "summary": {
            "coherence_suppression_ratio": float(coherence_suppression_ratio),
            "final_coherence_median": float(final_coherence_median),
            "epsilon_static_overlap": float(max_static_sign_error),
            "epsilon_consensus": float(epsilon_consensus),
            "epsilon_instability": float(epsilon_instability),
            "epsilon_response_nominal": float(epsilon_nominal),
            "epsilon_response_upper_bound": float(epsilon_upper_bound),
            "branch_plus_prior_resolved": float(prior_plus),
            "branch_minus_prior_resolved": float(prior_minus),
            "luders_limit_distance_nominal": float(epsilon_nominal),
            "luders_limit_distance_upper_bound": float(epsilon_upper_bound),
        },
        "scenarios": {
            "nominal": nominal_summary,
            "upper_bound": upper_summary,
        },
        "confusion_rows": [asdict(row) for row in nominal_rows + upper_rows],
        "decision": {
            "c2_conditioning_status": "closed_effective_derivation" if all_pass else "not_closed",
            "measurement_status": "conditioning_closed_irreversibility_pending" if all_pass else "conditioning_not_closed",
            "born_route_status": "conditional_detection_and_update_closed_irreversibility_pending" if all_pass else "update_not_recovered",
            "passes": passes,
            "new_pmodel_free_parameters_introduced": False,
            "next_required_steps": ["8.7.50.6", "8.7.50.7"],
        },
    }


# Function: write the response/confusion table in CSV form.

def _write_confusion_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("confusion_rows") if isinstance(payload.get("confusion_rows"), list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "record_label",
                "branch_label",
                "prior_branch_probability",
                "response_probability",
                "joint_probability",
                "posterior_branch_probability_given_record",
            ],
        )
        writer.writeheader()
        for row in rows:
            # Guard: skip malformed rows instead of failing the whole export.
            if not isinstance(row, dict):
                continue

            writer.writerow(row)


# Function: parse CLI arguments, run the audit, and write the outputs.

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Recover the effective Lueders/Kraus update from decoherence plus conditioning.")
    ap.add_argument(
        "--sim-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_dynamic_collapse_simulation_metrics.json"),
        help="Input dynamic-collapse metrics JSON path.",
    )
    ap.add_argument(
        "--pointer-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_pointer_basis_grounding_audit_metrics.json"),
        help="Input pointer-basis grounding metrics JSON path.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_conditioning_kraus_audit_metrics.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_conditioning_kraus_confusion.csv"),
        help="Output CSV path.",
    )
    args = ap.parse_args(argv)

    sim_metrics_path = Path(args.sim_metrics)
    pointer_metrics_path = Path(args.pointer_metrics)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    # Guard: resolve relative input/output paths against the repository root.
    if not sim_metrics_path.is_absolute():
        sim_metrics_path = (ROOT / sim_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not pointer_metrics_path.is_absolute():
        pointer_metrics_path = (ROOT / pointer_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    sim_metrics = _read_json(sim_metrics_path)
    pointer_metrics = _read_json(pointer_metrics_path)
    payload = build_payload(
        sim_metrics=sim_metrics,
        pointer_metrics=pointer_metrics,
        sim_metrics_path=sim_metrics_path,
        pointer_metrics_path=pointer_metrics_path,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_confusion_csv(out_csv, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_measurement_conditioning_kraus_audit",
                "phase": "8.7.50.5",
                "inputs": {
                    "dynamic_collapse_metrics_json": _rel(sim_metrics_path),
                    "pointer_basis_grounding_metrics_json": _rel(pointer_metrics_path),
                },
                "outputs": {
                    "quantum_measurement_conditioning_kraus_audit_metrics_json": _rel(out_json),
                    "quantum_measurement_conditioning_kraus_confusion_csv": _rel(out_csv),
                },
                "decision": payload.get("decision"),
            }
        )
    except Exception:
        pass

    return 0


# Guard: support direct CLI execution.

if __name__ == "__main__":
    raise SystemExit(main())
