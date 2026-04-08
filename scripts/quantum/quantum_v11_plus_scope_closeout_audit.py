#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_v11_plus_scope_closeout_audit.py

Step 8.7.50.7:
Freeze the v1.1+ quantum closeout boundary by combining the already-fixed
Born, measurement, and entanglement artifacts into one scope audit.

Inputs:
  - output/public/quantum/born_phase_diffusion_audit_metrics.json
  - output/public/quantum/born_linear_detector_response_audit_metrics.json
  - output/public/quantum/quantum_measurement_pointer_basis_grounding_audit_metrics.json
  - output/public/quantum/quantum_measurement_conditioning_kraus_audit_metrics.json
  - output/public/quantum/quantum_measurement_dynamic_collapse_stability_audit.json
  - output/public/quantum/entanglement_source_dynamics_three_wave_mixing_metrics.json
  - output/public/quantum/entanglement_bell_dataset_connection_audit_metrics.json
  - output/public/quantum/quantum_information_direct_p_evaluation_metrics.json
  - output/public/quantum/quantum_information_error_channel_mapping_metrics.json

Outputs:
  - output/public/quantum/quantum_v11_plus_scope_closeout_audit_metrics.json
  - output/public/quantum/quantum_v11_plus_scope_closeout_cases.csv

Assumptions:
  - A1/A2/C1/C2/B1/B2/B3 artifacts are already frozen and become the sole
    evidence base for the v1.1+ closeout boundary.
  - Irreversibility is evaluated only at the coarse-grained measurement level;
    microscopic Loschmidt/Poincare issues are classified as generic
    thermodynamic-limit residuals rather than P-model-specific failures.
  - No new free parameters are introduced in this closeout audit.
"""

from __future__ import annotations

import argparse
import csv
import json
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


# Class: store one route-level closeout row for CSV export.
@dataclass(frozen=True)
class RouteCloseoutRow:
    route_id: str
    closeout_status: str
    representative_metric_name: str
    representative_metric_value: float
    support_metric_name: str
    support_metric_value: float
    residual_scope: str


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


# Function: return a nested dictionary when present, else an empty mapping.

def _as_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


# Function: coerce one key from a dictionary into a finite float-compatible value.

def _float(payload: Dict[str, Any], key: str) -> float:
    return float(payload.get(key))


# Function: normalize the free-parameter flag across older/newer artifact schemas.

def _introduced_new_parameters(decision: Dict[str, Any]) -> bool:
    raw = decision.get("new_pmodel_free_parameters_introduced")

    if raw is None:
        raw = decision.get("new_free_parameters_introduced")

    return bool(raw)


# Function: build one route-level CSV row for the overall closeout table.

def _route_rows(
    *,
    born_linear_summary: Dict[str, Any],
    conditioning_summary: Dict[str, Any],
    stability_summary: Dict[str, Any],
    entanglement_summary: Dict[str, Any],
    born_status: str,
    measurement_status: str,
    entanglement_status: str,
) -> List[RouteCloseoutRow]:
    return [
        RouteCloseoutRow(
            route_id="born",
            closeout_status=born_status,
            representative_metric_name="max_flat_field_frequency_error",
            representative_metric_value=float(born_linear_summary.get("max_flat_field_frequency_error")),
            support_metric_name="max_relative_nonlinear_correction_bound",
            support_metric_value=float(born_linear_summary.get("max_relative_nonlinear_correction_bound")),
            residual_scope="single_event_probability_foundations_not_p_specific",
        ),
        RouteCloseoutRow(
            route_id="measurement",
            closeout_status=measurement_status,
            representative_metric_name="coherence_suppression_ratio",
            representative_metric_value=float(conditioning_summary.get("coherence_suppression_ratio")),
            support_metric_name="tau50_cv",
            support_metric_value=float(_as_dict(stability_summary, "tau50_stats_s").get("cv")),
            residual_scope="thermodynamic_limit_irreversibility_not_p_specific",
        ),
        RouteCloseoutRow(
            route_id="entanglement",
            closeout_status=entanglement_status,
            representative_metric_name="min_chsh_visibility_proxy",
            representative_metric_value=float(entanglement_summary.get("min_chsh_visibility_proxy")),
            support_metric_name="max_chsh_pair_decoherence_budget",
            support_metric_value=float(entanglement_summary.get("max_chsh_pair_decoherence_budget")),
            residual_scope="selection_watch_retained_nonblocking",
        ),
    ]


# Function: combine the route-level artifacts into the 8.7.50.7 closeout payload.

def build_payload(
    *,
    born_phase_metrics: Dict[str, Any],
    born_linear_metrics: Dict[str, Any],
    pointer_metrics: Dict[str, Any],
    conditioning_metrics: Dict[str, Any],
    stability_metrics: Dict[str, Any],
    entanglement_source_metrics: Dict[str, Any],
    entanglement_dataset_metrics: Dict[str, Any],
    quantum_information_direct_metrics: Dict[str, Any],
    quantum_information_error_channel_metrics: Dict[str, Any],
    born_phase_metrics_path: Path,
    born_linear_metrics_path: Path,
    pointer_metrics_path: Path,
    conditioning_metrics_path: Path,
    stability_metrics_path: Path,
    entanglement_source_metrics_path: Path,
    entanglement_dataset_metrics_path: Path,
    quantum_information_direct_metrics_path: Path,
    quantum_information_error_channel_metrics_path: Path,
) -> Dict[str, Any]:
    born_phase_summary = _as_dict(born_phase_metrics, "summary")
    born_phase_decision = _as_dict(born_phase_metrics, "decision")
    born_linear_summary = _as_dict(born_linear_metrics, "summary")
    born_linear_decision = _as_dict(born_linear_metrics, "decision")
    pointer_summary = _as_dict(pointer_metrics, "summary")
    pointer_decision = _as_dict(pointer_metrics, "decision")
    conditioning_summary = _as_dict(conditioning_metrics, "summary")
    conditioning_decision = _as_dict(conditioning_metrics, "decision")
    stability_summary = _as_dict(stability_metrics, "summary")
    entanglement_source_summary = _as_dict(entanglement_source_metrics, "summary")
    entanglement_source_decision = _as_dict(entanglement_source_metrics, "decision")
    entanglement_dataset_summary = _as_dict(entanglement_dataset_metrics, "summary")
    entanglement_dataset_decision = _as_dict(entanglement_dataset_metrics, "decision")
    entanglement_dataset_passes = _as_dict(entanglement_dataset_decision, "passes")
    quantum_information_direct_summary = _as_dict(quantum_information_direct_metrics, "summary")
    quantum_information_direct_decision = _as_dict(quantum_information_direct_metrics, "decision")
    quantum_information_direct_passes = _as_dict(quantum_information_direct_decision, "passes")
    quantum_information_error_summary = _as_dict(quantum_information_error_channel_metrics, "summary")
    quantum_information_error_decision = _as_dict(quantum_information_error_channel_metrics, "decision")
    quantum_information_error_passes = _as_dict(quantum_information_error_decision, "passes")

    passes = {
        "born_a1_closed": born_phase_decision.get("a1_gap_status") == "closed",
        "born_a2_closed": born_linear_decision.get("a2_gap_status") == "closed",
        "measurement_c1_closed": pointer_decision.get("c1_pointer_basis_status") == "closed",
        "measurement_c2_closed": conditioning_decision.get("c2_conditioning_status") == "closed_effective_derivation",
        "measurement_env_seed_stability_confirmed": str(stability_summary.get("overall_status")) == "pass",
        "entanglement_b1_closed": entanglement_source_decision.get("b1_source_dynamics_status") == "closed",
        "entanglement_b2_closed": entanglement_dataset_decision.get("b2_dataset_connection_status") == "closed_effective_audit",
        "entanglement_b3_closed": entanglement_dataset_decision.get("b3_pair_decoherence_status") == "closed_integrated_bound",
        "selection_watch_preserved": bool(entanglement_dataset_passes.get("selection_watch_preserved")),
    }

    no_new_parameters = not any(
        [
            _introduced_new_parameters(born_phase_decision),
            _introduced_new_parameters(born_linear_decision),
            _introduced_new_parameters(pointer_decision),
            _introduced_new_parameters(conditioning_decision),
            _introduced_new_parameters(entanglement_source_decision),
            _introduced_new_parameters(entanglement_dataset_decision),
            _introduced_new_parameters(quantum_information_direct_decision),
            _introduced_new_parameters(quantum_information_error_decision),
        ]
    )
    passes["no_new_pmodel_free_parameters"] = no_new_parameters
    quantum_information_fixed = all(
        [
            str(quantum_information_direct_decision.get("overall_status")) == "entry_fixed_consistency_pass",
            str(quantum_information_direct_decision.get("minimal_connection_status")) == "entry_fixed",
            bool(quantum_information_direct_passes.get("no_new_pmodel_free_parameters")),
            bool(quantum_information_error_passes.get("dephasing_rate_available_for_all_platforms")),
            bool(quantum_information_error_passes.get("amplitude_damping_origin_fixed")),
            bool(quantum_information_error_passes.get("depolarizing_not_primary")),
            str(quantum_information_error_decision.get("overall_status")) == "minimal_connection_error_origin_fixed",
            not bool(quantum_information_error_decision.get("direct_platform_t1_prediction")),
        ]
    )
    passes["quantum_information_minimal_connection_fixed"] = quantum_information_fixed

    measurement_closed = all(
        [
            passes["measurement_c1_closed"],
            passes["measurement_c2_closed"],
            passes["measurement_env_seed_stability_confirmed"],
        ]
    )
    born_closed = all(
        [
            passes["born_a1_closed"],
            passes["born_a2_closed"],
            passes["measurement_c1_closed"],
            passes["measurement_c2_closed"],
            passes["measurement_env_seed_stability_confirmed"],
        ]
    )
    entanglement_closed = all(
        [
            passes["entanglement_b1_closed"],
            passes["entanglement_b2_closed"],
            passes["entanglement_b3_closed"],
            passes["selection_watch_preserved"],
        ]
    )
    overall_closed = born_closed and measurement_closed and entanglement_closed and no_new_parameters

    born_status = "p_specific_derivation_closed_common_probability_residual" if born_closed else "closeout_incomplete"
    measurement_status = "effective_derivation_closed_statmech_residual" if measurement_closed else "closeout_incomplete"
    entanglement_status = (
        "effective_audit_closed_selection_watch_retained" if entanglement_closed else "closeout_incomplete"
    )
    overall_status = "completed_scope_fixed" if overall_closed else "closeout_incomplete"
    quantum_information_status = (
        "minimal_connection_entry_fixed" if quantum_information_fixed else "minimal_connection_not_fixed"
    )
    quantum_information_protocol_status = "standard_hilbert_math_scope_out"

    route_rows = _route_rows(
        born_linear_summary=born_linear_summary,
        conditioning_summary=conditioning_summary,
        stability_summary=stability_summary,
        entanglement_summary=entanglement_dataset_summary,
        born_status=born_status,
        measurement_status=measurement_status,
        entanglement_status=entanglement_status,
    )

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.50.7", "name": "Quantum v1.1+ scope closeout audit"},
        "inputs": {
            "born_phase_diffusion_metrics_json": _rel(born_phase_metrics_path),
            "born_linear_detector_response_metrics_json": _rel(born_linear_metrics_path),
            "measurement_pointer_basis_grounding_metrics_json": _rel(pointer_metrics_path),
            "measurement_conditioning_kraus_metrics_json": _rel(conditioning_metrics_path),
            "measurement_dynamic_collapse_stability_audit_json": _rel(stability_metrics_path),
            "entanglement_source_dynamics_metrics_json": _rel(entanglement_source_metrics_path),
            "entanglement_bell_dataset_connection_metrics_json": _rel(entanglement_dataset_metrics_path),
            "quantum_information_direct_p_evaluation_metrics_json": _rel(quantum_information_direct_metrics_path),
            "quantum_information_error_channel_mapping_metrics_json": _rel(
                quantum_information_error_channel_metrics_path
            ),
        },
        "intent": "Freeze the v1.1+ quantum closeout boundary by separating the already-closed P-specific derivations, the quantum-information minimal connection, the common probability/statistical residuals, and the remaining protocol-level scope-out items.",
        "assumptions": [
            "A1/A2/C1/C2/B1/B2/B3 are already frozen and become the sole evidence base for this closeout.",
            "Single-event probability assignment is treated as a common foundations-of-probability residual rather than a P-model-specific gap.",
            "Microscopic irreversibility is not re-derived here; only the coarse-grained one-way behavior of the reduced detector record is audited.",
            "Selection watch remains a diagnostic guard and is not optimized away.",
            "Quantum-information protocol mathematics remains standard Hilbert-space structure and is not rederived in P language.",
        ],
        "formulas": {
            "born_closeout_rule": "A1 + A2 + C1 + C2 => p(x) propto |psi(x)|^2 is fixed by the P-specific route; the remaining single-event issue belongs to common probability foundations.",
            "measurement_scope_rule": "r_coh << 1, tau50/tauD = O(1), pointer consensus -> 1, and branch reversal << 1 support effective one-way measurement dynamics.",
            "irreversibility_boundary": "The unresolved issue is not the form of the update rule but why Tr_E looks irreversible in the thermodynamic limit (Loschmidt/Poincare class).",
            "entanglement_scope_rule": "B1 + B2 + B3 freeze source dynamics, Bell-dataset connection, and integrated pair-dephasing bounds while preserving selection watch.",
            "quantum_information_minimal_connection_rule": "Gamma_deph = omega_*^2 (k_B T_env / chi_P) tau_free fixes T2 / gate-loss / pair-depth at minimal-connection level, while C1 fixes amplitude-damping origin and depolarizing stays subleading.",
        },
        "summary": {
            "born_probe_rows_with_phase_mixing": int(born_phase_summary.get("probe_rows_with_phase_mixing")),
            "born_probe_rows_total": int(born_phase_summary.get("probe_rows_total")),
            "born_critical_thermal_ratio_min": float(born_phase_summary.get("critical_thermal_ratio_min")),
            "born_critical_thermal_ratio_max": float(born_phase_summary.get("critical_thermal_ratio_max")),
            "born_max_flat_field_frequency_error": float(born_linear_summary.get("max_flat_field_frequency_error")),
            "born_max_relative_nonlinear_correction_bound": float(born_linear_summary.get("max_relative_nonlinear_correction_bound")),
            "measurement_coherence_suppression_ratio": float(conditioning_summary.get("coherence_suppression_ratio")),
            "measurement_tau_D_s": float(pointer_summary.get("tau_D_s")),
            "measurement_tau50_reference_s": float(pointer_summary.get("tau50_reference_s")),
            "measurement_tau50_over_tauD": float(pointer_summary.get("tau50_over_tauD")),
            "measurement_tau50_cv": float(_as_dict(stability_summary, "tau50_stats_s").get("cv")),
            "measurement_pointer_consensus_min": float(_as_dict(stability_summary, "pointer_consensus_stats").get("min")),
            "measurement_branch_reversal_max": float(_as_dict(stability_summary, "branch_reversal_stats").get("max")),
            "measurement_epsilon_response_nominal": float(conditioning_summary.get("epsilon_response_nominal")),
            "measurement_epsilon_response_upper_bound": float(conditioning_summary.get("epsilon_response_upper_bound")),
            "measurement_stability_run_count": int(stability_summary.get("n_runs")),
            "entanglement_min_effective_schmidt_rank": float(entanglement_source_summary.get("min_effective_schmidt_rank")),
            "entanglement_min_chsh_visibility_proxy": float(entanglement_dataset_summary.get("min_chsh_visibility_proxy")),
            "entanglement_max_pair_decoherence_budget": float(entanglement_dataset_summary.get("max_chsh_pair_decoherence_budget")),
            "entanglement_selection_watch_active_dataset_count": int(entanglement_dataset_summary.get("selection_watch_active_dataset_count")),
            "quantum_information_exact_proxy_pass_count": int(quantum_information_direct_summary.get("exact_proxy_pass_count")),
            "quantum_information_entry_fixed_consistency_count": int(
                quantum_information_direct_summary.get("entry_fixed_consistency_count")
            ),
            "quantum_information_min_log10_required_thermal_ratio": float(
                quantum_information_direct_summary.get("min_log10_required_thermal_ratio")
            ),
            "quantum_information_max_log10_required_thermal_ratio": float(
                quantum_information_direct_summary.get("max_log10_required_thermal_ratio")
            ),
            "quantum_information_error_origin_status": str(
                quantum_information_error_decision.get("overall_status", "unknown")
            ),
            "quantum_information_dominant_dephasing_count": int(
                quantum_information_error_summary.get("dominant_dephasing_count")
            ),
        },
        "scope": {
            "closed_in_scope": [
                "Born detection probability p(x) propto |psi(x)|^2",
                "pointer basis grounding as detector stable modes",
                "diagonal Lueders/Kraus recovery from decoherence plus conditioning",
                "entanglement source dynamics via three-wave mixing",
                "Bell-dataset connection and integrated pair-decoherence bound",
            ],
            "minimal_connection_in_scope": [
                "quantum-information T2 / gate-fidelity / pair-depth connection from Gamma_deph",
                "dephasing-primary, amplitude-origin-fixed, depolarizing-not-primary error-channel map",
            ],
            "nonblocking_residuals": [
                "single_event_probability_foundations_not_p_specific",
                "thermodynamic_limit_irreversibility_not_p_specific",
            ],
            "out_of_scope": [
                "quantum_information_protocol_mathematics",
            ],
        },
        "routes": [asdict(row) for row in route_rows],
        "decision": {
            "born_closeout_status": born_status,
            "measurement_closeout_status": measurement_status,
            "entanglement_closeout_status": entanglement_status,
            "quantum_information_status": quantum_information_status,
            "quantum_information_protocol_status": quantum_information_protocol_status,
            "single_event_probability_status": "residual_not_p_specific" if born_closed else "scope_not_fixed",
            "irreversibility_status": "residual_not_p_specific" if measurement_closed else "scope_not_fixed",
            "overall_closeout_status": overall_status,
            "v11_plus_scope_boundary_fixed": overall_closed,
            "full_first_principles_derivation": False,
            "passes": passes,
            "remaining_blocking_items": [] if overall_closed else ["8.7.50 closeout incomplete"],
            "nonblocking_residuals": [
                "single_event_probability_foundations_not_p_specific",
                "thermodynamic_limit_irreversibility_not_p_specific",
            ],
            "out_of_scope_items": ["quantum_information_protocol_mathematics"],
            "next_required_steps": ["8.7.52.6"],
        },
    }


# Function: write the route-level closeout rows in CSV form.

def _write_cases_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("routes") if isinstance(payload.get("routes"), list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "route_id",
                "closeout_status",
                "representative_metric_name",
                "representative_metric_value",
                "support_metric_name",
                "support_metric_value",
                "residual_scope",
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
    ap = argparse.ArgumentParser(description="Freeze the v1.1+ quantum closeout boundary from the existing route artifacts.")
    ap.add_argument(
        "--born-phase-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_phase_diffusion_audit_metrics.json"),
        help="Input Born A1 phase-diffusion metrics JSON path.",
    )
    ap.add_argument(
        "--born-linear-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_linear_detector_response_audit_metrics.json"),
        help="Input Born A2 linear-response metrics JSON path.",
    )
    ap.add_argument(
        "--pointer-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_pointer_basis_grounding_audit_metrics.json"),
        help="Input measurement pointer-basis grounding metrics JSON path.",
    )
    ap.add_argument(
        "--conditioning-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_conditioning_kraus_audit_metrics.json"),
        help="Input measurement conditioning/Kraus metrics JSON path.",
    )
    ap.add_argument(
        "--stability-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_dynamic_collapse_stability_audit.json"),
        help="Input measurement stability audit JSON path.",
    )
    ap.add_argument(
        "--entanglement-source-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "entanglement_source_dynamics_three_wave_mixing_metrics.json"),
        help="Input entanglement source-dynamics metrics JSON path.",
    )
    ap.add_argument(
        "--entanglement-dataset-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "entanglement_bell_dataset_connection_audit_metrics.json"),
        help="Input entanglement dataset-connection metrics JSON path.",
    )
    ap.add_argument(
        "--quantum-information-direct-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_information_direct_p_evaluation_metrics.json"),
        help="Input quantum-information direct P-evaluation metrics JSON path.",
    )
    ap.add_argument(
        "--quantum-information-error-channel-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_information_error_channel_mapping_metrics.json"),
        help="Input quantum-information error-channel mapping metrics JSON path.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_v11_plus_scope_closeout_audit_metrics.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_v11_plus_scope_closeout_cases.csv"),
        help="Output CSV path.",
    )
    args = ap.parse_args(argv)

    born_phase_metrics_path = Path(args.born_phase_metrics)
    born_linear_metrics_path = Path(args.born_linear_metrics)
    pointer_metrics_path = Path(args.pointer_metrics)
    conditioning_metrics_path = Path(args.conditioning_metrics)
    stability_metrics_path = Path(args.stability_metrics)
    entanglement_source_metrics_path = Path(args.entanglement_source_metrics)
    entanglement_dataset_metrics_path = Path(args.entanglement_dataset_metrics)
    quantum_information_direct_metrics_path = Path(args.quantum_information_direct_metrics)
    quantum_information_error_channel_metrics_path = Path(args.quantum_information_error_channel_metrics)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    # Guard: resolve relative input/output paths against the repository root.
    if not born_phase_metrics_path.is_absolute():
        born_phase_metrics_path = (ROOT / born_phase_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not born_linear_metrics_path.is_absolute():
        born_linear_metrics_path = (ROOT / born_linear_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not pointer_metrics_path.is_absolute():
        pointer_metrics_path = (ROOT / pointer_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not conditioning_metrics_path.is_absolute():
        conditioning_metrics_path = (ROOT / conditioning_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not stability_metrics_path.is_absolute():
        stability_metrics_path = (ROOT / stability_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not entanglement_source_metrics_path.is_absolute():
        entanglement_source_metrics_path = (ROOT / entanglement_source_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not entanglement_dataset_metrics_path.is_absolute():
        entanglement_dataset_metrics_path = (ROOT / entanglement_dataset_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not quantum_information_direct_metrics_path.is_absolute():
        quantum_information_direct_metrics_path = (ROOT / quantum_information_direct_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not quantum_information_error_channel_metrics_path.is_absolute():
        quantum_information_error_channel_metrics_path = (
            ROOT / quantum_information_error_channel_metrics_path
        ).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    payload = build_payload(
        born_phase_metrics=_read_json(born_phase_metrics_path),
        born_linear_metrics=_read_json(born_linear_metrics_path),
        pointer_metrics=_read_json(pointer_metrics_path),
        conditioning_metrics=_read_json(conditioning_metrics_path),
        stability_metrics=_read_json(stability_metrics_path),
        entanglement_source_metrics=_read_json(entanglement_source_metrics_path),
        entanglement_dataset_metrics=_read_json(entanglement_dataset_metrics_path),
        quantum_information_direct_metrics=_read_json(quantum_information_direct_metrics_path),
        quantum_information_error_channel_metrics=_read_json(quantum_information_error_channel_metrics_path),
        born_phase_metrics_path=born_phase_metrics_path,
        born_linear_metrics_path=born_linear_metrics_path,
        pointer_metrics_path=pointer_metrics_path,
        conditioning_metrics_path=conditioning_metrics_path,
        stability_metrics_path=stability_metrics_path,
        entanglement_source_metrics_path=entanglement_source_metrics_path,
        entanglement_dataset_metrics_path=entanglement_dataset_metrics_path,
        quantum_information_direct_metrics_path=quantum_information_direct_metrics_path,
        quantum_information_error_channel_metrics_path=quantum_information_error_channel_metrics_path,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_cases_csv(out_csv, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_v11_plus_scope_closeout_audit",
                "phase": "8.7.50.7",
                "inputs": {
                    "born_phase_diffusion_metrics_json": _rel(born_phase_metrics_path),
                    "born_linear_detector_response_metrics_json": _rel(born_linear_metrics_path),
                    "measurement_pointer_basis_grounding_metrics_json": _rel(pointer_metrics_path),
                    "measurement_conditioning_kraus_metrics_json": _rel(conditioning_metrics_path),
                    "measurement_dynamic_collapse_stability_audit_json": _rel(stability_metrics_path),
                    "entanglement_source_dynamics_metrics_json": _rel(entanglement_source_metrics_path),
                    "entanglement_bell_dataset_connection_metrics_json": _rel(entanglement_dataset_metrics_path),
                    "quantum_information_direct_p_evaluation_metrics_json": _rel(
                        quantum_information_direct_metrics_path
                    ),
                    "quantum_information_error_channel_mapping_metrics_json": _rel(
                        quantum_information_error_channel_metrics_path
                    ),
                },
                "outputs": {
                    "quantum_v11_plus_scope_closeout_audit_metrics_json": _rel(out_json),
                    "quantum_v11_plus_scope_closeout_cases_csv": _rel(out_csv),
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
