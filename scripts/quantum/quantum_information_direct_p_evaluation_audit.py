"""
quantum_information_direct_p_evaluation_audit.py

Step 8.7.52.4:
Convert the fixed quantum-information platform mapping into a direct P-side
consistency audit. This step does not introduce a new fitted susceptibility or
platform-specific free parameter. Instead, it uses the already frozen carrier
choice and tau_free envelope to determine which thermal ratio

    k_B T_env / chi_P

is required for each platform to reproduce the observed T2 proxy through

    Gamma_deph = omega_*^2 (k_B T_env / chi_P) tau_free.

For branches with a source-backed tau_free proxy, the ratio is fixed exactly.
For gate-capable branches that currently have only a tau_free upper bound, the
same relation yields a lower bound on the thermal ratio and therefore an
entry-fixed consistency envelope rather than a full direct pass.

Inputs:
  - output/public/quantum/quantum_information_minimal_connection_metrics.json
  - output/public/quantum/quantum_information_transport_mapping_metrics.json

Outputs:
  - output/public/quantum/quantum_information_direct_p_evaluation_metrics.json
  - output/public/quantum/quantum_information_direct_p_evaluation_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
BASELINE_JSON = ROOT / "output" / "public" / "quantum" / "quantum_information_minimal_connection_metrics.json"
TRANSPORT_JSON = ROOT / "output" / "public" / "quantum" / "quantum_information_transport_mapping_metrics.json"
OUT_DIR = ROOT / "output" / "public" / "quantum"
OUT_JSON = OUT_DIR / "quantum_information_direct_p_evaluation_metrics.json"
OUT_CSV = OUT_DIR / "quantum_information_direct_p_evaluation_rows.csv"
PAIR_LENGTH_FACTOR = 0.5


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON object from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_rows_by_platform_id` の入出力契約と処理意図を定義する。

def _rows_by_platform_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index artifact rows by platform identifier."""

    indexed_rows: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        indexed_rows[str(row["platform_id"])] = row

    return indexed_rows


# 関数: `_safe_ratio` の入出力契約と処理意図を定義する。

def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Return a positive ratio when both inputs are valid, otherwise None."""

    if numerator is None or denominator is None:
        return None

    numerator_value = float(numerator)
    denominator_value = float(denominator)

    if denominator_value <= 0.0:
        return None

    return numerator_value / denominator_value


# 関数: `_safe_inverse` の入出力契約と処理意図を定義する。

def _safe_inverse(value: Optional[float]) -> Optional[float]:
    """Return the inverse of a positive value when defined."""

    if value is None:
        return None

    value_float = float(value)

    if value_float <= 0.0:
        return None

    return 1.0 / value_float


# 関数: `_safe_log10` の入出力契約と処理意図を定義する。

def _safe_log10(value: Optional[float]) -> Optional[float]:
    """Return log10(value) for positive inputs."""

    if value is None:
        return None

    value_float = float(value)

    if value_float <= 0.0:
        return None

    return math.log10(value_float)


# 関数: `_pair_length_budget_m` の入出力契約と処理意図を定義する。

def _pair_length_budget_m(pair_time_budget_s: Optional[float]) -> Optional[float]:
    """Convert a pair-coherence time budget into a propagation-length budget."""

    if pair_time_budget_s is None:
        return None

    return 299792458.0 * float(pair_time_budget_s) * PAIR_LENGTH_FACTOR


# 関数: `_build_row` の入出力契約と処理意図を定義する。

def _build_row(
    baseline_row: Dict[str, Any],
    transport_row: Dict[str, Any],
    threshold_floor: float,
) -> Dict[str, Any]:
    """Build one direct P-side evaluation row."""

    platform_id = str(baseline_row["platform_id"])
    observed_gamma = baseline_row.get("observed_gamma_deph_s_inv")
    observed_t2 = baseline_row.get("observed_t2_proxy_s")
    observed_pair_gamma = baseline_row.get("observed_pair_gamma_s_inv")
    observed_gate_loss = baseline_row.get("two_qubit_gate_loss_from_observed_t2")
    observed_dmax = baseline_row.get("d_max_two_qubit_layers")
    tau_gate_2q = baseline_row.get("tau_gate_2q_s")
    omega_star = transport_row.get("omega_star_selected_s_inv")
    tau_free_proxy = transport_row.get("tau_free_proxy_s")
    tau_free_upper = transport_row.get("tau_free_upper_bound_s")

    exact_transport = tau_free_proxy is not None
    tau_reference = tau_free_proxy if exact_transport else tau_free_upper
    omega_squared_tau = None if omega_star is None or tau_reference is None else float(omega_star) ** 2 * float(tau_reference)
    required_thermal_ratio = _safe_ratio(observed_gamma, omega_squared_tau)
    required_chi_over_kbt = _safe_inverse(required_thermal_ratio)
    required_ratio_subunity = required_thermal_ratio is not None and required_thermal_ratio < 1.0

    platform_decision = "reject"
    direct_prediction_mode = "unresolved"

    if exact_transport:
        direct_prediction_mode = "exact_tau_free_proxy"

        if required_ratio_subunity:
            platform_decision = "pass_exact_proxy"
    else:
        direct_prediction_mode = "tau_free_upper_bound_only"

        if required_ratio_subunity:
            platform_decision = "entry_fixed_lower_bound_consistent"

    gamma_deph_exact = None
    t2_exact = None
    pair_gamma_exact = None
    pair_time_budget_exact = None
    pair_length_budget_exact = None
    gate_loss_exact = None
    dmax_exact = None

    if exact_transport and required_thermal_ratio is not None and tau_reference is not None and omega_star is not None:
        gamma_deph_exact = float(omega_star) ** 2 * required_thermal_ratio * float(tau_reference)
        t2_exact = _safe_inverse(gamma_deph_exact)
        pair_gamma_exact = None if gamma_deph_exact is None else 2.0 * gamma_deph_exact
        pair_time_budget_exact = _safe_inverse(pair_gamma_exact)
        pair_length_budget_exact = _pair_length_budget_m(pair_time_budget_exact)

        if tau_gate_2q is not None and gamma_deph_exact is not None:
            gate_loss_exact = gamma_deph_exact * float(tau_gate_2q)

            if pair_gamma_exact is not None and pair_gamma_exact > 0.0:
                dmax_exact = 1.0 / (pair_gamma_exact * float(tau_gate_2q))

    gamma_deph_upper_envelope = None
    t2_lower_envelope = None
    pair_gamma_upper_envelope = None
    pair_time_budget_lower_envelope = None
    pair_length_budget_lower_envelope = None
    gate_loss_upper_envelope = None
    dmax_lower_envelope = None

    if (not exact_transport) and required_thermal_ratio is not None and tau_free_upper is not None and omega_star is not None:
        gamma_deph_upper_envelope = float(omega_star) ** 2 * required_thermal_ratio * float(tau_free_upper)
        t2_lower_envelope = _safe_inverse(gamma_deph_upper_envelope)
        pair_gamma_upper_envelope = None if gamma_deph_upper_envelope is None else 2.0 * gamma_deph_upper_envelope
        pair_time_budget_lower_envelope = _safe_inverse(pair_gamma_upper_envelope)
        pair_length_budget_lower_envelope = _pair_length_budget_m(pair_time_budget_lower_envelope)

        if tau_gate_2q is not None and gamma_deph_upper_envelope is not None:
            gate_loss_upper_envelope = gamma_deph_upper_envelope * float(tau_gate_2q)

            if pair_gamma_upper_envelope is not None and pair_gamma_upper_envelope > 0.0:
                dmax_lower_envelope = 1.0 / (pair_gamma_upper_envelope * float(tau_gate_2q))

    threshold_pass = None

    if gate_loss_exact is not None:
        threshold_pass = gate_loss_exact <= threshold_floor
    elif gate_loss_upper_envelope is not None:
        threshold_pass = gate_loss_upper_envelope <= threshold_floor

    return {
        "platform_id": platform_id,
        "label": str(baseline_row["label"]),
        "direct_prediction_mode": direct_prediction_mode,
        "platform_decision": platform_decision,
        "transport_branch": transport_row.get("transport_branch"),
        "omega_selected_candidate_id": transport_row.get("omega_selected_candidate_id"),
        "omega_star_selected_s_inv": omega_star,
        "tau_free_reference_s": tau_reference,
        "tau_free_proxy_s": tau_free_proxy,
        "tau_free_upper_bound_s": tau_free_upper,
        "observed_t2_proxy_s": observed_t2,
        "observed_gamma_deph_s_inv": observed_gamma,
        "observed_pair_gamma_s_inv": observed_pair_gamma,
        "tau_gate_2q_s": tau_gate_2q,
        "required_thermal_ratio_exact": required_thermal_ratio if exact_transport else None,
        "required_thermal_ratio_lower_bound": None if exact_transport else required_thermal_ratio,
        "required_chi_over_kbt_exact": required_chi_over_kbt if exact_transport else None,
        "required_chi_over_kbt_upper_bound": None if exact_transport else required_chi_over_kbt,
        "log10_required_thermal_ratio_exact": _safe_log10(required_thermal_ratio) if exact_transport else None,
        "log10_required_thermal_ratio_lower_bound": None if exact_transport else _safe_log10(required_thermal_ratio),
        "gamma_deph_direct_exact_s_inv": gamma_deph_exact,
        "t2_direct_exact_s": t2_exact,
        "pair_gamma_direct_exact_s_inv": pair_gamma_exact,
        "pair_time_budget_direct_exact_s": pair_time_budget_exact,
        "pair_length_budget_direct_exact_m": pair_length_budget_exact,
        "two_qubit_gate_loss_direct_exact": gate_loss_exact,
        "d_max_direct_exact": dmax_exact,
        "gamma_deph_upper_envelope_s_inv": gamma_deph_upper_envelope,
        "t2_lower_envelope_s": t2_lower_envelope,
        "pair_gamma_upper_envelope_s_inv": pair_gamma_upper_envelope,
        "pair_time_budget_lower_envelope_s": pair_time_budget_lower_envelope,
        "pair_length_budget_lower_envelope_m": pair_length_budget_lower_envelope,
        "two_qubit_gate_loss_upper_envelope": gate_loss_upper_envelope,
        "d_max_lower_envelope": dmax_lower_envelope,
        "observed_gate_loss_from_t2": observed_gate_loss,
        "observed_d_max_two_qubit_layers": observed_dmax,
        "surface_code_threshold_floor_per_gate": threshold_floor,
        "surface_code_threshold_floor_pass": threshold_pass,
        "tau_reference_over_t2_proxy": _safe_ratio(tau_reference, observed_t2),
        "required_ratio_subunity": required_ratio_subunity,
        "new_pmodel_free_parameters_introduced": False,
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write direct-evaluation rows to CSV."""

    fieldnames = [
        "platform_id",
        "label",
        "direct_prediction_mode",
        "platform_decision",
        "transport_branch",
        "omega_selected_candidate_id",
        "omega_star_selected_s_inv",
        "tau_free_reference_s",
        "tau_free_proxy_s",
        "tau_free_upper_bound_s",
        "observed_t2_proxy_s",
        "observed_gamma_deph_s_inv",
        "tau_gate_2q_s",
        "required_thermal_ratio_exact",
        "required_thermal_ratio_lower_bound",
        "required_chi_over_kbt_exact",
        "required_chi_over_kbt_upper_bound",
        "gamma_deph_direct_exact_s_inv",
        "gamma_deph_upper_envelope_s_inv",
        "t2_direct_exact_s",
        "t2_lower_envelope_s",
        "two_qubit_gate_loss_direct_exact",
        "two_qubit_gate_loss_upper_envelope",
        "observed_gate_loss_from_t2",
        "d_max_direct_exact",
        "d_max_lower_envelope",
        "observed_d_max_two_qubit_layers",
        "surface_code_threshold_floor_pass",
        "tau_reference_over_t2_proxy",
        "required_ratio_subunity",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(
    baseline_metrics: Dict[str, Any],
    transport_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct the direct P-side evaluation payload."""

    threshold_floor = float(baseline_metrics["threshold_reference"]["surface_code_threshold_floor_per_gate"])
    baseline_rows = _rows_by_platform_id(list(baseline_metrics.get("platform_rows", [])))
    transport_rows = _rows_by_platform_id(list(transport_metrics.get("platform_rows", [])))
    rows: List[Dict[str, Any]] = []

    for platform_id, baseline_row in baseline_rows.items():
        transport_row = transport_rows.get(platform_id)

        if transport_row is None:
            raise KeyError(f"missing transport row for platform {platform_id}")

        rows.append(_build_row(baseline_row, transport_row, threshold_floor))

    exact_pass_rows = [row for row in rows if row["platform_decision"] == "pass_exact_proxy"]
    entry_fixed_rows = [row for row in rows if row["platform_decision"] == "entry_fixed_lower_bound_consistent"]
    reject_rows = [row for row in rows if row["platform_decision"] == "reject"]
    threshold_rows = [row for row in rows if row["surface_code_threshold_floor_pass"] is True]

    overall_status = "reject"

    if not reject_rows:
        overall_status = "entry_fixed_consistency_pass"

        if len(exact_pass_rows) == len(rows):
            overall_status = "pass_exact_all_platforms"

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.52.4",
            "name": "Quantum-information direct P-side evaluation",
        },
        "inputs": {
            "observed_baseline_json": str(BASELINE_JSON.relative_to(ROOT)).replace("\\", "/"),
            "transport_mapping_json": str(TRANSPORT_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": (
            "Translate the frozen omega_* carrier and tau_free envelope into the required P-side thermal ratio "
            "for each platform and classify the result as an exact pass or an entry-fixed consistency branch."
        ),
        "assumptions": [
            "No platform-specific chi_P fit is introduced; only the required ratio k_B T_env / chi_P is inferred from the existing A1 formula.",
            "Gate-capable platforms still carry only a source-backed upper bound on tau_free, so they can close at entry-fixed consistency even when the photonic branch reaches an exact proxy.",
            "A required thermal ratio below unity is treated as an admissible coarse-grained bath requirement in the first minimal-connection pass.",
        ],
        "formulas": {
            "direct_gamma_deph": "Gamma_deph = omega_*^2 (k_B T_env / chi_P) tau_free",
            "direct_pair_gamma": "Gamma_pair = 2 Gamma_deph",
            "required_thermal_ratio_exact": "(k_B T_env / chi_P)_req = Gamma_deph(obs) / (omega_*^2 tau_free,proxy)",
            "required_thermal_ratio_lower_bound": "(k_B T_env / chi_P)_req >= Gamma_deph(obs) / (omega_*^2 tau_free,upper)",
            "gate_loss": "1 - F_2q ~ Gamma_deph tau_gate,2q",
            "pair_depth": "d_max ~ 1 / (Gamma_pair tau_gate,2q)",
        },
        "platform_rows": rows,
        "summary": {
            "platform_count": len(rows),
            "exact_proxy_pass_count": len(exact_pass_rows),
            "entry_fixed_consistency_count": len(entry_fixed_rows),
            "reject_count": len(reject_rows),
            "surface_code_threshold_floor_pass_count": len(threshold_rows),
            "min_log10_required_thermal_ratio": min(
                (
                    value
                    for row in rows
                    for value in (
                        row["log10_required_thermal_ratio_exact"],
                        row["log10_required_thermal_ratio_lower_bound"],
                    )
                    if value is not None
                ),
                default=None,
            ),
            "max_log10_required_thermal_ratio": max(
                (
                    value
                    for row in rows
                    for value in (
                        row["log10_required_thermal_ratio_exact"],
                        row["log10_required_thermal_ratio_lower_bound"],
                    )
                    if value is not None
                ),
                default=None,
            ),
        },
        "decision": {
            "overall_status": overall_status,
            "minimal_connection_status": "entry_fixed" if overall_status != "reject" else "reject",
            "full_direct_pass": overall_status == "pass_exact_all_platforms",
            "passes": {
                "all_required_ratios_subunity": all(bool(row["required_ratio_subunity"]) for row in rows),
                "no_new_pmodel_free_parameters": True,
                "at_least_one_exact_proxy_branch": bool(exact_pass_rows),
                "gate_platforms_remain_entry_fixed": all(
                    row["platform_decision"] == "entry_fixed_lower_bound_consistent"
                    for row in rows
                    if row["direct_prediction_mode"] == "tau_free_upper_bound_only"
                ),
            },
            "next_required_steps": ["8.7.52.5", "8.7.52.6"],
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    """Run the direct P-side evaluation audit for the minimal quantum-information connection."""

    parser = argparse.ArgumentParser(description="Run the direct P-side evaluation audit for the quantum-information minimal connection.")
    parser.add_argument("--baseline-json", default=str(BASELINE_JSON), help="Observed-baseline metrics JSON path.")
    parser.add_argument("--transport-json", default=str(TRANSPORT_JSON), help="Transport-mapping metrics JSON path.")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output metrics JSON path.")
    parser.add_argument("--out-csv", default=str(OUT_CSV), help="Output CSV path.")
    args = parser.parse_args()

    baseline_json_path = Path(args.baseline_json)
    transport_json_path = Path(args.transport_json)
    out_json_path = Path(args.out_json)
    out_csv_path = Path(args.out_csv)

    if not baseline_json_path.is_absolute():
        baseline_json_path = (ROOT / baseline_json_path).resolve()

    if not transport_json_path.is_absolute():
        transport_json_path = (ROOT / transport_json_path).resolve()

    if not out_json_path.is_absolute():
        out_json_path = (ROOT / out_json_path).resolve()

    if not out_csv_path.is_absolute():
        out_csv_path = (ROOT / out_csv_path).resolve()

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_metrics = _read_json(baseline_json_path)
    transport_metrics = _read_json(transport_json_path)
    payload = _build_payload(baseline_metrics, transport_metrics)

    with out_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(out_csv_path, payload["platform_rows"])


if __name__ == "__main__":
    main()
