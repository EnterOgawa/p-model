"""
quantum_information_minimal_connection_audit.py

Freeze the first-pass observed baseline for the quantum-information minimal
connection. This step does not yet claim a direct P-side prediction of platform
T2 because the platform-specific transport mapping of tau_free and chi_P is not
frozen. Instead, it converts source-backed observed T2/coherence baselines into
dephasing-rate, gate-loss, and pair-depth metrics that the later P-side mapping
must reproduce without introducing new free parameters.

Inputs:
  - data/quantum/quantum_information_minimal_connection_platforms.json

Outputs:
  - output/public/quantum/quantum_information_minimal_connection_metrics.json
  - output/public/quantum/quantum_information_minimal_connection_platforms.csv
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
INPUT_JSON = ROOT / "data" / "quantum" / "quantum_information_minimal_connection_platforms.json"
OUT_DIR = ROOT / "output" / "public" / "quantum"
OUT_JSON = OUT_DIR / "quantum_information_minimal_connection_metrics.json"
OUT_CSV = OUT_DIR / "quantum_information_minimal_connection_platforms.csv"
C_LIGHT_M_S = 299792458.0


# 関数: `_read_json` の入出力契約と処理意図を定義する。
def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。

def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


# 関数: `_observed_t2_proxy_s` の入出力契約と処理意図を定義する。

def _observed_t2_proxy_s(platform: Dict[str, Any]) -> Optional[float]:
    """Return the observed coherence-time proxy for one platform."""

    observed_t2 = platform.get("observed_t2_s")

    if observed_t2 is not None:
        return float(observed_t2)

    coherence_length = platform.get("observed_coherence_length_m")

    if coherence_length is not None:
        return float(coherence_length) / C_LIGHT_M_S

    return None


# 関数: `_omega_summary` の入出力契約と処理意図を定義する。

def _omega_summary(platform: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize the current omega_* mapping state without choosing new physics."""

    candidates = platform.get("omega_candidates") or []
    omega_values = [float(candidate["omega_star_nominal_s_inv"]) for candidate in candidates]

    if not omega_values:
        return {
            "omega_mapping_status": "missing",
            "omega_candidate_count": 0,
            "omega_star_min_s_inv": None,
            "omega_star_max_s_inv": None,
        }

    return {
        "omega_mapping_status": str(platform.get("omega_mapping_status", "unknown")),
        "omega_candidate_count": len(omega_values),
        "omega_star_min_s_inv": min(omega_values),
        "omega_star_max_s_inv": max(omega_values),
    }


# 関数: `_build_platform_row` の入出力契約と処理意図を定義する。

def _build_platform_row(platform: Dict[str, Any], threshold_floor: float) -> Dict[str, Any]:
    """Convert one source-backed platform pack entry into observed baseline metrics."""

    t2_proxy_s = _observed_t2_proxy_s(platform)
    observed_gamma = None if t2_proxy_s is None or t2_proxy_s <= 0.0 else 1.0 / t2_proxy_s
    pair_gamma = None if observed_gamma is None else 2.0 * observed_gamma

    tau_gate_2q = platform.get("tau_gate_2q_s")
    observed_2q_fidelity = platform.get("observed_gate_fidelity_2q")
    observed_2q_infidelity = None if observed_2q_fidelity is None else 1.0 - float(observed_2q_fidelity)
    gate_loss_2q = None
    gate_loss_fraction_of_observed = None
    d_max_two_qubit_layers = None
    surface_code_floor_pass = None

    if observed_gamma is not None and tau_gate_2q is not None:
        tau_gate_2q = float(tau_gate_2q)
        gate_loss_2q = observed_gamma * tau_gate_2q
        d_max_two_qubit_layers = 1.0 / (pair_gamma * tau_gate_2q) if pair_gamma is not None and pair_gamma > 0.0 else None
        surface_code_floor_pass = bool(gate_loss_2q <= threshold_floor)

        if observed_2q_infidelity is not None and observed_2q_infidelity > 0.0:
            gate_loss_fraction_of_observed = gate_loss_2q / observed_2q_infidelity

    pair_time_budget_s = None if pair_gamma is None or pair_gamma <= 0.0 else 1.0 / pair_gamma
    pair_length_budget_m = None if pair_time_budget_s is None else C_LIGHT_M_S * pair_time_budget_s
    omega_summary = _omega_summary(platform)

    return {
        "platform_id": str(platform["platform_id"]),
        "label": str(platform["label"]),
        "sources": list(platform.get("sources", [])),
        "observed_t2_proxy_s": t2_proxy_s,
        "observed_gamma_deph_s_inv": observed_gamma,
        "observed_pair_gamma_s_inv": pair_gamma,
        "tau_gate_2q_s": tau_gate_2q,
        "observed_gate_fidelity_1q": platform.get("observed_gate_fidelity_1q"),
        "observed_gate_fidelity_2q": observed_2q_fidelity,
        "observed_gate_infidelity_2q": observed_2q_infidelity,
        "two_qubit_gate_loss_from_observed_t2": gate_loss_2q,
        "gate_loss_fraction_of_observed_2q_error": gate_loss_fraction_of_observed,
        "surface_code_threshold_floor_per_gate": threshold_floor,
        "surface_code_threshold_floor_pass": surface_code_floor_pass,
        "pair_time_budget_s": pair_time_budget_s,
        "pair_length_budget_m": pair_length_budget_m,
        "d_max_two_qubit_layers": d_max_two_qubit_layers,
        "t_env_nominal_k": float(platform["t_env_nominal_k"]),
        "omega_mapping_status": omega_summary["omega_mapping_status"],
        "omega_candidate_count": omega_summary["omega_candidate_count"],
        "omega_star_min_s_inv": omega_summary["omega_star_min_s_inv"],
        "omega_star_max_s_inv": omega_summary["omega_star_max_s_inv"],
        "transport_mapping_status": str(platform.get("transport_mapping_status", "unknown")),
        "tau_free_mapping_status": str(platform.get("tau_free_mapping_status", "unknown")),
        "direct_pmodel_prediction_status": "pending_tau_free_transport_mapping",
        "notes": list(platform.get("notes", [])),
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write flattened platform rows to CSV."""

    fieldnames = [
        "platform_id",
        "label",
        "observed_t2_proxy_s",
        "observed_gamma_deph_s_inv",
        "observed_pair_gamma_s_inv",
        "tau_gate_2q_s",
        "observed_gate_fidelity_1q",
        "observed_gate_fidelity_2q",
        "observed_gate_infidelity_2q",
        "two_qubit_gate_loss_from_observed_t2",
        "gate_loss_fraction_of_observed_2q_error",
        "surface_code_threshold_floor_per_gate",
        "surface_code_threshold_floor_pass",
        "pair_time_budget_s",
        "pair_length_budget_m",
        "d_max_two_qubit_layers",
        "t_env_nominal_k",
        "omega_mapping_status",
        "omega_candidate_count",
        "omega_star_min_s_inv",
        "omega_star_max_s_inv",
        "tau_free_mapping_status",
        "direct_pmodel_prediction_status",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `_build_metrics` の入出力契約と処理意図を定義する。

def _build_metrics(seed_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the first-pass observed-baseline metrics pack."""

    threshold_floor = float(seed_pack["threshold_reference"]["surface_code_threshold_floor_per_gate"])
    rows = [_build_platform_row(platform, threshold_floor) for platform in seed_pack.get("platforms", [])]

    gate_rows = [row for row in rows if row["two_qubit_gate_loss_from_observed_t2"] is not None]
    threshold_pass_rows = [row for row in gate_rows if row["surface_code_threshold_floor_pass"]]
    omega_watch_rows = [row for row in rows if row["omega_mapping_status"] not in {"fixed_seed", "fixed_choice"}]
    transport_pending_rows = [row for row in rows if row["transport_mapping_status"] != "branch_fixed"]
    next_required_steps = ["8.7.52.4", "8.7.52.5"] if not transport_pending_rows else ["8.7.52.3", "8.7.52.4", "8.7.52.5"]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.52.2",
            "name": "Quantum-information minimal connection observed-baseline audit",
        },
        "inputs": {
            "platform_seed_pack_json": str(INPUT_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": (
            "Freeze a source-backed observed baseline for T2, dephasing-limited two-qubit gate loss, "
            "pair-decoherence depth, and photonic coherence length before the direct P-side tau_free mapping is fixed."
        ),
        "assumptions": [
            "This first pass does not introduce a platform-specific fitted tau_free or chi_P.",
            "Observed T2 or coherence length is used only to define a baseline Gamma_deph(obs) that the later direct P-side mapping must reproduce.",
            "Surface-code threshold comparison uses the primary-source floor from Fowler et al. 2013; tighter engineering windows can be layered later.",
        ],
        "formulas": {
            "observed_gamma_deph": "Gamma_deph(obs) = 1 / T2_obs",
            "two_qubit_gate_loss": "1 - F_2q(dephasing only) ~ Gamma_deph(obs) tau_gate,2q",
            "pair_decoherence": "Gamma_pair(obs) = 2 Gamma_deph(obs)",
            "pair_depth": "d_max(obs) ~ 1 / (Gamma_pair(obs) tau_gate,2q)",
            "photonic_proxy": "T2_proxy = L_coh / c for the fiber-interference branch",
        },
        "threshold_reference": seed_pack["threshold_reference"],
        "platform_rows": rows,
        "summary": {
            "platform_count": len(rows),
            "gate_capable_platform_count": len(gate_rows),
            "surface_code_threshold_pass_count": len(threshold_pass_rows),
            "omega_mapping_watch_count": len(omega_watch_rows),
            "min_two_qubit_gate_loss_from_observed_t2": min(
                (row["two_qubit_gate_loss_from_observed_t2"] for row in gate_rows),
                default=None,
            ),
            "max_two_qubit_gate_loss_from_observed_t2": max(
                (row["two_qubit_gate_loss_from_observed_t2"] for row in gate_rows),
                default=None,
            ),
            "min_d_max_two_qubit_layers": min(
                (row["d_max_two_qubit_layers"] for row in gate_rows if row["d_max_two_qubit_layers"] is not None),
                default=None,
            ),
            "max_pair_length_budget_m": max(
                (row["pair_length_budget_m"] for row in rows if row["pair_length_budget_m"] is not None),
                default=None,
            ),
        },
        "decision": {
            "source_registration_status": "complete",
            "observed_baseline_status": "fixed",
            "direct_pmodel_prediction_status": "pending_tau_free_transport_mapping",
            "quantum_information_scope_status": "minimal_connection_seeded",
            "next_required_steps": next_required_steps,
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    """Run the observed-baseline audit for the quantum-information minimal connection."""

    parser = argparse.ArgumentParser(description="Freeze the observed baseline for the quantum-information minimal connection.")
    parser.add_argument("--input", default=str(INPUT_JSON), help="Platform seed pack JSON path.")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output metrics JSON path.")
    parser.add_argument("--out-csv", default=str(OUT_CSV), help="Output metrics CSV path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_json_path = Path(args.out_json)
    out_csv_path = Path(args.out_csv)

    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()

    if not out_json_path.is_absolute():
        out_json_path = (ROOT / out_json_path).resolve()

    if not out_csv_path.is_absolute():
        out_csv_path = (ROOT / out_csv_path).resolve()

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    seed_pack = _read_json(input_path)
    metrics = _build_metrics(seed_pack)

    with out_json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(out_csv_path, metrics["platform_rows"])


if __name__ == "__main__":
    main()
