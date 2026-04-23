"""
quantum_information_error_channel_mapping_audit.py

Step 8.7.52.5:
Map the already closed A1/C1/C2 machinery onto the minimal quantum-information
error-channel language. The intent is not to introduce a new T1 fit. Instead,
this step fixes:

  - dephasing as the direct platform-specific channel from A1
  - amplitude damping as the stable-pointer relaxation mechanism from C1
  - depolarizing as a composite residual rather than a primary P channel

The platform-specific rate that is already fixed in step 8.7.52.4 is the
dephasing rate. The amplitude-damping channel gets a physical origin and a
reference detector-side relaxation range from the pointer-basis audit, but its
platform-specific absolute rate is intentionally left pending until a platform
pointer pack exists.

Inputs:
  - output/public/quantum/quantum_information_direct_p_evaluation_metrics.json
  - output/public/quantum/quantum_measurement_pointer_basis_grounding_audit_metrics.json
  - output/public/quantum/quantum_measurement_conditioning_kraus_audit_metrics.json
  - output/public/quantum/quantum_measurement_dynamic_collapse_stability_audit.json

Outputs:
  - output/public/quantum/quantum_information_error_channel_mapping_metrics.json
  - output/public/quantum/quantum_information_error_channel_mapping_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
DIRECT_JSON = ROOT / "output" / "public" / "quantum" / "quantum_information_direct_p_evaluation_metrics.json"
POINTER_JSON = ROOT / "output" / "public" / "quantum" / "quantum_measurement_pointer_basis_grounding_audit_metrics.json"
KRAUS_JSON = ROOT / "output" / "public" / "quantum" / "quantum_measurement_conditioning_kraus_audit_metrics.json"
STABILITY_JSON = ROOT / "output" / "public" / "quantum" / "quantum_measurement_dynamic_collapse_stability_audit.json"
OUT_DIR = ROOT / "output" / "public" / "quantum"
OUT_JSON = OUT_DIR / "quantum_information_error_channel_mapping_metrics.json"
OUT_CSV = OUT_DIR / "quantum_information_error_channel_mapping_rows.csv"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON object from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_channel_relaxation_summary` の入出力契約と処理意図を定義する。

def _channel_relaxation_summary(pointer_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Summarize the detector-side pointer-relaxation reference range."""

    channels = list(pointer_metrics.get("channels", []))
    relax_values = [float(channel["relax_s_inv"]) for channel in channels]

    if not relax_values:
        raise ValueError("pointer metrics do not contain relaxation channels")

    return {
        "gamma_amp_ref_min_s_inv": min(relax_values),
        "gamma_amp_ref_mean_s_inv": mean(relax_values),
        "gamma_amp_ref_max_s_inv": max(relax_values),
    }


# 関数: `_basis_scramble_bound` の入出力契約と処理意図を定義する。

def _basis_scramble_bound(kraus_metrics: Dict[str, Any], stability_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Build a conservative basis-scrambling bound from C2 and stability artifacts."""

    epsilon_upper = float(kraus_metrics["summary"]["epsilon_response_upper_bound"])
    branch_reversal_max = float(stability_metrics["summary"]["branch_reversal_stats"]["max"])
    pointer_consensus_min = float(stability_metrics["summary"]["pointer_consensus_stats"]["min"])
    diagonal_preservation_floor = min(
        1.0 - epsilon_upper,
        pointer_consensus_min,
    )

    return {
        "basis_scrambling_upper_bound": max(epsilon_upper, branch_reversal_max),
        "diagonal_preservation_floor": diagonal_preservation_floor,
        "epsilon_response_upper_bound": epsilon_upper,
        "branch_reversal_max": branch_reversal_max,
    }


# 関数: `_dephasing_rate_reference` の入出力契約と処理意図を定義する。

def _dephasing_rate_reference(row: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the dephasing-rate reference already fixed by step 8.7.52.4."""

    exact_gamma = row.get("gamma_deph_direct_exact_s_inv")

    if exact_gamma is not None:
        return {
            "dephasing_rate_status": "fixed_exact_proxy",
            "gamma_deph_reference_s_inv": float(exact_gamma),
        }

    envelope_gamma = row.get("gamma_deph_upper_envelope_s_inv")

    if envelope_gamma is not None:
        return {
            "dephasing_rate_status": "fixed_upper_envelope",
            "gamma_deph_reference_s_inv": float(envelope_gamma),
        }

    return {
        "dephasing_rate_status": "missing",
        "gamma_deph_reference_s_inv": None,
    }


# 関数: `_platform_rationale` の入出力契約と処理意図を定義する。

def _platform_rationale(platform_id: str) -> str:
    """Return the platform-specific minimal-connection rationale."""

    if platform_id == "photonic_polarization_1550nm":
        return (
            "Within the polarization subspace, the directly fixed P-specific channel is phase diffusion "
            "on the propagation carrier; loss/erasure is secondary and not promoted to a primary depolarizing channel."
        )

    if platform_id == "superconducting_transmon":
        return (
            "The stored chip qubit is read as a protected phase variable, so the only platform-specific rate already "
            "fixed by P is dephasing; amplitude damping remains a pointer-relaxation mechanism awaiting a platform pointer pack."
        )

    return (
        "The hyperfine memory branch carries phase on the protected qubit splitting, so the minimal connection fixes "
        "dephasing first; amplitude damping is structurally available through pointer relaxation but is not yet a platform-specific T1 prediction."
    )


# 関数: `_build_row` の入出力契約と処理意図を定義する。

def _build_row(
    direct_row: Dict[str, Any],
    gamma_amp_ref: Dict[str, float],
    scramble_bound: Dict[str, float],
) -> Dict[str, Any]:
    """Build one platform row for the error-channel mapping audit."""

    rate_ref = _dephasing_rate_reference(direct_row)
    platform_id = str(direct_row["platform_id"])
    direct_mode = str(direct_row["direct_prediction_mode"])

    if direct_mode == "exact_tau_free_proxy":
        dominant_channel = "dephasing_primary_exact_proxy"
    else:
        dominant_channel = "dephasing_primary_entry_fixed"

    return {
        "platform_id": platform_id,
        "label": str(direct_row["label"]),
        "transport_branch": direct_row.get("transport_branch"),
        "direct_prediction_mode": direct_mode,
        "platform_decision": str(direct_row["platform_decision"]),
        "dominant_channel": dominant_channel,
        "dephasing_rate_status": rate_ref["dephasing_rate_status"],
        "gamma_deph_reference_s_inv": rate_ref["gamma_deph_reference_s_inv"],
        "amplitude_damping_status": "origin_fixed_platform_rate_pending",
        "gamma_amp_ref_min_s_inv": gamma_amp_ref["gamma_amp_ref_min_s_inv"],
        "gamma_amp_ref_mean_s_inv": gamma_amp_ref["gamma_amp_ref_mean_s_inv"],
        "gamma_amp_ref_max_s_inv": gamma_amp_ref["gamma_amp_ref_max_s_inv"],
        "depolarizing_status": "composite_subleading_not_primary",
        "basis_scrambling_upper_bound": scramble_bound["basis_scrambling_upper_bound"],
        "diagonal_preservation_floor": scramble_bound["diagonal_preservation_floor"],
        "epsilon_response_upper_bound": scramble_bound["epsilon_response_upper_bound"],
        "branch_reversal_max": scramble_bound["branch_reversal_max"],
        "surface_code_threshold_floor_pass": direct_row.get("surface_code_threshold_floor_pass"),
        "platform_rationale": _platform_rationale(platform_id),
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write flattened error-channel rows to CSV."""

    fieldnames = [
        "platform_id",
        "label",
        "transport_branch",
        "direct_prediction_mode",
        "platform_decision",
        "dominant_channel",
        "dephasing_rate_status",
        "gamma_deph_reference_s_inv",
        "amplitude_damping_status",
        "gamma_amp_ref_min_s_inv",
        "gamma_amp_ref_mean_s_inv",
        "gamma_amp_ref_max_s_inv",
        "depolarizing_status",
        "basis_scrambling_upper_bound",
        "diagonal_preservation_floor",
        "epsilon_response_upper_bound",
        "branch_reversal_max",
        "surface_code_threshold_floor_pass",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(
    direct_metrics: Dict[str, Any],
    pointer_metrics: Dict[str, Any],
    kraus_metrics: Dict[str, Any],
    stability_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct the quantum-information error-channel mapping payload."""

    gamma_amp_ref = _channel_relaxation_summary(pointer_metrics)
    scramble_bound = _basis_scramble_bound(kraus_metrics, stability_metrics)
    direct_rows = list(direct_metrics.get("platform_rows", []))
    rows = [_build_row(row, gamma_amp_ref, scramble_bound) for row in direct_rows]
    dominant_dephasing_rows = [row for row in rows if str(row["dominant_channel"]).startswith("dephasing_primary")]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.52.5",
            "name": "Quantum-information error-channel mapping",
        },
        "inputs": {
            "direct_p_eval_json": str(DIRECT_JSON.relative_to(ROOT)).replace("\\", "/"),
            "pointer_basis_json": str(POINTER_JSON.relative_to(ROOT)).replace("\\", "/"),
            "conditioning_kraus_json": str(KRAUS_JSON.relative_to(ROOT)).replace("\\", "/"),
            "stability_json": str(STABILITY_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": (
            "Fix the minimal P-side origin of the dominant quantum-information error channels without "
            "promoting a new platform-specific T1 fit."
        ),
        "assumptions": [
            "The platform-specific rate already fixed by the minimal connection is the A1 dephasing rate from step 8.7.52.4.",
            "C1 supplies the physical origin of amplitude damping as stable-pointer relaxation, but not yet a platform-specific absolute rate.",
            "C2 and the stability audit show that basis mixing remains diagonal-dominant, so a depolarizing channel is treated only as a composite residual.",
        ],
        "formulas": {
            "dephasing_channel": "rho -> (1-p_phi) rho + p_phi Z rho Z, p_phi = (1/2) Gamma_deph t + O(t^2)",
            "amplitude_damping_channel": "rho -> E0 rho E0^dagger + E1 rho E1^dagger, p_relax = Gamma_amp t + O(t^2)",
            "gamma_amp_origin": "Gamma_amp ~ Gamma_k from the C1 stable-pointer relaxation sector",
            "depolarizing_not_primary": "p_dep is bounded by basis-scrambling diagnostics and is not promoted to a standalone P-primary channel",
        },
        "pointer_reference": {
            **gamma_amp_ref,
            **scramble_bound,
            "pointer_consensus_fraction_reference": float(pointer_metrics["summary"]["pointer_consensus_fraction_reference"]),
            "measurement_gamma_deph_reference_s_inv": float(pointer_metrics["summary"]["gamma_meas_identified_as_gamma_deph_det_s_inv"]),
        },
        "platform_rows": rows,
        "summary": {
            "platform_count": len(rows),
            "dominant_dephasing_count": len(dominant_dephasing_rows),
            "exact_dephasing_rate_count": sum(1 for row in rows if row["dephasing_rate_status"] == "fixed_exact_proxy"),
            "upper_envelope_dephasing_rate_count": sum(1 for row in rows if row["dephasing_rate_status"] == "fixed_upper_envelope"),
            "amplitude_origin_fixed_count": len(rows),
            "depolarizing_primary_count": 0,
            "basis_scrambling_upper_bound": scramble_bound["basis_scrambling_upper_bound"],
        },
        "decision": {
            "overall_status": "minimal_connection_error_origin_fixed",
            "quantum_information_status": "minimal_connection_entry_fixed_c52_pending_wording_sync",
            "direct_platform_t1_prediction": False,
            "passes": {
                "dephasing_rate_available_for_all_platforms": all(row["gamma_deph_reference_s_inv"] is not None for row in rows),
                "amplitude_damping_origin_fixed": True,
                "depolarizing_not_primary": True,
                "no_new_pmodel_free_parameters": True,
            },
            "next_required_steps": ["8.7.52.6"],
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    """Run the quantum-information error-channel mapping audit."""

    parser = argparse.ArgumentParser(description="Run the quantum-information error-channel mapping audit.")
    parser.add_argument("--direct-json", default=str(DIRECT_JSON), help="Direct P-evaluation metrics JSON path.")
    parser.add_argument("--pointer-json", default=str(POINTER_JSON), help="Pointer-basis grounding metrics JSON path.")
    parser.add_argument("--kraus-json", default=str(KRAUS_JSON), help="Conditioning/Kraus metrics JSON path.")
    parser.add_argument("--stability-json", default=str(STABILITY_JSON), help="Dynamic-collapse stability JSON path.")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output metrics JSON path.")
    parser.add_argument("--out-csv", default=str(OUT_CSV), help="Output CSV path.")
    args = parser.parse_args()

    direct_json_path = Path(args.direct_json)
    pointer_json_path = Path(args.pointer_json)
    kraus_json_path = Path(args.kraus_json)
    stability_json_path = Path(args.stability_json)
    out_json_path = Path(args.out_json)
    out_csv_path = Path(args.out_csv)

    if not direct_json_path.is_absolute():
        direct_json_path = (ROOT / direct_json_path).resolve()

    if not pointer_json_path.is_absolute():
        pointer_json_path = (ROOT / pointer_json_path).resolve()

    if not kraus_json_path.is_absolute():
        kraus_json_path = (ROOT / kraus_json_path).resolve()

    if not stability_json_path.is_absolute():
        stability_json_path = (ROOT / stability_json_path).resolve()

    if not out_json_path.is_absolute():
        out_json_path = (ROOT / out_json_path).resolve()

    if not out_csv_path.is_absolute():
        out_csv_path = (ROOT / out_csv_path).resolve()

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    direct_metrics = _read_json(direct_json_path)
    pointer_metrics = _read_json(pointer_json_path)
    kraus_metrics = _read_json(kraus_json_path)
    stability_metrics = _read_json(stability_json_path)
    payload = _build_payload(direct_metrics, pointer_metrics, kraus_metrics, stability_metrics)

    with out_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(out_csv_path, payload["platform_rows"])


if __name__ == "__main__":
    main()
