"""
quantum_information_transport_mapping_audit.py

Step 8.7.52.3:
Freeze the platform transport mapping used by the minimal quantum-information
connection. This step does not yet claim a direct P-side prediction of T2. It
fixes which Part I transport branch is used for each platform, which omega_*
carrier is the relevant free-precession bookkeeping variable, and which
source-backed operational timescale can be used as a tau_free proxy or upper
bound for the next direct-evaluation step.

Inputs:
  - data/quantum/quantum_information_minimal_connection_platforms.json

Outputs:
  - output/public/quantum/quantum_information_transport_mapping_metrics.json
  - output/public/quantum/quantum_information_transport_mapping_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = ROOT / "data" / "quantum" / "quantum_information_minimal_connection_platforms.json"
OUT_DIR = ROOT / "output" / "public" / "quantum"
OUT_JSON = OUT_DIR / "quantum_information_transport_mapping_metrics.json"
OUT_CSV = OUT_DIR / "quantum_information_transport_mapping_rows.csv"
C_LIGHT_M_S = 299792458.0


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_selected_candidate` の入出力契約と処理意図を定義する。

def _selected_candidate(platform: Dict[str, Any]) -> Dict[str, Any]:
    """Return the source-of-truth omega_* candidate selected for one platform."""

    candidates = platform.get("omega_candidates") or []
    selected_id = platform.get("omega_selected_candidate_id")

    for candidate in candidates:
        if str(candidate.get("carrier_id")) == str(selected_id):
            return candidate

    if not candidates:
        raise ValueError(f"missing omega_candidates for platform {platform.get('platform_id')}")

    return candidates[0]


# 関数: `_transport_formula` の入出力契約と処理意図を定義する。

def _transport_formula(platform: Dict[str, Any]) -> str:
    """Return the branch-specific tau_free formula used in the next step."""

    branch = str(platform.get("transport_branch", "unknown"))

    if branch == "collisional_dominant_material_bath":
        return "tau_free ≈ (A_col rho T_env^(-3/2))^(-1), with v/L_corr << Gamma_coll"

    if branch == "advective_dominant_secular_motion":
        return "tau_free ≈ L_trap / v_sec, with Gamma_coll << v_sec / L_trap"

    if branch == "advective_dominant_propagation":
        return "tau_free ≈ L_corr / c_w"

    return "undetermined"


# 関数: `_tau_free_reference` の入出力契約と処理意図を定義する。

def _tau_free_reference(platform: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Return the first source-backed tau_free proxy or operational upper bound."""

    branch = str(platform.get("transport_branch", "unknown"))
    tau_gate_2q = platform.get("tau_gate_2q_s")
    coherence_length = platform.get("observed_coherence_length_m")

    if branch == "advective_dominant_propagation" and coherence_length is not None:
        tau_free_proxy_s = float(coherence_length) / C_LIGHT_M_S
        return {
            "tau_free_proxy_s": tau_free_proxy_s,
            "tau_free_upper_bound_s": tau_free_proxy_s,
            "operational_reference_s": tau_free_proxy_s,
        }

    if tau_gate_2q is not None:
        tau_gate_value = float(tau_gate_2q)
        return {
            "tau_free_proxy_s": None,
            "tau_free_upper_bound_s": tau_gate_value,
            "operational_reference_s": tau_gate_value,
        }

    return {
        "tau_free_proxy_s": None,
        "tau_free_upper_bound_s": None,
        "operational_reference_s": None,
    }


# 関数: `_build_row` の入出力契約と処理意図を定義する。

def _build_row(platform: Dict[str, Any]) -> Dict[str, Any]:
    """Build one transport-mapping row from the platform seed pack."""

    candidate = _selected_candidate(platform)
    tau_refs = _tau_free_reference(platform)
    branch = str(platform.get("transport_branch", "unknown"))

    if branch == "collisional_dominant_material_bath":
        carrier_rationale = "The stored qubit phase is set by the transmon transition; the dense chip environment is represented by the collisional/material bath branch."
    elif branch == "advective_dominant_secular_motion":
        carrier_rationale = "The memory phase of the trapped-ion qubit is the hyperfine splitting; the optical carrier is a drive channel and is not the free-precession omega_*."
    else:
        carrier_rationale = "The photonic branch carries phase directly on the optical propagation carrier, so omega_* follows the telecom envelope carrier."

    direct_eval_status = (
        "ready_with_tau_free_proxy"
        if tau_refs["tau_free_proxy_s"] is not None
        else "ready_with_tau_free_upper_bound_only"
    )

    return {
        "platform_id": str(platform["platform_id"]),
        "label": str(platform["label"]),
        "transport_branch": branch,
        "transport_mapping_status": str(platform.get("transport_mapping_status", "unknown")),
        "omega_mapping_status": str(platform.get("omega_mapping_status", "unknown")),
        "omega_selected_candidate_id": str(platform.get("omega_selected_candidate_id", "")),
        "omega_star_selected_s_inv": float(candidate["omega_star_nominal_s_inv"]),
        "tau_free_mapping_status": str(platform.get("tau_free_mapping_status", "unknown")),
        "tau_free_formula": _transport_formula(platform),
        "tau_free_proxy_s": tau_refs["tau_free_proxy_s"],
        "tau_free_upper_bound_s": tau_refs["tau_free_upper_bound_s"],
        "operational_reference_s": tau_refs["operational_reference_s"],
        "t_env_nominal_k": float(platform["t_env_nominal_k"]),
        "direct_eval_status": direct_eval_status,
        "carrier_choice_rationale": carrier_rationale,
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write flattened transport-mapping rows to CSV."""

    fieldnames = [
        "platform_id",
        "label",
        "transport_branch",
        "transport_mapping_status",
        "omega_mapping_status",
        "omega_selected_candidate_id",
        "omega_star_selected_s_inv",
        "tau_free_mapping_status",
        "tau_free_formula",
        "tau_free_proxy_s",
        "tau_free_upper_bound_s",
        "operational_reference_s",
        "t_env_nominal_k",
        "direct_eval_status",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(seed_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the transport-mapping audit payload."""

    rows = [_build_row(platform) for platform in seed_pack.get("platforms", [])]
    ready_with_proxy = [row for row in rows if row["direct_eval_status"] == "ready_with_tau_free_proxy"]
    ready_with_bound = [row for row in rows if row["direct_eval_status"] == "ready_with_tau_free_upper_bound_only"]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.52.3",
            "name": "Quantum-information transport mapping",
        },
        "inputs": {
            "platform_seed_pack_json": str(INPUT_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": (
            "Freeze which Part I transport branch and which omega_* carrier are used for each quantum-information platform "
            "before the next direct P-side decoherence evaluation."
        ),
        "assumptions": [
            "The same Part I transport split Gamma_path = Gamma_adv + Gamma_coll is used without adding new model parameters.",
            "For gate-capable hardware, the shortest source-backed two-qubit gate time is treated as an operational upper bound on tau_free in the first direct-evaluation envelope.",
            "For the photonic branch, the observed coherence length provides a direct first-pass proxy tau_free ≈ L_coh / c_w.",
        ],
        "formulas": {
            "gamma_path_split": "Gamma_path = Gamma_adv + Gamma_coll",
            "advective_branch": "Gamma_adv ≈ v / L_corr",
            "collisional_branch": "Gamma_coll ≈ A_col rho T_env^(-3/2)",
            "photonic_proxy": "tau_free,ph ≈ L_coh / c_w",
            "gate_upper_bound": "tau_free <= tau_gate,2q is the first operational bound for Markov-compatible gate-capable platforms",
        },
        "platform_rows": rows,
        "summary": {
            "platform_count": len(rows),
            "omega_fixed_count": sum(1 for row in rows if row["omega_mapping_status"] == "fixed_choice"),
            "transport_branch_fixed_count": sum(1 for row in rows if row["transport_mapping_status"] == "branch_fixed"),
            "ready_with_tau_free_proxy_count": len(ready_with_proxy),
            "ready_with_tau_free_upper_bound_count": len(ready_with_bound),
        },
        "decision": {
            "transport_mapping_status": "branch_and_carrier_fixed",
            "new_free_parameters_introduced": False,
            "next_required_steps": ["8.7.52.4", "8.7.52.5", "8.7.52.6"],
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    """Run the transport-mapping audit for the quantum-information minimal connection."""

    parser = argparse.ArgumentParser(description="Freeze the transport mapping for the quantum-information minimal connection.")
    parser.add_argument("--input", default=str(INPUT_JSON), help="Platform seed pack JSON path.")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output metrics JSON path.")
    parser.add_argument("--out-csv", default=str(OUT_CSV), help="Output mapping CSV path.")
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
    payload = _build_payload(seed_pack)

    with out_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(out_csv_path, payload["platform_rows"])


if __name__ == "__main__":
    main()
