#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_part4_closeout_sync_pack.py

Step 8.7.49.7:
Sync the Part IV verification-materials side to the v1.1+ quantum closeout
boundary without regenerating PDFs.

Inputs:
  - output/public/quantum/quantum_v11_plus_scope_closeout_audit_metrics.json
  - output/public/quantum/derivation_observable_chain_lock_audit_watch_policy.json
  - output/public/quantum/derivation_parameter_falsification_pack.json
  - output/public/quantum/quantum_connection_born_ab_gate.json

Outputs:
  - output/public/quantum/quantum_part4_closeout_sync_pack.json
  - output/public/quantum/quantum_part4_closeout_sync_rows.csv

Assumptions:
  - The v1.1+ closeout artifact is already the canonical statement for the
    Born / measurement / entanglement boundary.
  - The legacy Part IV chain artifacts remain valid operational guards, but
    Bell-pairing-only shared-KPI rejects are demoted to watch by the frozen
    watch policy.
  - No new free parameters are introduced by this sync pack.
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


# Class: store one Part IV sync row for CSV export.
@dataclass(frozen=True)
class SyncRow:
    row_id: str
    status: str
    gate_snapshot: str
    closeout_status: str
    representative_metric_name: str
    representative_metric_value: float
    note: str


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

    if isinstance(value, dict):
        return value

    return {}


# Function: build the CSV rows that summarize the Part IV sync state.

def _rows(
    *,
    closeout_summary: Dict[str, Any],
    closeout_decision: Dict[str, Any],
    watch_decision: Dict[str, Any],
) -> List[SyncRow]:
    born_gate_snapshot = f"{watch_decision.get('route_a_gate', 'unknown')}/{watch_decision.get('transition', 'unknown')}"
    return [
        SyncRow(
            row_id="born_closeout",
            status=str(closeout_decision.get("born_closeout_status", "unknown")),
            gate_snapshot=born_gate_snapshot,
            closeout_status=str(closeout_decision.get("born_closeout_status", "unknown")),
            representative_metric_name="born_max_flat_field_frequency_error",
            representative_metric_value=float(closeout_summary.get("born_max_flat_field_frequency_error")),
            note="Born detection probability is fixed by the P-specific route; the only remaining residual is single-event probability foundations, not a P-specific failure.",
        ),
        SyncRow(
            row_id="measurement_closeout",
            status=str(closeout_decision.get("measurement_closeout_status", "unknown")),
            gate_snapshot="pointer_basis + conditioning",
            closeout_status=str(closeout_decision.get("measurement_closeout_status", "unknown")),
            representative_metric_name="measurement_tau50_over_tauD",
            representative_metric_value=float(closeout_summary.get("measurement_tau50_over_tauD")),
            note="Measurement update shape is fixed effectively; irreversibility remains a non-P-specific residual.",
        ),
        SyncRow(
            row_id="entanglement_closeout",
            status=str(closeout_decision.get("entanglement_closeout_status", "unknown")),
            gate_snapshot="selection_watch_retained",
            closeout_status=str(closeout_decision.get("entanglement_closeout_status", "unknown")),
            representative_metric_name="entanglement_min_chsh_visibility_proxy",
            representative_metric_value=float(closeout_summary.get("entanglement_min_chsh_visibility_proxy")),
            note="Bell dataset connection and pair-decoherence bound are closed while keeping selection watch nonblocking.",
        ),
        SyncRow(
            row_id="quantum_information_closeout",
            status=str(closeout_decision.get("quantum_information_status", "unknown")),
            gate_snapshot="minimal_connection_entry_fixed",
            closeout_status=str(closeout_decision.get("quantum_information_status", "unknown")),
            representative_metric_name="quantum_information_min_log10_required_thermal_ratio",
            representative_metric_value=float(closeout_summary.get("quantum_information_min_log10_required_thermal_ratio")),
            note="Quantum information is promoted from full scope-out to minimal connection: dephasing is primary, amplitude-damping origin is fixed, and depolarizing remains subleading.",
        ),
        SyncRow(
            row_id="part4_sync_policy",
            status="pass" if bool(closeout_decision.get("v11_plus_scope_boundary_fixed")) else "reject",
            gate_snapshot=born_gate_snapshot,
            closeout_status=str(closeout_decision.get("overall_closeout_status", "unknown")),
            representative_metric_name="measurement_branch_reversal_max",
            representative_metric_value=float(closeout_summary.get("measurement_branch_reversal_max")),
            note="Part IV reads legacy chain guards through the frozen watch-policy demotion for Bell-pairing-only shared-KPI rejects.",
        ),
    ]


# Function: combine the closeout artifact with the Part IV legacy gates.

def build_payload(
    *,
    closeout_metrics: Dict[str, Any],
    chain_watch_policy_metrics: Dict[str, Any],
    derivation_pack_metrics: Dict[str, Any],
    born_ab_metrics: Dict[str, Any],
    closeout_metrics_path: Path,
    chain_watch_policy_metrics_path: Path,
    derivation_pack_metrics_path: Path,
    born_ab_metrics_path: Path,
) -> Dict[str, Any]:
    closeout_summary = _as_dict(closeout_metrics, "summary")
    closeout_decision = _as_dict(closeout_metrics, "decision")
    closeout_passes = _as_dict(closeout_decision, "passes")
    chain_watch_decision = _as_dict(chain_watch_policy_metrics, "decision")
    chain_watch_policy_result = _as_dict(chain_watch_decision, "shared_gate_policy_result")
    derivation_pack_decision = _as_dict(derivation_pack_metrics, "decision")
    born_ab_decision = _as_dict(born_ab_metrics, "decision")

    remaining_blocking_items = closeout_decision.get("remaining_blocking_items")

    if not isinstance(remaining_blocking_items, list):
        remaining_blocking_items = []

    out_of_scope_items = closeout_decision.get("out_of_scope_items")

    if not isinstance(out_of_scope_items, list):
        out_of_scope_items = []

    passes = {
        "closeout_scope_fixed": bool(closeout_decision.get("v11_plus_scope_boundary_fixed")),
        "closeout_overall_completed": str(closeout_decision.get("overall_closeout_status")) == "completed_scope_fixed",
        "no_remaining_blocking_items": len(remaining_blocking_items) == 0,
        "born_ab_gate_continue": str(born_ab_decision.get("route_a_gate")) == "A_continue",
        "derivation_pack_gate_continue": str(derivation_pack_decision.get("route_a_gate")) == "A_continue",
        "derivation_pack_transition_stay": str(derivation_pack_decision.get("transition")) == "A_stay",
        "chain_watch_policy_continue": str(chain_watch_decision.get("route_a_gate")) == "A_continue",
        "chain_watch_policy_transition_stay": str(chain_watch_decision.get("transition")) == "A_stay",
        "shared_reject_demoted_to_watch": bool(chain_watch_policy_result.get("demoted_to_watch")),
        "selection_watch_preserved": bool(closeout_passes.get("selection_watch_preserved")),
        "single_event_probability_classified_nonblocking": str(closeout_decision.get("single_event_probability_status")) == "residual_not_p_specific",
        "irreversibility_classified_nonblocking": str(closeout_decision.get("irreversibility_status")) == "residual_not_p_specific",
        "quantum_information_minimal_connection_fixed": str(
            closeout_decision.get("quantum_information_status")
        ) == "minimal_connection_entry_fixed",
        "quantum_information_protocol_scope_preserved": "quantum_information_protocol_mathematics" in out_of_scope_items,
    }
    all_pass = all(passes.values())
    rows = _rows(
        closeout_summary=closeout_summary,
        closeout_decision=closeout_decision,
        watch_decision=chain_watch_decision,
    )

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.49.7", "name": "Part IV quantum closeout sync pack"},
        "inputs": {
            "quantum_v11_plus_scope_closeout_audit_metrics_json": _rel(closeout_metrics_path),
            "derivation_observable_chain_lock_audit_watch_policy_json": _rel(chain_watch_policy_metrics_path),
            "derivation_parameter_falsification_pack_json": _rel(derivation_pack_metrics_path),
            "quantum_connection_born_ab_gate_json": _rel(born_ab_metrics_path),
        },
        "intent": "Make the Part IV quantum verification materials read the v1.1+ closeout boundary explicitly, while preserving the older route-A operational guards as legacy diagnostics.",
        "assumptions": [
            "The v1.1+ closeout audit is the canonical boundary statement for Born / measurement / entanglement / quantum-information minimal connection.",
            "Bell-pairing-only shared-KPI rejects remain nonblocking if the frozen watch policy demotes them to watch.",
            "Protocol-level scope-out items are not upgraded to blocking failures inside Part IV.",
        ],
        "summary": {
            "closeout_overall_status": str(closeout_decision.get("overall_closeout_status", "unknown")),
            "born_closeout_status": str(closeout_decision.get("born_closeout_status", "unknown")),
            "measurement_closeout_status": str(closeout_decision.get("measurement_closeout_status", "unknown")),
            "entanglement_closeout_status": str(closeout_decision.get("entanglement_closeout_status", "unknown")),
            "quantum_information_status": str(closeout_decision.get("quantum_information_status", "unknown")),
            "quantum_information_protocol_status": str(
                closeout_decision.get("quantum_information_protocol_status", "unknown")
            ),
            "single_event_probability_status": str(closeout_decision.get("single_event_probability_status", "unknown")),
            "irreversibility_status": str(closeout_decision.get("irreversibility_status", "unknown")),
            "remaining_blocking_items_n": len(remaining_blocking_items),
            "shared_gate_policy": str(chain_watch_decision.get("shared_gate_policy", "unknown")),
            "shared_gate_policy_demoted_to_watch": bool(chain_watch_policy_result.get("demoted_to_watch")),
            "legacy_route_a_gate": str(chain_watch_decision.get("route_a_gate", "unknown")),
            "legacy_transition": str(chain_watch_decision.get("transition", "unknown")),
            "selection_watch_active_dataset_count": int(closeout_summary.get("entanglement_selection_watch_active_dataset_count")),
            "measurement_tau50_over_tauD": float(closeout_summary.get("measurement_tau50_over_tauD")),
            "measurement_branch_reversal_max": float(closeout_summary.get("measurement_branch_reversal_max")),
            "entanglement_min_chsh_visibility_proxy": float(closeout_summary.get("entanglement_min_chsh_visibility_proxy")),
            "quantum_information_exact_proxy_pass_count": int(
                closeout_summary.get("quantum_information_exact_proxy_pass_count")
            ),
            "quantum_information_entry_fixed_consistency_count": int(
                closeout_summary.get("quantum_information_entry_fixed_consistency_count")
            ),
        },
        "rows": [asdict(row) for row in rows],
        "decision": {
            "part4_quantum_sync_status": "pass" if all_pass else "reject",
            "closeout_overall_status": str(closeout_decision.get("overall_closeout_status", "unknown")),
            "closeout_remaining_blocking_items_n": len(remaining_blocking_items),
            "selection_watch_policy": "nonblocking_watch_retained" if passes["selection_watch_preserved"] else "policy_broken",
            "legacy_route_status": f"{chain_watch_decision.get('route_a_gate', 'unknown')}/{chain_watch_decision.get('transition', 'unknown')}",
            "shared_gate_policy": str(chain_watch_decision.get("shared_gate_policy", "unknown")),
            "quantum_information_status": str(closeout_decision.get("quantum_information_status", "unknown")),
            "quantum_information_protocol_status": str(
                closeout_decision.get("quantum_information_protocol_status", "unknown")
            ),
            "single_event_probability_status": str(closeout_decision.get("single_event_probability_status", "unknown")),
            "nonblocking_residual": str(closeout_decision.get("irreversibility_status", "unknown")),
            "nonblocking_residuals": closeout_decision.get("nonblocking_residuals"),
            "out_of_scope_items": out_of_scope_items,
            "passes": passes,
            "next_required_steps": ["8.7.52.6"],
        },
    }


# Function: write the sync rows in CSV form.

def _write_rows_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "status",
                "gate_snapshot",
                "closeout_status",
                "representative_metric_name",
                "representative_metric_value",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            # Guard: skip malformed rows instead of failing the whole export.
            if not isinstance(row, dict):
                continue

            writer.writerow(row)


# Function: parse CLI arguments, run the sync pack, and write the outputs.

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sync Part IV quantum verification materials to the v1.1+ closeout boundary.")
    ap.add_argument(
        "--closeout-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_v11_plus_scope_closeout_audit_metrics.json"),
        help="Input v1.1+ quantum closeout metrics JSON path.",
    )
    ap.add_argument(
        "--chain-watch-policy",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_observable_chain_lock_audit_watch_policy.json"),
        help="Input derivation-observable chain watch-policy JSON path.",
    )
    ap.add_argument(
        "--derivation-pack",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_parameter_falsification_pack.json"),
        help="Input derivation-parameter falsification pack JSON path.",
    )
    ap.add_argument(
        "--born-ab",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_born_ab_gate.json"),
        help="Input Born A/B gate JSON path.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_part4_closeout_sync_pack.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_part4_closeout_sync_rows.csv"),
        help="Output CSV path.",
    )
    args = ap.parse_args(argv)

    closeout_metrics_path = Path(args.closeout_metrics)
    chain_watch_policy_metrics_path = Path(args.chain_watch_policy)
    derivation_pack_metrics_path = Path(args.derivation_pack)
    born_ab_metrics_path = Path(args.born_ab)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    # Guard: resolve relative input/output paths against the repository root.
    if not closeout_metrics_path.is_absolute():
        closeout_metrics_path = (ROOT / closeout_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not chain_watch_policy_metrics_path.is_absolute():
        chain_watch_policy_metrics_path = (ROOT / chain_watch_policy_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not derivation_pack_metrics_path.is_absolute():
        derivation_pack_metrics_path = (ROOT / derivation_pack_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not born_ab_metrics_path.is_absolute():
        born_ab_metrics_path = (ROOT / born_ab_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    payload = build_payload(
        closeout_metrics=_read_json(closeout_metrics_path),
        chain_watch_policy_metrics=_read_json(chain_watch_policy_metrics_path),
        derivation_pack_metrics=_read_json(derivation_pack_metrics_path),
        born_ab_metrics=_read_json(born_ab_metrics_path),
        closeout_metrics_path=closeout_metrics_path,
        chain_watch_policy_metrics_path=chain_watch_policy_metrics_path,
        derivation_pack_metrics_path=derivation_pack_metrics_path,
        born_ab_metrics_path=born_ab_metrics_path,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_rows_csv(out_csv, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_part4_closeout_sync_pack",
                "phase": "8.7.49.7",
                "inputs": {
                    "quantum_v11_plus_scope_closeout_audit_metrics_json": _rel(closeout_metrics_path),
                    "derivation_observable_chain_lock_audit_watch_policy_json": _rel(chain_watch_policy_metrics_path),
                    "derivation_parameter_falsification_pack_json": _rel(derivation_pack_metrics_path),
                    "quantum_connection_born_ab_gate_json": _rel(born_ab_metrics_path),
                },
                "outputs": {
                    "quantum_part4_closeout_sync_pack_json": _rel(out_json),
                    "quantum_part4_closeout_sync_rows_csv": _rel(out_csv),
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
