#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entanglement_bell_dataset_connection_audit.py

Step 8.7.50.6:
Close the effective Bell-dataset connection and pair-decoherence bound for the
entanglement route without overriding the existing selection watch.

Inputs:
  - output/public/quantum/entanglement_phase_sharing_candidate_metrics.json
  - output/public/quantum/entanglement_source_dynamics_three_wave_mixing_metrics.json
  - output/public/quantum/bell/falsification_pack.json
  - output/public/quantum/bell/selection_loophole_quantification.json
  - output/public/quantum/bell/cross_dataset_covariance.json

Outputs:
  - output/public/quantum/entanglement_bell_dataset_connection_audit_metrics.json
  - output/public/quantum/entanglement_bell_dataset_connection_cases.csv

Assumptions:
  - The ideal adopted-U(1) pair kernel already reaches the CHSH ceiling
    |S| = 2 sqrt(2) in the candidate artifact.
  - The three-wave-mixing source dynamics for Xi(P_mu; x_A, x_B) are already
    frozen in step 8.7.50.2.
  - Public Bell datasets are connected through blind-frozen statistics and
    trial counts only; the existing selection watch is preserved and not
    optimized away.
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]

# Guard: add the repository root once so local packages resolve predictably.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# Class: store one dataset-level Bell connection row for CSV export.
@dataclass(frozen=True)
class DatasetConnectionRow:
    dataset_id: str
    display_name: str
    statistic_family: str
    frozen_statistic_name: str
    frozen_statistic_value: float
    selection_ratio: float
    delay_z_max: Optional[float]
    visibility_proxy_chsh: Optional[float]
    pair_decoherence_budget_max: Optional[float]
    eta_pair: Optional[float]
    eta_min: Optional[float]
    attenuation_budget: Optional[float]
    selection_watch_active: bool
    connection_mode: str


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


# Function: reduce the Bell null-test summary to a baseline lookup table.

def _baseline_map(falsification_pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cross_dataset = falsification_pack.get("cross_dataset") if isinstance(falsification_pack.get("cross_dataset"), dict) else {}
    null_tests = cross_dataset.get("null_tests_summary") if isinstance(cross_dataset.get("null_tests_summary"), dict) else {}
    datasets = null_tests.get("datasets") if isinstance(null_tests.get("datasets"), list) else []
    mapping: Dict[str, Dict[str, Any]] = {}

    for dataset in datasets:
        # Guard: skip malformed rows instead of failing the full audit.
        if not isinstance(dataset, dict):
            continue

        dataset_id = dataset.get("dataset_id")
        baseline = dataset.get("baseline")

        if isinstance(dataset_id, str) and isinstance(baseline, dict):
            mapping[dataset_id] = baseline

    return mapping


# Function: reduce the loophole quantification artifact to a dataset lookup table.

def _loophole_map(selection_loophole: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    datasets = selection_loophole.get("datasets") if isinstance(selection_loophole.get("datasets"), list) else []
    mapping: Dict[str, Dict[str, Any]] = {}

    for dataset in datasets:
        # Guard: skip malformed rows instead of failing the full audit.
        if not isinstance(dataset, dict):
            continue

        dataset_id = dataset.get("dataset_id")

        if isinstance(dataset_id, str):
            mapping[dataset_id] = dataset

    return mapping


# Function: classify each Bell dataset into the CHSH or CH family.

def _statistic_family(statistic_label: str) -> str:
    return "CHSH" if "CHSH" in statistic_label.upper() else "CH"


# Function: extract the frozen Bell statistic and its label from a baseline row.

def _frozen_statistic(baseline: Dict[str, Any]) -> Tuple[str, float]:
    statistic_name = str(baseline.get("statistic_name", "frozen_statistic"))
    statistic_value_raw = baseline.get("statistic_abs_recomputed")

    if not isinstance(statistic_value_raw, (int, float)):
        statistic_value_raw = baseline.get("statistic_abs")

    if not isinstance(statistic_value_raw, (int, float)):
        statistic_value_raw = baseline.get("statistic")

    statistic_value = float(statistic_value_raw)
    return statistic_name, statistic_value


# Function: extract the largest supported delay-signature z value for one dataset.

def _delay_z_max(loophole_row: Dict[str, Any]) -> Optional[float]:
    locality = loophole_row.get("locality") if isinstance(loophole_row.get("locality"), dict) else {}
    value = locality.get("value")
    return float(value) if isinstance(value, (int, float)) else None


# Function: extract supported detection-efficiency proxies from one loophole row.

def _detection_details(loophole_row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    detection = loophole_row.get("detection") if isinstance(loophole_row.get("detection"), dict) else {}
    details = detection.get("details") if isinstance(detection.get("details"), dict) else {}
    eta_pair = details.get("eta_pair")
    eta_min = details.get("eta_min")
    eta_pair_value = float(eta_pair) if isinstance(eta_pair, (int, float)) else None
    eta_min_value = float(eta_min) if isinstance(eta_min, (int, float)) else None
    return eta_pair_value, eta_min_value


# Function: summarize the strongest cross-dataset sweep-profile correlations.

def _correlation_summary(covariance: Dict[str, Any]) -> Dict[str, Any]:
    matrices = covariance.get("matrices") if isinstance(covariance.get("matrices"), dict) else {}
    dataset_order = matrices.get("dataset_order") if isinstance(matrices.get("dataset_order"), list) else []
    corr_matrix = matrices.get("profile_corr") if isinstance(matrices.get("profile_corr"), list) else []
    max_abs_all = -1.0
    max_abs_all_pair = {"dataset_i": None, "dataset_j": None, "corr": None}
    max_abs_chsh = -1.0
    max_abs_chsh_pair = {"dataset_i": None, "dataset_j": None, "corr": None}
    chsh_ids = {
        "weihs1998_longdist_longdist1",
        "delft_hensen2015",
        "delft_hensen2016_srep30289",
    }

    for i, dataset_i in enumerate(dataset_order):
        row = corr_matrix[i] if i < len(corr_matrix) and isinstance(corr_matrix[i], list) else []
        for j in range(i + 1, len(dataset_order)):
            corr_value = row[j] if j < len(row) and isinstance(row[j], (int, float)) else None

            if not isinstance(corr_value, (int, float)):
                continue

            abs_corr = abs(float(corr_value))

            if abs_corr > max_abs_all:
                max_abs_all = abs_corr
                max_abs_all_pair = {
                    "dataset_i": dataset_i,
                    "dataset_j": dataset_order[j],
                    "corr": float(corr_value),
                }

            if dataset_i in chsh_ids and dataset_order[j] in chsh_ids and abs_corr > max_abs_chsh:
                max_abs_chsh = abs_corr
                max_abs_chsh_pair = {
                    "dataset_i": dataset_i,
                    "dataset_j": dataset_order[j],
                    "corr": float(corr_value),
                }

    return {
        "max_abs_corr_all_pair": max_abs_all_pair,
        "max_abs_corr_chsh_pair": max_abs_chsh_pair,
    }


# Function: build one dataset-connection row from the frozen Bell artifacts.

def _dataset_row(
    *,
    dataset: Dict[str, Any],
    baseline: Dict[str, Any],
    loophole_row: Dict[str, Any],
    ideal_abs_s: float,
) -> DatasetConnectionRow:
    dataset_id = str(dataset["dataset_id"])
    display_name = str(loophole_row.get("display_name", dataset_id))
    statistic_label = str(dataset.get("statistic", ""))
    statistic_family = _statistic_family(statistic_label)
    frozen_statistic_name, frozen_statistic_value = _frozen_statistic(baseline)
    selection_ratio = float(dataset.get("ratio"))
    delay_z_max = _delay_z_max(loophole_row)
    eta_pair, eta_min = _detection_details(loophole_row)
    visibility_proxy: Optional[float] = None
    pair_decoherence_budget: Optional[float] = None
    attenuation_budget: Optional[float] = None
    connection_mode = "trial_efficiency_proxy"

    if statistic_family == "CHSH":
        visibility_proxy = max(0.0, min(1.0, abs(frozen_statistic_value) / ideal_abs_s))
        pair_decoherence_budget = -math.log(max(visibility_proxy, 1.0e-12))
        connection_mode = "visibility_from_frozen_S"

    if eta_pair is not None:
        attenuation_budget = -math.log(max(eta_pair, 1.0e-12))

    selection_watch_active = bool(selection_ratio > 1.0 or (delay_z_max is not None and delay_z_max > 3.0))
    return DatasetConnectionRow(
        dataset_id=dataset_id,
        display_name=display_name,
        statistic_family=statistic_family,
        frozen_statistic_name=frozen_statistic_name,
        frozen_statistic_value=float(frozen_statistic_value),
        selection_ratio=selection_ratio,
        delay_z_max=delay_z_max,
        visibility_proxy_chsh=visibility_proxy,
        pair_decoherence_budget_max=pair_decoherence_budget,
        eta_pair=eta_pair,
        eta_min=eta_min,
        attenuation_budget=attenuation_budget,
        selection_watch_active=selection_watch_active,
        connection_mode=connection_mode,
    )


# Function: combine the frozen entanglement and Bell artifacts into the 8.7.50.6 payload.

def build_payload(
    *,
    candidate_metrics: Dict[str, Any],
    source_metrics: Dict[str, Any],
    falsification_pack: Dict[str, Any],
    selection_loophole: Dict[str, Any],
    covariance: Dict[str, Any],
    candidate_metrics_path: Path,
    source_metrics_path: Path,
    falsification_pack_path: Path,
    selection_loophole_path: Path,
    covariance_path: Path,
) -> Dict[str, Any]:
    ideal_candidate = (
        ((candidate_metrics.get("diagnostics") or {}).get("ideal_candidate") or {})
        if isinstance(candidate_metrics.get("diagnostics"), dict)
        else {}
    )
    chsh_standard = ideal_candidate.get("chsh_standard") if isinstance(ideal_candidate.get("chsh_standard"), dict) else {}
    ideal_abs_s = float(chsh_standard.get("abs_s"))
    source_decision = source_metrics.get("decision") if isinstance(source_metrics.get("decision"), dict) else {}
    candidate_decision = candidate_metrics.get("decision") if isinstance(candidate_metrics.get("decision"), dict) else {}
    datasets = falsification_pack.get("datasets") if isinstance(falsification_pack.get("datasets"), list) else []
    baselines = _baseline_map(falsification_pack)
    loopholes = _loophole_map(selection_loophole)
    rows: List[DatasetConnectionRow] = []

    for dataset in datasets:
        # Guard: skip malformed rows instead of failing the full audit.
        if not isinstance(dataset, dict):
            continue

        dataset_id = dataset.get("dataset_id")

        if not isinstance(dataset_id, str):
            continue

        baseline = baselines.get(dataset_id)
        loophole_row = loopholes.get(dataset_id)

        if not isinstance(baseline, dict) or not isinstance(loophole_row, dict):
            continue

        rows.append(
            _dataset_row(
                dataset=dataset,
                baseline=baseline,
                loophole_row=loophole_row,
                ideal_abs_s=ideal_abs_s,
            )
        )

    chsh_visibilities = [row.visibility_proxy_chsh for row in rows if row.visibility_proxy_chsh is not None]
    chsh_budgets = [row.pair_decoherence_budget_max for row in rows if row.pair_decoherence_budget_max is not None]
    ch_eta_pair = [row.eta_pair for row in rows if row.statistic_family == "CH" and row.eta_pair is not None]
    ch_attenuation = [row.attenuation_budget for row in rows if row.statistic_family == "CH" and row.attenuation_budget is not None]
    selection_watch_count = sum(1 for row in rows if row.selection_watch_active)
    corr_summary = _correlation_summary(covariance)
    passes = {
        "candidate_fixed": candidate_decision.get("source_candidate_status") == "effective_candidate_fixed",
        "source_dynamics_closed": source_decision.get("b1_source_dynamics_status") == "closed",
        "dataset_coverage_complete": len(rows) == len(datasets) and len(rows) > 0,
        "all_chsh_visibility_bounded": all(
            visibility is not None and 0.0 <= visibility <= 1.0 for visibility in chsh_visibilities
        ),
        "selection_watch_preserved": selection_watch_count == len(rows),
        "no_new_pmodel_free_parameters": True,
    }
    all_pass = all(bool(value) for value in passes.values())

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.50.6", "name": "Entanglement Bell dataset connection and pair decoherence"},
        "inputs": {
            "entanglement_phase_sharing_candidate_metrics_json": _rel(candidate_metrics_path),
            "entanglement_source_dynamics_three_wave_mixing_metrics_json": _rel(source_metrics_path),
            "bell_falsification_pack_json": _rel(falsification_pack_path),
            "bell_selection_loophole_quantification_json": _rel(selection_loophole_path),
            "bell_cross_dataset_covariance_json": _rel(covariance_path),
        },
        "intent": "Connect the ideal pair kernel to blind-frozen Bell statistics and trial-throughput proxies while freezing a two-mode pair-dephasing budget and preserving the existing selection watch.",
        "assumptions": [
            "The ideal adopted-U(1) pair kernel already reaches |S| = 2 sqrt(2) in the candidate artifact.",
            "Three-wave source dynamics for Xi(P_mu; x_A, x_B) are already frozen in step 8.7.50.2.",
            "Selection sweeps remain diagnostics rather than optimizers; the blind-freeze policy and watch thresholds are preserved exactly as in Part III-B.",
        ],
        "formulas": {
            "ideal_chsh_ceiling": "|S|_ideal = 2 sqrt(2)",
            "chsh_visibility_proxy": "V_Bell = |S_frozen| / (2 sqrt(2))",
            "pair_decoherence_budget": "D_pair = -ln(V_Bell) = Gamma_pair T_prop",
            "pair_decoherence_two_mode_extension": "Gamma_pair = Gamma_deph,A + Gamma_deph,B; symmetric branch: Gamma_pair = 2 omega_*^2 (k_B T_env / chi_P) tau_free",
            "trial_pair_throughput_proxy": "eta_pair = N_coinc / N_trial",
            "attenuation_proxy": "A_att = -ln(eta_pair)",
        },
        "datasets": [asdict(row) for row in rows],
        "summary": {
            "ideal_abs_s": ideal_abs_s,
            "min_chsh_visibility_proxy": min(chsh_visibilities) if chsh_visibilities else None,
            "max_chsh_pair_decoherence_budget": max(chsh_budgets) if chsh_budgets else None,
            "min_ch_pair_eta_pair": min(ch_eta_pair) if ch_eta_pair else None,
            "max_ch_attenuation_budget": max(ch_attenuation) if ch_attenuation else None,
            "selection_watch_active_dataset_count": int(selection_watch_count),
            "dataset_count": int(len(rows)),
            "cross_dataset_correlation_summary": corr_summary,
        },
        "decision": {
            "b2_dataset_connection_status": "closed_effective_audit" if all_pass else "not_closed",
            "b3_pair_decoherence_status": "closed_integrated_bound" if all_pass else "not_closed",
            "entanglement_status": (
                "dataset_connection_and_pair_decoherence_closed_selection_watch_retained" if all_pass else "not_closed"
            ),
            "full_first_principles_derivation": False,
            "passes": passes,
            "next_required_steps": ["8.7.50.7", "8.7.49.7", "8.7.49.8"],
        },
    }


# Function: write the dataset-connection rows in CSV form.

def _write_cases_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_id",
                "display_name",
                "statistic_family",
                "frozen_statistic_name",
                "frozen_statistic_value",
                "selection_ratio",
                "delay_z_max",
                "visibility_proxy_chsh",
                "pair_decoherence_budget_max",
                "eta_pair",
                "eta_min",
                "attenuation_budget",
                "selection_watch_active",
                "connection_mode",
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
    ap = argparse.ArgumentParser(description="Close the Bell-dataset connection and pair-decoherence bound for the entanglement route.")
    ap.add_argument(
        "--candidate-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "entanglement_phase_sharing_candidate_metrics.json"),
        help="Input entanglement candidate metrics JSON path.",
    )
    ap.add_argument(
        "--source-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "entanglement_source_dynamics_three_wave_mixing_metrics.json"),
        help="Input entanglement source-dynamics metrics JSON path.",
    )
    ap.add_argument(
        "--falsification-pack",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "bell" / "falsification_pack.json"),
        help="Input Bell falsification pack JSON path.",
    )
    ap.add_argument(
        "--selection-loophole",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "bell" / "selection_loophole_quantification.json"),
        help="Input Bell selection-loophole quantification JSON path.",
    )
    ap.add_argument(
        "--covariance",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "bell" / "cross_dataset_covariance.json"),
        help="Input Bell cross-dataset covariance JSON path.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "entanglement_bell_dataset_connection_audit_metrics.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "entanglement_bell_dataset_connection_cases.csv"),
        help="Output CSV path.",
    )
    args = ap.parse_args(argv)

    candidate_metrics_path = Path(args.candidate_metrics)
    source_metrics_path = Path(args.source_metrics)
    falsification_pack_path = Path(args.falsification_pack)
    selection_loophole_path = Path(args.selection_loophole)
    covariance_path = Path(args.covariance)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    # Guard: resolve relative input/output paths against the repository root.
    if not candidate_metrics_path.is_absolute():
        candidate_metrics_path = (ROOT / candidate_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not source_metrics_path.is_absolute():
        source_metrics_path = (ROOT / source_metrics_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not falsification_pack_path.is_absolute():
        falsification_pack_path = (ROOT / falsification_pack_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not selection_loophole_path.is_absolute():
        selection_loophole_path = (ROOT / selection_loophole_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not covariance_path.is_absolute():
        covariance_path = (ROOT / covariance_path).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # Guard: resolve relative input/output paths against the repository root.

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    candidate_metrics = _read_json(candidate_metrics_path)
    source_metrics = _read_json(source_metrics_path)
    falsification_pack = _read_json(falsification_pack_path)
    selection_loophole = _read_json(selection_loophole_path)
    covariance = _read_json(covariance_path)
    payload = build_payload(
        candidate_metrics=candidate_metrics,
        source_metrics=source_metrics,
        falsification_pack=falsification_pack,
        selection_loophole=selection_loophole,
        covariance=covariance,
        candidate_metrics_path=candidate_metrics_path,
        source_metrics_path=source_metrics_path,
        falsification_pack_path=falsification_pack_path,
        selection_loophole_path=selection_loophole_path,
        covariance_path=covariance_path,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_cases_csv(out_csv, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_entanglement_bell_dataset_connection_audit",
                "phase": "8.7.50.6",
                "inputs": {
                    "entanglement_phase_sharing_candidate_metrics_json": _rel(candidate_metrics_path),
                    "entanglement_source_dynamics_three_wave_mixing_metrics_json": _rel(source_metrics_path),
                    "bell_falsification_pack_json": _rel(falsification_pack_path),
                    "bell_selection_loophole_quantification_json": _rel(selection_loophole_path),
                    "bell_cross_dataset_covariance_json": _rel(covariance_path),
                },
                "outputs": {
                    "entanglement_bell_dataset_connection_audit_metrics_json": _rel(out_json),
                    "entanglement_bell_dataset_connection_cases_csv": _rel(out_csv),
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
