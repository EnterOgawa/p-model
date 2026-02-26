#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
derivation_parameter_falsification_pack.py

Step 8.7.11:
導出由来パラメータ（自由係数・境界条件・近似順序）を、
Bell / 干渉 / 物性・熱 holdout への観測拘束に写像し、
`A継続 / A棄却→B` を機械判定できる pack として固定する。

出力:
  - output/public/quantum/derivation_parameter_falsification_pack.json
  - output/public/quantum/derivation_parameter_falsification_pack.csv
  - output/public/quantum/derivation_parameter_falsification_pack.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> Optional[float]:
    # 条件分岐: `isinstance(value, (int, float))` を満たす経路を評価する。
    if isinstance(value, (int, float)):
        number = float(value)
        # 条件分岐: `math.isfinite(number)` を満たす経路を評価する。
        if math.isfinite(number):
            return number

    return None


def _compute_pass(value: Optional[float], threshold: Optional[float], operator: str) -> Optional[bool]:
    # 条件分岐: `value is None or threshold is None` を満たす経路を評価する。
    if value is None or threshold is None:
        return None

    # 条件分岐: `operator == "<="` を満たす経路を評価する。

    if operator == "<=":
        return bool(value <= threshold)

    # 条件分岐: `operator == ">="` を満たす経路を評価する。

    if operator == ">=":
        return bool(value >= threshold)

    return None


def _normalized_score(value: Optional[float], threshold: Optional[float], operator: str) -> Optional[float]:
    # 条件分岐: `value is None or threshold is None or threshold == 0.0` を満たす経路を評価する。
    if value is None or threshold is None or threshold == 0.0:
        return None

    # 条件分岐: `operator == "<="` を満たす経路を評価する。

    if operator == "<=":
        return float(value / threshold)

    # 条件分岐: `operator == ">="` を満たす経路を評価する。

    if operator == ">=":
        # 条件分岐: `value == 0.0` を満たす経路を評価する。
        if value == 0.0:
            return math.inf

        return float(threshold / value)

    return None


def _row_status(passed: Optional[bool], gate_level: str) -> str:
    # 条件分岐: `passed is True` を満たす経路を評価する。
    if passed is True:
        return "pass"

    # 条件分岐: `passed is None` を満たす経路を評価する。

    if passed is None:
        return "unknown"

    # 条件分岐: `gate_level == "hard"` を満たす経路を評価する。

    if gate_level == "hard":
        return "reject"

    return "watch"


def _criteria_map(criteria: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    # 条件分岐: `not isinstance(criteria, list)` を満たす経路を評価する。
    if not isinstance(criteria, list):
        return out

    for row in criteria:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        criterion_id = str(row.get("id") or "")
        # 条件分岐: `criterion_id` を満たす経路を評価する。
        if criterion_id:
            out[criterion_id] = row

    return out


def _channel_entry(shared_kpi: Dict[str, Any], channel_name: str) -> Dict[str, Any]:
    channels = shared_kpi.get("channels") if isinstance(shared_kpi.get("channels"), list) else []
    for row in channels:
        # 条件分岐: `isinstance(row, dict) and str(row.get("channel") or "") == channel_name` を満たす経路を評価する。
        if isinstance(row, dict) and str(row.get("channel") or "") == channel_name:
            return row

    return {}


def _extract_condensed_counts(shared_kpi: Dict[str, Any], condensed_summary: Dict[str, Any]) -> Dict[str, int]:
    channel = _channel_entry(shared_kpi, "condensed_thermal_holdout")
    kpi = channel.get("kpi") if isinstance(channel.get("kpi"), dict) else {}
    reject_n = kpi.get("reject_n")
    inconclusive_n = kpi.get("inconclusive_n")

    reject_count = int(reject_n) if isinstance(reject_n, (int, float)) else 0
    inconclusive_count = int(inconclusive_n) if isinstance(inconclusive_n, (int, float)) else 0

    # 条件分岐: `isinstance(reject_n, (int, float)) and isinstance(inconclusive_n, (int, float))` を満たす経路を評価する。
    if isinstance(reject_n, (int, float)) and isinstance(inconclusive_n, (int, float)):
        return {"reject_n": reject_count, "inconclusive_n": inconclusive_count}

    summary = condensed_summary.get("summary") if isinstance(condensed_summary.get("summary"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    # 条件分岐: `not datasets` を満たす経路を評価する。
    if not datasets:
        return {"reject_n": reject_count, "inconclusive_n": inconclusive_count}

    reject_count = 0
    inconclusive_count = 0
    for dataset in datasets:
        # 条件分岐: `not isinstance(dataset, dict)` を満たす経路を評価する。
        if not isinstance(dataset, dict):
            continue

        audit_gates = dataset.get("audit_gates") if isinstance(dataset.get("audit_gates"), dict) else {}
        falsification = audit_gates.get("falsification") if isinstance(audit_gates.get("falsification"), dict) else {}
        status = str(falsification.get("status") or "").strip().lower()
        # 条件分岐: `status == "reject"` を満たす経路を評価する。
        if status == "reject":
            reject_count += 1
        # 条件分岐: 前段条件が不成立で、`status != "ok"` を追加評価する。
        elif status != "ok":
            inconclusive_count += 1

    return {"reject_n": reject_count, "inconclusive_n": inconclusive_count}


def _add_mapped_row(
    rows: List[Dict[str, Any]],
    *,
    row_id: str,
    parameter_id: str,
    parameter_type: str,
    channel: str,
    metric: str,
    value: Optional[float],
    threshold: Optional[float],
    operator: str,
    gate_level: str,
    source: str,
    note: str,
) -> None:
    passed = _compute_pass(value=value, threshold=threshold, operator=operator)
    rows.append(
        {
            "id": row_id,
            "parameter_id": parameter_id,
            "parameter_type": parameter_type,
            "channel": channel,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "operator": operator,
            "pass": passed,
            "gate_level": gate_level,
            "status": _row_status(passed=passed, gate_level=gate_level),
            "normalized_score": _normalized_score(value=value, threshold=threshold, operator=operator),
            "source": source,
            "note": note,
        }
    )


def build_pack() -> Dict[str, Any]:
    action_path = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
    nonrel_path = ROOT / "output" / "public" / "quantum" / "nonrelativistic_reduction_schrodinger_mapping_audit.json"
    born_path = ROOT / "output" / "public" / "quantum" / "born_route_a_proxy_constraints_pack.json"
    born_ab_path = ROOT / "output" / "public" / "quantum" / "quantum_connection_born_ab_gate.json"
    shared_kpi_path = ROOT / "output" / "public" / "quantum" / "quantum_connection_shared_kpi.json"
    condensed_summary_path = ROOT / "output" / "public" / "quantum" / "condensed_holdout_audit_summary.json"

    action = _read_json(action_path)
    nonrel = _read_json(nonrel_path)
    born = _read_json(born_path)
    born_ab = _read_json(born_ab_path)
    shared_kpi = _read_json(shared_kpi_path)
    condensed_summary = _read_json(condensed_summary_path)

    action_criteria = _criteria_map((action.get("numerical_audit") or {}).get("criteria"))
    born_criteria = _criteria_map(born.get("criteria"))
    nonrel_criteria = nonrel.get("criteria") if isinstance(nonrel.get("criteria"), list) else []

    rows: List[Dict[str, Any]] = []

    for criterion_id, metric_label in [
        ("covariant_derivative_gauge_covariance", "max_rel_error(D_mu P)"),
        ("kinetic_density_gauge_invariance", "max_rel_error(|D_mu P|^2)"),
        ("noether_current_gauge_invariance", "max_rel_error(j_mu)"),
    ]:
        source_row = action_criteria.get(criterion_id, {})
        _add_mapped_row(
            rows,
            row_id=f"derivation::{criterion_id}",
            parameter_id="boundary_variation_and_regularization",
            parameter_type="boundary_condition",
            channel="derivation",
            metric=metric_label,
            value=_as_float(source_row.get("value")),
            threshold=_as_float(source_row.get("threshold")),
            operator=str(source_row.get("operator") or "<="),
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="作用原理→EL 導出の境界条件/共変性監査。",
        )

    for source_row in nonrel_criteria:
        # 条件分岐: `not isinstance(source_row, dict)` を満たす経路を評価する。
        if not isinstance(source_row, dict):
            continue

        channel_name = str(source_row.get("channel") or "")
        _add_mapped_row(
            rows,
            row_id=f"nonrel::{channel_name}",
            parameter_id="nonrelativistic_epsilon_order",
            parameter_type="approximation_order",
            channel="interference",
            metric=f"{channel_name}::{str(source_row.get('metric') or 'epsilon_max')}",
            value=_as_float(source_row.get("value")),
            threshold=_as_float(source_row.get("threshold")),
            operator=str(source_row.get("operator") or "<="),
            gate_level="hard",
            source="nonrelativistic_reduction_schrodinger_mapping_audit",
            note="2.6→2.5 の近似順序ゲート（max ε）。",
        )

    for criterion_id, channel_name, gate_level, parameter_id in [
        ("selection_delay_signature_fast", "bell", "hard", "event_window_boundary_condition"),
        ("selection_sweep_sensitivity_fast", "bell", "hard", "minimal_coupling_strength_proxy"),
        ("phase_alpha_consistency", "interference", "hard", "minimal_coupling_strength_proxy"),
        ("phase_molecular_scaling", "interference", "hard", "minimal_coupling_strength_proxy"),
        ("visibility_atom_precision_gap", "interference", "soft", "visibility_resolution_budget"),
    ]:
        source_row = born_criteria.get(criterion_id, {})
        _add_mapped_row(
            rows,
            row_id=f"born::{criterion_id}",
            parameter_id=parameter_id,
            parameter_type="free_coefficient",
            channel=channel_name,
            metric=str(source_row.get("metric") or criterion_id),
            value=_as_float(source_row.get("value")),
            threshold=_as_float(source_row.get("threshold")),
            operator=str(source_row.get("operator") or "<="),
            gate_level=gate_level,
            source="born_route_a_proxy_constraints_pack",
            note=str(source_row.get("note") or "BornルートA観測proxy拘束。"),
        )

    condensed_counts = _extract_condensed_counts(shared_kpi=shared_kpi, condensed_summary=condensed_summary)
    _add_mapped_row(
        rows,
        row_id="condensed::reject_count",
        parameter_id="higher_derivative_and_material_effective_terms",
        parameter_type="boundary_condition",
        channel="condensed_thermal_holdout",
        metric="holdout_reject_n",
        value=float(condensed_counts["reject_n"]),
        threshold=0.0,
        operator="<=",
        gate_level="soft",
        source="quantum_connection_shared_kpi/condensed_holdout_audit_summary",
        note="物性/熱 holdout の reject 件数（0 へ収束が pass 条件）。",
    )
    _add_mapped_row(
        rows,
        row_id="condensed::inconclusive_count",
        parameter_id="higher_derivative_and_material_effective_terms",
        parameter_type="boundary_condition",
        channel="condensed_thermal_holdout",
        metric="holdout_inconclusive_n",
        value=float(condensed_counts["inconclusive_n"]),
        threshold=0.0,
        operator="<=",
        gate_level="soft",
        source="quantum_connection_shared_kpi/condensed_holdout_audit_summary",
        note="物性/熱 holdout の inconclusive 件数（0 が望ましい）。",
    )

    rows_by_parameter: Dict[str, List[str]] = {}
    for row in rows:
        parameter_key = str(row.get("parameter_id") or "")
        rows_by_parameter.setdefault(parameter_key, []).append(str(row.get("id") or ""))

    charge_q = _as_float(((action.get("numerical_audit") or {}).get("charge_q")))
    epsilon_gate = None
    for row in nonrel_criteria:
        # 条件分岐: `isinstance(row, dict)` を満たす経路を評価する。
        if isinstance(row, dict):
            epsilon_gate = _as_float(row.get("threshold"))
            # 条件分岐: `epsilon_gate is not None` を満たす経路を評価する。
            if epsilon_gate is not None:
                break

    parameter_catalog = [
        {
            "parameter_id": "minimal_coupling_strength_proxy",
            "type": "free_coefficient",
            "frozen_value": charge_q,
            "frozen_rule": "q is treated as environment-independent in the current route-A operational audit.",
            "mapped_rows": rows_by_parameter.get("minimal_coupling_strength_proxy", []),
            "mapped_channels": ["bell", "interference"],
        },
        {
            "parameter_id": "event_window_boundary_condition",
            "type": "boundary_condition",
            "frozen_value": None,
            "frozen_rule": "event-ready / window definition is fixed before route-A decision.",
            "mapped_rows": rows_by_parameter.get("event_window_boundary_condition", []),
            "mapped_channels": ["bell"],
        },
        {
            "parameter_id": "boundary_variation_and_regularization",
            "type": "boundary_condition",
            "frozen_value": "delta_fields_at_boundary = 0 (operational)",
            "frozen_rule": "EL derivation audit enforces gauge covariance/invariance under fixed boundary conditions.",
            "mapped_rows": rows_by_parameter.get("boundary_variation_and_regularization", []),
            "mapped_channels": ["derivation"],
        },
        {
            "parameter_id": "nonrelativistic_epsilon_order",
            "type": "approximation_order",
            "frozen_value": epsilon_gate,
            "frozen_rule": "max(epsilon_v2, epsilon_phi, epsilon_env) <= threshold for 2.6->2.5 reduction.",
            "mapped_rows": rows_by_parameter.get("nonrelativistic_epsilon_order", []),
            "mapped_channels": ["interference"],
        },
        {
            "parameter_id": "higher_derivative_and_material_effective_terms",
            "type": "boundary_condition",
            "frozen_value": "effective correction terms are accepted only if holdout reject/inconclusive counts are zero",
            "frozen_rule": "material/thermal channel remains watch until reject_n=inconclusive_n=0.",
            "mapped_rows": rows_by_parameter.get("higher_derivative_and_material_effective_terms", []),
            "mapped_channels": ["condensed_thermal_holdout"],
        },
    ]

    hard_fail_ids = [str(row.get("id") or "") for row in rows if str(row.get("gate_level") or "") == "hard" and row.get("pass") is False]
    hard_unknown_ids = [str(row.get("id") or "") for row in rows if str(row.get("gate_level") or "") == "hard" and row.get("pass") is None]
    soft_watch_ids = [
        str(row.get("id") or "")
        for row in rows
        if str(row.get("gate_level") or "") != "hard" and row.get("pass") is not True
    ]

    born_ab_decision = born_ab.get("decision") if isinstance(born_ab.get("decision"), dict) else {}
    born_pack_decision = born.get("decision") if isinstance(born.get("decision"), dict) else {}
    cross_gate = str(born_ab_decision.get("route_a_gate") or "")
    cross_transition = str(born_ab_decision.get("transition") or "")
    source_watchlist = born_ab_decision.get("watchlist") if isinstance(born_ab_decision.get("watchlist"), list) else []

    route_a_gate = "A_continue"
    # 条件分岐: `hard_fail_ids or hard_unknown_ids` を満たす経路を評価する。
    if hard_fail_ids or hard_unknown_ids:
        route_a_gate = "A_reject"
    # 条件分岐: 前段条件が不成立で、`cross_gate in ("A_continue", "A_reject")` を追加評価する。
    elif cross_gate in ("A_continue", "A_reject"):
        route_a_gate = cross_gate
    # 条件分岐: 前段条件が不成立で、`str(born_pack_decision.get("route_a_gate") or "") in ("A_continue", "A_reject")` を追加評価する。
    elif str(born_pack_decision.get("route_a_gate") or "") in ("A_continue", "A_reject"):
        route_a_gate = str(born_pack_decision.get("route_a_gate"))

    transition = "A_to_B" if route_a_gate == "A_reject" else "A_stay"
    # 条件分岐: `cross_transition == "A_to_B" and route_a_gate == "A_continue"` を満たす経路を評価する。
    if cross_transition == "A_to_B" and route_a_gate == "A_continue":
        transition = "A_stay"

    watchlist = sorted(set([str(x) for x in source_watchlist if isinstance(x, str)] + soft_watch_ids))

    channel_summary: Dict[str, Dict[str, Any]] = {}
    for channel_name in ["derivation", "bell", "interference", "condensed_thermal_holdout"]:
        channel_rows = [row for row in rows if str(row.get("channel") or "") == channel_name]
        status_count = {"pass": 0, "watch": 0, "reject": 0, "unknown": 0}
        for row in channel_rows:
            status = str(row.get("status") or "unknown")
            # 条件分岐: `status not in status_count` を満たす経路を評価する。
            if status not in status_count:
                status = "unknown"

            status_count[status] += 1

        shared_row = _channel_entry(shared_kpi, channel_name)
        channel_summary[channel_name] = {
            "rows_n": len(channel_rows),
            "status_counts": status_count,
            "shared_kpi_status": shared_row.get("status") if isinstance(shared_row, dict) else None,
        }

    overall_status = "pass" if route_a_gate == "A_continue" else "reject"

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.11", "name": "Derivation-parameter falsification packaging"},
        "intent": (
            "Freeze derivation-driven free coefficients / boundary conditions as machine-auditable "
            "falsification gates mapped to Bell, interference, and condensed-thermal holdout channels."
        ),
        "inputs": {
            "action_principle_el_derivation_audit_json": _rel(action_path),
            "nonrelativistic_reduction_schrodinger_mapping_audit_json": _rel(nonrel_path),
            "born_route_a_proxy_constraints_pack_json": _rel(born_path),
            "quantum_connection_born_ab_gate_json": _rel(born_ab_path),
            "quantum_connection_shared_kpi_json": _rel(shared_kpi_path),
            "condensed_holdout_audit_summary_json": _rel(condensed_summary_path),
        },
        "parameter_catalog": parameter_catalog,
        "criteria": rows,
        "channel_summary": channel_summary,
        "decision": {
            "route_a_gate": route_a_gate,
            "transition": transition,
            "overall_status": overall_status,
            "hard_fail_ids": hard_fail_ids,
            "hard_unknown_ids": hard_unknown_ids,
            "watchlist": watchlist,
            "rule": "A_to_B if any hard derivation gate fails/unknown; otherwise A_stay (watchlist tracked separately).",
        },
        "diagnostics": {
            "source_route_a_gate_born_pack": born_pack_decision.get("route_a_gate"),
            "source_route_a_gate_cross_channel": cross_gate,
            "source_transition_cross_channel": cross_transition,
            "shared_overall_status": (shared_kpi.get("overall") or {}).get("status") if isinstance(shared_kpi.get("overall"), dict) else None,
            "condensed_reject_n": condensed_counts["reject_n"],
            "condensed_inconclusive_n": condensed_counts["inconclusive_n"],
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "parameter_id",
                "parameter_type",
                "channel",
                "metric",
                "value",
                "threshold",
                "operator",
                "pass",
                "gate_level",
                "status",
                "normalized_score",
                "source",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("criteria") if isinstance(payload.get("criteria"), list) else []
    labels: List[str] = []
    scores: List[float] = []
    colors: List[str] = []
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        labels.append(str(row.get("id") or ""))
        score = _as_float(row.get("normalized_score"))
        scores.append(score if score is not None else math.nan)
        status = str(row.get("status") or "unknown")
        # 条件分岐: `status == "pass"` を満たす経路を評価する。
        if status == "pass":
            colors.append("#2f9e44")
        # 条件分岐: 前段条件が不成立で、`status == "reject"` を追加評価する。
        elif status == "reject":
            colors.append("#e03131")
        # 条件分岐: 前段条件が不成立で、`status == "watch"` を追加評価する。
        elif status == "watch":
            colors.append("#f2c94c")
        else:
            colors.append("#9ca3af")

    figure_height = max(4.8, 0.33 * len(labels) + 1.6)
    y_values = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.4, figure_height), dpi=180)
    ax.barh(y_values, scores, color=colors)
    ax.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_yticks(y_values, labels)
    ax.set_xlabel("normalized score (<=1 means threshold satisfied)")
    decision = (payload.get("decision") or {}).get("route_a_gate") if isinstance(payload.get("decision"), dict) else "unknown"
    ax.set_title(f"Derivation-parameter falsification pack ({decision})")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate derivation-parameter falsification pack (Step 8.7.11).")
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_parameter_falsification_pack.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_parameter_falsification_pack.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_parameter_falsification_pack.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    def _resolve(path_text: str) -> Path:
        path = Path(path_text)
        # 条件分岐: `path.is_absolute()` を満たす経路を評価する。
        if path.is_absolute():
            return path.resolve()

        return (ROOT / path).resolve()

    out_json = _resolve(args.out_json)
    out_csv = _resolve(args.out_csv)
    out_png = _resolve(args.out_png)

    payload = build_pack()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = payload.get("criteria") if isinstance(payload.get("criteria"), list) else []
    _write_csv(out_csv, rows)
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_derivation_parameter_falsification_pack",
                "phase": "8.7.11",
                "inputs": payload.get("inputs"),
                "outputs": {
                    "derivation_parameter_falsification_pack_json": _rel(out_json),
                    "derivation_parameter_falsification_pack_csv": _rel(out_csv),
                    "derivation_parameter_falsification_pack_png": _rel(out_png),
                },
                "decision": payload.get("decision"),
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
