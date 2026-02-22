#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_connection_born_ab_gate.py

Step 7.21.3:
既存の Born 2.6.2 ルートAゲート（A_continue / A_reject）を、
COW / atom / HOM proxy へ横断適用して単一ロジック化する。

出力:
  - output/public/quantum/quantum_connection_born_ab_gate.json
  - output/public/quantum/quantum_connection_born_ab_gate.csv
  - output/public/quantum/quantum_connection_born_ab_gate.png
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
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _find_row(rows: List[Dict[str, Any]], channel: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("channel") or "") == channel:
            return row
    return None


def _compute_pass(value: Optional[float], threshold: Optional[float], operator: str) -> Optional[bool]:
    if value is None or threshold is None:
        return None
    if operator == "<=":
        return bool(value <= threshold)
    if operator == ">=":
        return bool(value >= threshold)
    return None


def _normalized_score(value: Optional[float], threshold: Optional[float], operator: str) -> Optional[float]:
    if value is None or threshold is None or threshold == 0.0:
        return None
    if operator == "<=":
        return float(value) / float(threshold)
    if operator == ">=":
        if value == 0.0:
            return math.inf
        return float(threshold) / float(value)
    return None


def _row_status(passed: Optional[bool], gate: bool) -> str:
    if passed is True:
        return "pass"
    if gate:
        return "reject"
    return "watch"


def _criterion(
    *,
    cid: str,
    channel: str,
    proxy: str,
    metric: str,
    value: Optional[float],
    threshold: float,
    operator: str,
    gate: bool,
    note: str,
    source: str,
) -> Dict[str, Any]:
    passed = _compute_pass(value, threshold, operator)
    return {
        "id": cid,
        "channel": channel,
        "proxy": proxy,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "operator": operator,
        "pass": passed,
        "gate": gate,
        "status": _row_status(passed, gate),
        "normalized_score": _normalized_score(value, threshold, operator),
        "source": source,
        "note": note,
    }


def _from_born_pack(born_pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    criteria_in = born_pack.get("criteria") if isinstance(born_pack.get("criteria"), list) else []
    channel_map = {
        "phase_alpha_consistency": "atom",
        "phase_molecular_scaling": "molecule",
        "selection_delay_signature_fast": "bell_selection",
        "selection_sweep_sensitivity_fast": "bell_selection",
        "visibility_atom_precision_gap": "atom",
    }
    out: List[Dict[str, Any]] = []
    for row in criteria_in:
        if not isinstance(row, dict):
            continue
        value = _as_float(row.get("value"))
        threshold = _as_float(row.get("threshold"))
        operator = str(row.get("operator") or "")
        if threshold is None or operator not in ("<=", ">="):
            continue
        cid = str(row.get("id") or "")
        out.append(
            _criterion(
                cid=cid,
                channel=channel_map.get(cid, "unknown"),
                proxy=str(row.get("proxy") or ""),
                metric=str(row.get("metric") or ""),
                value=value,
                threshold=threshold,
                operator=operator,
                gate=bool(row.get("gate")),
                note=str(row.get("note") or ""),
                source="born_route_a_proxy_constraints_pack",
            )
        )
    return out


def build_payload(
    *,
    born_pack_json: Path,
    cow_metrics_json: Path,
    matter_metrics_json: Path,
    hom_metrics_json: Path,
) -> Dict[str, Any]:
    born_pack = _read_json(born_pack_json)
    cow = _read_json(cow_metrics_json)
    matter = _read_json(matter_metrics_json)
    hom = _read_json(hom_metrics_json)

    criteria = _from_born_pack(born_pack)

    cow_metrics = cow.get("metrics") if isinstance(cow.get("metrics"), dict) else {}
    cow_residual = (
        cow_metrics.get("residual_audit")
        if isinstance(cow_metrics.get("residual_audit"), dict)
        else {}
    )
    cow_max_residual_fraction = _as_float(cow_residual.get("max_abs_residual_fraction"))
    cow_coverage_ratio_legacy = _as_float(cow_metrics.get("numeric_coverage_ratio"))
    cow_coverage_ratio_targeted = _as_float(cow_metrics.get("numeric_coverage_ratio_targeted"))
    if cow_coverage_ratio_targeted is not None:
        cow_coverage_metric = "numeric_coverage_ratio_targeted"
        cow_coverage_ratio = cow_coverage_ratio_targeted
        cow_coverage_threshold = 1.0
        cow_coverage_note = "COW 数値化カタログの被覆率（targeted）は監視項目。numeric-target rows で 1.0 を閾値とする。"
    else:
        cow_coverage_metric = "numeric_coverage_ratio"
        cow_coverage_ratio = cow_coverage_ratio_legacy
        cow_coverage_threshold = 0.20
        cow_coverage_note = "COW 数値化カタログの被覆率（legacy total-record）は監視項目（未達でも即棄却しない）。"

    criteria.append(
        _criterion(
            cid="cow_residual_fraction",
            channel="cow",
            proxy="phase",
            metric="max_abs_residual_fraction",
            value=cow_max_residual_fraction,
            threshold=0.10,
            operator="<=",
            gate=True,
            note="COW 位相差の観測-予測残差（fraction）は 10% 以下を hard gate。",
            source="cow_experiment_complete_analysis_metrics",
        )
    )
    criteria.append(
        _criterion(
            cid="cow_numeric_coverage_ratio",
            channel="cow",
            proxy="coverage",
            metric=cow_coverage_metric,
            value=cow_coverage_ratio,
            threshold=cow_coverage_threshold,
            operator=">=",
            gate=False,
            note=cow_coverage_note,
            source="cow_experiment_complete_analysis_metrics",
        )
    )

    hom_rows = hom.get("rows") if isinstance(hom.get("rows"), list) else []
    hom_v13 = _find_row(hom_rows, "hom_visibility_d13ns")
    hom_v1000 = _find_row(hom_rows, "hom_visibility_d1000ns")
    hom_delay = _find_row(hom_rows, "hom_delay_dependence")
    hom_noise = _find_row(hom_rows, "noise_psd_shape")

    criteria.append(
        _criterion(
            cid="hom_visibility_13ns_significance",
            channel="hom",
            proxy="visibility",
            metric="z_vs_classical_0p5",
            value=_as_float((hom_v13 or {}).get("metric_value")),
            threshold=3.0,
            operator=">=",
            gate=True,
            note="HOM dip（13 ns）の古典限界 0.5 からの有意差 z。",
            source="hom_squeezed_light_unified_audit_metrics",
        )
    )
    criteria.append(
        _criterion(
            cid="hom_visibility_1000ns_significance",
            channel="hom",
            proxy="visibility",
            metric="z_vs_classical_0p5",
            value=_as_float((hom_v1000 or {}).get("metric_value")),
            threshold=3.0,
            operator=">=",
            gate=True,
            note="HOM dip（1 us）の古典限界 0.5 からの有意差 z。",
            source="hom_squeezed_light_unified_audit_metrics",
        )
    )
    criteria.append(
        _criterion(
            cid="hom_delay_stability",
            channel="hom",
            proxy="phase",
            metric="z_delta_between_13ns_1us",
            value=_as_float((hom_delay or {}).get("metric_value")),
            threshold=3.0,
            operator="<=",
            gate=True,
            note="13 ns ↔ 1 us の可視度差分 z は 3 以下を hard gate。",
            source="hom_squeezed_light_unified_audit_metrics",
        )
    )
    criteria.append(
        _criterion(
            cid="hom_noise_psd_shape",
            channel="hom",
            proxy="noise",
            metric="lf_to_hf_ratio",
            value=_as_float((hom_noise or {}).get("metric_value")),
            threshold=1.0,
            operator=">=",
            gate=False,
            note="低周波/高周波 PSD 比は運用監視（selectionや装置ドリフトの監視項目）。",
            source="hom_squeezed_light_unified_audit_metrics",
        )
    )

    matter_rows = matter.get("rows") if isinstance(matter.get("rows"), list) else []
    precision_gap_watch = matter.get("precision_gap_watch") if isinstance(matter.get("precision_gap_watch"), dict) else {}
    atom_alpha = _find_row(matter_rows, "atom_recoil_alpha")
    atom_precision = _find_row(matter_rows, "atom_interferometer_precision")
    atom_precision_ratio = _as_float((atom_precision or {}).get("metric_value"))
    if atom_precision_ratio is None:
        atom_precision_ratio = _as_float(precision_gap_watch.get("median_ratio"))
    if atom_alpha and atom_precision_ratio is not None:
        atom_section = {
            "atom_alpha_abs_z": _as_float(atom_alpha.get("metric_value")),
            "atom_precision_ratio": atom_precision_ratio,
        }
    else:
        atom_section = {}

    hard_fail = [c["id"] for c in criteria if c.get("gate") and c.get("pass") is False]
    hard_unknown = [c["id"] for c in criteria if c.get("gate") and c.get("pass") is None]
    soft_watch = [c["id"] for c in criteria if not c.get("gate") and c.get("pass") is not True]

    route_a_gate = "A_reject" if hard_fail or hard_unknown else "A_continue"
    transition = "A_to_B" if route_a_gate == "A_reject" else "A_stay"

    channel_summary: Dict[str, Dict[str, int]] = {}
    for row in criteria:
        channel = str(row.get("channel") or "unknown")
        status = str(row.get("status") or "watch")
        bucket = channel_summary.setdefault(channel, {"pass": 0, "watch": 0, "reject": 0})
        if status not in bucket:
            status = "watch"
        bucket[status] += 1

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 7, "step": "7.21.3", "name": "Born 2.6.2 A/B cross-channel gate"},
        "intent": "Apply the same A/B gate logic to COW/atom/HOM proxies with one machine-readable decision.",
        "inputs": {
            "born_route_a_proxy_constraints_pack_json": _rel(born_pack_json),
            "cow_experiment_complete_analysis_metrics_json": _rel(cow_metrics_json),
            "matter_wave_interference_precision_audit_metrics_json": _rel(matter_metrics_json),
            "hom_squeezed_light_unified_audit_metrics_json": _rel(hom_metrics_json),
        },
        "criteria": criteria,
        "decision": {
            "route_a_gate": route_a_gate,
            "transition": transition,
            "hard_fail_ids": hard_fail,
            "hard_unknown_ids": hard_unknown,
            "watchlist": soft_watch,
            "rule": "A_reject if any hard gate fails/unknown; otherwise A_continue.",
        },
        "channel_summary": channel_summary,
        "diagnostics": {
            "atom_proxy_snapshot": atom_section,
            "cow_coverage_snapshot": {
                "targeted_ratio": cow_coverage_ratio_targeted,
                "legacy_ratio": cow_coverage_ratio_legacy,
                "selected_metric": cow_coverage_metric,
                "selected_threshold": cow_coverage_threshold,
            },
            "criteria_n": len(criteria),
        },
    }


def _write_csv(path: Path, criteria: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "channel",
                "proxy",
                "metric",
                "value",
                "threshold",
                "operator",
                "pass",
                "gate",
                "status",
                "normalized_score",
                "source",
                "note",
            ],
        )
        writer.writeheader()
        for row in criteria:
            writer.writerow(row)


def _plot(path: Path, payload: Dict[str, Any]) -> None:
    criteria = payload.get("criteria") if isinstance(payload.get("criteria"), list) else []
    labels: List[str] = []
    scores: List[float] = []
    colors: List[str] = []
    for row in criteria:
        if not isinstance(row, dict):
            continue
        labels.append(str(row.get("id") or ""))
        score = _as_float(row.get("normalized_score"))
        scores.append(score if score is not None else math.nan)
        status = str(row.get("status") or "watch")
        if status == "pass":
            colors.append("#2f9e44")
        elif status == "reject":
            colors.append("#e03131")
        else:
            colors.append("#f2c94c")

    fig_height = max(4.8, 0.33 * len(labels) + 1.4)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.0, fig_height), dpi=180)
    ax.barh(y, scores, color=colors)
    ax.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("normalized score (<=1 means threshold satisfied)")
    decision = str(payload.get("decision", {}).get("route_a_gate") or "unknown")
    ax.set_title(f"Born 2.6.2 A/B cross-channel gate ({decision})")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Born 2.6.2 A/B gate to COW/atom/HOM proxies and freeze one transition decision.")
    parser.add_argument(
        "--born-pack",
        default=str(ROOT / "output" / "public" / "quantum" / "born_route_a_proxy_constraints_pack.json"),
        help="Input Born route-A proxy pack JSON.",
    )
    parser.add_argument(
        "--cow-metrics",
        default=str(ROOT / "output" / "public" / "quantum" / "cow_experiment_complete_analysis_metrics.json"),
        help="Input COW complete analysis metrics JSON.",
    )
    parser.add_argument(
        "--matter-metrics",
        default=str(ROOT / "output" / "public" / "quantum" / "matter_wave_interference_precision_audit_metrics.json"),
        help="Input matter-wave interference precision audit metrics JSON.",
    )
    parser.add_argument(
        "--hom-metrics",
        default=str(ROOT / "output" / "public" / "quantum" / "hom_squeezed_light_unified_audit_metrics.json"),
        help="Input HOM/squeezed-light unified audit metrics JSON.",
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_born_ab_gate.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_born_ab_gate.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_born_ab_gate.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    def _resolve(path_text: str) -> Path:
        path = Path(path_text)
        if path.is_absolute():
            return path.resolve()
        return (ROOT / path).resolve()

    born_pack = _resolve(args.born_pack)
    cow_metrics = _resolve(args.cow_metrics)
    matter_metrics = _resolve(args.matter_metrics)
    hom_metrics = _resolve(args.hom_metrics)
    out_json = _resolve(args.out_json)
    out_csv = _resolve(args.out_csv)
    out_png = _resolve(args.out_png)

    for input_path in [born_pack, cow_metrics, matter_metrics, hom_metrics]:
        if not input_path.exists():
            raise FileNotFoundError(f"required input not found: {_rel(input_path)}")

    payload = build_payload(
        born_pack_json=born_pack,
        cow_metrics_json=cow_metrics,
        matter_metrics_json=matter_metrics,
        hom_metrics_json=hom_metrics,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    criteria = payload.get("criteria") if isinstance(payload.get("criteria"), list) else []
    _write_csv(out_csv, criteria)
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_connection_born_ab_gate",
                "phase": "7.21.3",
                "inputs": payload.get("inputs"),
                "outputs": {
                    "quantum_connection_born_ab_gate_json": _rel(out_json),
                    "quantum_connection_born_ab_gate_csv": _rel(out_csv),
                    "quantum_connection_born_ab_gate_png": _rel(out_png),
                },
                "decision": payload.get("decision"),
                "channel_summary": payload.get("channel_summary"),
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
