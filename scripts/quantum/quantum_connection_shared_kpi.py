#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_connection_shared_kpi.py

Step 7.21.1:
Bell / 干渉 / 物性・熱 holdout を共通KPI（pass/watch/reject）へ射影し、
量子接続の横断判定を固定出力する。

出力:
  - output/public/quantum/quantum_connection_shared_kpi.json
  - output/public/quantum/quantum_connection_shared_kpi.csv
  - output/public/quantum/quantum_connection_shared_kpi.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402

SEVERITY = {"pass": 0, "watch": 1, "reject": 2}
STATUS_BY_SEVERITY = {0: "pass", 1: "watch", 2: "reject"}


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(value: Any) -> Optional[float]:
    # 条件分岐: `isinstance(value, (int, float))` を満たす経路を評価する。
    if isinstance(value, (int, float)):
        number = float(value)
        # 条件分岐: `math.isfinite(number)` を満たす経路を評価する。
        if math.isfinite(number):
            return number

    return None


# 関数: `_safe_div` の入出力契約と処理意図を定義する。

def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    # 条件分岐: `num is None or den is None or den == 0.0` を満たす経路を評価する。
    if num is None or den is None or den == 0.0:
        return None

    return float(num) / float(den)


# 関数: `_count_status` の入出力契約と処理意図を定義する。

def _count_status(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"pass": 0, "watch": 0, "reject": 0}
    for row in rows:
        status = str(row.get("status") or "reject")
        # 条件分岐: `status not in counts` を満たす経路を評価する。
        if status not in counts:
            status = "reject"

        counts[status] += 1

    return counts


# 関数: `_channel_bell` の入出力契約と処理意図を定義する。

def _channel_bell(part3_audit: Dict[str, Any], bell_pack: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = bell_pack.get("thresholds") if isinstance(bell_pack.get("thresholds"), dict) else {}
    selection_thr = _as_float(thresholds.get("selection_origin_ratio_min")) or 1.0
    delay_thr = _as_float(thresholds.get("delay_signature_z_min")) or 3.0
    pairing_thr = 1.0

    cross = bell_pack.get("cross_dataset") if isinstance(bell_pack.get("cross_dataset"), dict) else {}
    cross_summary = cross.get("summary") if isinstance(cross.get("summary"), dict) else {}
    selection_min = _as_float(cross_summary.get("selection_ratio_min"))
    delay_fast_min = _as_float(cross_summary.get("delay_z_fast_min"))

    pairing = cross.get("pairing_crosscheck_summary") if isinstance(cross.get("pairing_crosscheck_summary"), dict) else {}
    pairing_rows = pairing.get("datasets") if isinstance(pairing.get("datasets"), list) else []
    delta_sigma: List[float] = []
    for row in pairing_rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        # 条件分岐: `not bool(row.get("supported"))` を満たす経路を評価する。

        if not bool(row.get("supported")):
            continue

        delta = row.get("delta") if isinstance(row.get("delta"), dict) else {}
        value = _as_float(delta.get("delta_over_sigma_boot"))
        # 条件分岐: `value is not None` を満たす経路を評価する。
        if value is not None:
            delta_sigma.append(abs(value))

    pairing_max = max(delta_sigma) if delta_sigma else None

    bell_gate = (
        part3_audit.get("gates", {}).get("bell", {})
        if isinstance(part3_audit.get("gates"), dict)
        else {}
    )
    dataset_gates = bell_gate.get("dataset_gates") if isinstance(bell_gate.get("dataset_gates"), list) else []
    delay_defined_n = 0
    delay_fail_n = 0
    for row in dataset_gates:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        flag = row.get("delay_z_ge_threshold")
        # 条件分岐: `isinstance(flag, bool)` を満たす経路を評価する。
        if isinstance(flag, bool):
            delay_defined_n += 1
            # 条件分岐: `not flag` を満たす経路を評価する。
            if not flag:
                delay_fail_n += 1

    selection_ok = selection_min is not None and selection_min >= selection_thr
    delay_ok = delay_fast_min is not None and delay_fast_min >= delay_thr
    pairing_ok = pairing_max is not None and pairing_max <= pairing_thr

    # 条件分岐: `selection_ok and pairing_ok and delay_ok` を満たす経路を評価する。
    if selection_ok and pairing_ok and delay_ok:
        status = "pass"
    # 条件分岐: 前段条件が不成立で、`selection_ok and pairing_ok` を追加評価する。
    elif selection_ok and pairing_ok:
        status = "watch"
    else:
        status = "reject"

    notes: List[str] = []
    # 条件分岐: `delay_defined_n > 0 and delay_fail_n > 0` を満たす経路を評価する。
    if delay_defined_n > 0 and delay_fail_n > 0:
        notes.append(f"delay gate fail in {delay_fail_n}/{delay_defined_n} datasets (time-tag subsets still tracked by delay_fast_min).")

    return {
        "channel": "bell",
        "title": "Bell cross-dataset connection",
        "status": status,
        "severity": SEVERITY[status],
        "kpi": {
            "selection_ratio_min": selection_min,
            "selection_ratio_threshold": selection_thr,
            "delay_fast_z_min": delay_fast_min,
            "delay_z_threshold": delay_thr,
            "pairing_delta_sigma_max": pairing_max,
            "pairing_delta_sigma_threshold": pairing_thr,
            "selection_score": _safe_div(selection_thr, selection_min),
            "delay_score": _safe_div(delay_thr, delay_fast_min),
            "pairing_score": _safe_div(pairing_max, pairing_thr),
        },
        "diagnostics": {
            "dataset_n": len(dataset_gates),
            "delay_defined_n": delay_defined_n,
            "delay_fail_n": delay_fail_n,
            "selection_ok": selection_ok,
            "delay_ok": delay_ok,
            "pairing_ok": pairing_ok,
        },
        "notes": notes,
    }


# 関数: `_channel_interference` の入出力契約と処理意図を定義する。

def _channel_interference(interference_metrics: Dict[str, Any]) -> Dict[str, Any]:
    rows = interference_metrics.get("rows") if isinstance(interference_metrics.get("rows"), list) else []
    precision_gap_watch = (
        interference_metrics.get("precision_gap_watch")
        if isinstance(interference_metrics.get("precision_gap_watch"), dict)
        else {}
    )
    total = len(rows)
    pass_n = 0
    fail_n = 0
    precision_gap_fail_n = 0
    non_precision_fail_n = 0
    max_fail_over_thr: Optional[float] = None
    fail_details: List[Dict[str, Any]] = []

    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        passed = bool(row.get("pass_3sigma"))
        metric_value = _as_float(row.get("metric_value"))
        threshold = _as_float(row.get("threshold_3sigma"))
        # 条件分岐: `passed` を満たす経路を評価する。
        if passed:
            pass_n += 1
            continue

        fail_n += 1
        ratio = _safe_div(metric_value, threshold)
        # 条件分岐: `ratio is not None` を満たす経路を評価する。
        if ratio is not None:
            max_fail_over_thr = ratio if max_fail_over_thr is None else max(max_fail_over_thr, ratio)

        observable = str(row.get("observable") or "")
        metric_name = str(row.get("metric_name") or "")
        is_precision_gap = observable == "current_over_required_precision_ratio" or metric_name.endswith("ratio")
        # 条件分岐: `is_precision_gap` を満たす経路を評価する。
        if is_precision_gap:
            precision_gap_fail_n += 1
        else:
            non_precision_fail_n += 1

        fail_details.append(
            {
                "channel": row.get("channel"),
                "observable": observable,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "threshold_3sigma": threshold,
            }
        )

    # 条件分岐: `fail_n == 0` を満たす経路を評価する。

    if fail_n == 0:
        status = "pass"
    # 条件分岐: 前段条件が不成立で、`non_precision_fail_n == 0` を追加評価する。
    elif non_precision_fail_n == 0:
        status = "watch"
    else:
        status = "reject"

    notes: List[str] = []
    # 条件分岐: `precision_gap_fail_n > 0 and non_precision_fail_n == 0` を満たす経路を評価する。
    if precision_gap_fail_n > 0 and non_precision_fail_n == 0:
        notes.append("all fails are precision-gap type (sensitivity shortage), not direct inconsistency.")

    return {
        "channel": "interference",
        "title": "Matter-wave / interference connection",
        "status": status,
        "severity": SEVERITY[status],
        "kpi": {
            "rows_total": total,
            "rows_pass": pass_n,
            "rows_fail": fail_n,
            "fail_ratio": _safe_div(float(fail_n), float(total)) if total > 0 else None,
            "max_fail_over_threshold": max_fail_over_thr,
        },
        "diagnostics": {
            "precision_gap_fail_n": precision_gap_fail_n,
            "non_precision_fail_n": non_precision_fail_n,
            "fail_rows": fail_details,
            "precision_gap_watch": {
                "median_ratio": _as_float(precision_gap_watch.get("median_ratio")),
                "min_ratio": _as_float(precision_gap_watch.get("min_ratio")),
                "max_ratio": _as_float(precision_gap_watch.get("max_ratio")),
                "pass_if_median_le_threshold": precision_gap_watch.get("pass_if_median_le_threshold"),
            }
            if precision_gap_watch
            else {},
        },
        "notes": notes,
    }


# 関数: `_channel_condensed` の入出力契約と処理意図を定義する。

def _channel_condensed(condensed_summary: Dict[str, Any]) -> Dict[str, Any]:
    summary = condensed_summary.get("summary") if isinstance(condensed_summary.get("summary"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []

    ok_n = 0
    reject_n = 0
    inconclusive_n = 0
    reject_rows: List[Tuple[str, Optional[float]]] = []
    excluded_rows: List[Dict[str, str]] = []

    for dataset in datasets:
        # 条件分岐: `not isinstance(dataset, dict)` を満たす経路を評価する。
        if not isinstance(dataset, dict):
            continue

        name = str(dataset.get("dataset") or "")
        # 条件分岐: `not bool(dataset.get("kpi_include", True))` を満たす経路を評価する。
        if not bool(dataset.get("kpi_include", True)):
            excluded_rows.append(
                {
                    "dataset": name,
                    "reason": str(dataset.get("kpi_exclusion_reason") or ""),
                }
            )
            continue

        audit = dataset.get("audit_gates") if isinstance(dataset.get("audit_gates"), dict) else {}
        falsification = audit.get("falsification") if isinstance(audit.get("falsification"), dict) else {}
        status = str(falsification.get("status") or "")
        worst = (
            audit.get("recommended_model_by_minimax_test_max_abs_z", {}).get("worst_test_max_abs_z")
            if isinstance(audit.get("recommended_model_by_minimax_test_max_abs_z"), dict)
            else None
        )
        worst_value = _as_float(worst)
        # 条件分岐: `status == "ok"` を満たす経路を評価する。
        if status == "ok":
            ok_n += 1
        # 条件分岐: 前段条件が不成立で、`status == "reject"` を追加評価する。
        elif status == "reject":
            reject_n += 1
            reject_rows.append((name, worst_value))
        else:
            inconclusive_n += 1

    total = ok_n + reject_n + inconclusive_n
    reject_ratio = _safe_div(float(reject_n), float(total)) if total > 0 else None
    reject_ratio_thr = 0.10

    # 条件分岐: `reject_n == 0 and inconclusive_n == 0` を満たす経路を評価する。
    if reject_n == 0 and inconclusive_n == 0:
        status = "pass"
    # 条件分岐: 前段条件が不成立で、`reject_ratio is not None and reject_ratio <= reject_ratio_thr and inconclusiv...` を追加評価する。
    elif reject_ratio is not None and reject_ratio <= reject_ratio_thr and inconclusive_n == 0:
        status = "watch"
    else:
        status = "reject"

    reject_rows_sorted = sorted(
        reject_rows,
        key=lambda item: item[1] if item[1] is not None else -math.inf,
        reverse=True,
    )
    top_reject = [{"dataset": name, "worst_test_max_abs_z": value} for name, value in reject_rows_sorted[:5]]

    notes: List[str] = []
    # 条件分岐: `reject_n > 0` を満たす経路を評価する。
    if reject_n > 0:
        notes.append(f"{reject_n} holdout datasets are reject (top offenders listed in diagnostics).")

    # 条件分岐: `excluded_rows` を満たす経路を評価する。

    if excluded_rows:
        notes.append(f"{len(excluded_rows)} dataset(s) are excluded from KPI gating (diagnostic-only).")

    return {
        "channel": "condensed_thermal_holdout",
        "title": "Condensed / thermal holdout connection",
        "status": status,
        "severity": SEVERITY[status],
        "kpi": {
            "datasets_total": total,
            "ok_n": ok_n,
            "reject_n": reject_n,
            "inconclusive_n": inconclusive_n,
            "reject_ratio": reject_ratio,
            "reject_ratio_watch_threshold": reject_ratio_thr,
            "excluded_n": int(len(excluded_rows)),
        },
        "diagnostics": {
            "top_reject": top_reject,
            "excluded_from_kpi": excluded_rows,
        },
        "notes": notes,
    }


# 関数: `build_payload` の入出力契約と処理意図を定義する。

def build_payload(
    *,
    part3_audit_json: Path,
    bell_pack_json: Path,
    interference_metrics_json: Path,
    condensed_holdout_summary_json: Path,
) -> Dict[str, Any]:
    part3_audit = _read_json(part3_audit_json)
    bell_pack = _read_json(bell_pack_json)
    interference = _read_json(interference_metrics_json)
    condensed = _read_json(condensed_holdout_summary_json)

    channels = [
        _channel_bell(part3_audit=part3_audit, bell_pack=bell_pack),
        _channel_interference(interference_metrics=interference),
        _channel_condensed(condensed_summary=condensed),
    ]

    max_severity = max((int(row.get("severity", 2)) for row in channels), default=2)
    overall_status = STATUS_BY_SEVERITY.get(max_severity, "reject")

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 7, "step": "7.21.1", "name": "Quantum connection shared KPI packaging"},
        "intent": "Project Bell/interference/condensed-holdout outputs onto a shared pass/watch/reject gate.",
        "inputs": {
            "part3_audit_summary_json": _rel(part3_audit_json),
            "bell_falsification_pack_json": _rel(bell_pack_json),
            "matter_wave_interference_precision_audit_metrics_json": _rel(interference_metrics_json),
            "condensed_holdout_audit_summary_json": _rel(condensed_holdout_summary_json),
        },
        "thresholds": {
            "status_order": ["pass", "watch", "reject"],
            "overall_rule": "worst_channel_severity",
            "interference_watch_rule": "all failures are precision-gap type",
            "condensed_watch_rule": "reject_ratio <= 0.10 with no inconclusive datasets",
        },
        "channels": channels,
        "overall": {
            "status": overall_status,
            "severity": max_severity,
            "status_counts": _count_status(channels),
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(out_csv: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("channels") if isinstance(payload.get("channels"), list) else []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "channel",
                "title",
                "status",
                "severity",
                "kpi_primary_name",
                "kpi_primary_value",
                "kpi_primary_threshold",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            kpi = row.get("kpi") if isinstance(row.get("kpi"), dict) else {}
            # 条件分岐: `row.get("channel") == "bell"` を満たす経路を評価する。
            if row.get("channel") == "bell":
                primary_name = "selection_ratio_min"
                primary_value = kpi.get("selection_ratio_min")
                primary_thr = kpi.get("selection_ratio_threshold")
            # 条件分岐: 前段条件が不成立で、`row.get("channel") == "interference"` を追加評価する。
            elif row.get("channel") == "interference":
                primary_name = "fail_ratio"
                primary_value = kpi.get("fail_ratio")
                primary_thr = 0.0
            else:
                primary_name = "reject_ratio"
                primary_value = kpi.get("reject_ratio")
                primary_thr = kpi.get("reject_ratio_watch_threshold")

            notes = row.get("notes") if isinstance(row.get("notes"), list) else []
            writer.writerow(
                {
                    "channel": row.get("channel"),
                    "title": row.get("title"),
                    "status": row.get("status"),
                    "severity": row.get("severity"),
                    "kpi_primary_name": primary_name,
                    "kpi_primary_value": primary_value,
                    "kpi_primary_threshold": primary_thr,
                    "note": " | ".join(str(x) for x in notes),
                }
            )


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(out_png: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("channels") if isinstance(payload.get("channels"), list) else []
    labels = []
    values = []
    colors = []
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        labels.append(str(row.get("title") or row.get("channel") or "unknown"))
        sev = int(row.get("severity", 2))
        values.append(sev)
        # 条件分岐: `sev == 0` を満たす経路を評価する。
        if sev == 0:
            colors.append("#2f9e44")
        # 条件分岐: 前段条件が不成立で、`sev == 1` を追加評価する。
        elif sev == 1:
            colors.append("#f2c94c")
        else:
            colors.append("#e03131")

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 2.2)
    ax.set_xticks([0, 1, 2], ["pass", "watch", "reject"])
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    overall_status = payload.get("overall", {}).get("status", "unknown")
    ax.set_title(f"Quantum connection shared KPI (overall={overall_status})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build shared KPI pack for quantum connection reinforcement (Step 7.21.1).")
    parser.add_argument(
        "--part3-audit",
        default=str(ROOT / "output" / "public" / "summary" / "part3_audit_summary.json"),
        help="Input JSON from part3 audit summary.",
    )
    parser.add_argument(
        "--bell-pack",
        default=str(ROOT / "output" / "public" / "quantum" / "bell" / "falsification_pack.json"),
        help="Input Bell falsification pack JSON.",
    )
    parser.add_argument(
        "--interference-metrics",
        default=str(ROOT / "output" / "public" / "quantum" / "matter_wave_interference_precision_audit_metrics.json"),
        help="Input matter-wave interference metrics JSON.",
    )
    parser.add_argument(
        "--condensed-summary",
        default=str(ROOT / "output" / "public" / "quantum" / "condensed_holdout_audit_summary.json"),
        help="Input condensed/thermal holdout summary JSON.",
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_shared_kpi.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_shared_kpi.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_shared_kpi.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    part3_audit_path = Path(args.part3_audit).resolve() if Path(args.part3_audit).is_absolute() else (ROOT / args.part3_audit).resolve()
    bell_pack_path = Path(args.bell_pack).resolve() if Path(args.bell_pack).is_absolute() else (ROOT / args.bell_pack).resolve()
    interference_path = Path(args.interference_metrics).resolve() if Path(args.interference_metrics).is_absolute() else (ROOT / args.interference_metrics).resolve()
    condensed_path = Path(args.condensed_summary).resolve() if Path(args.condensed_summary).is_absolute() else (ROOT / args.condensed_summary).resolve()
    out_json = Path(args.out_json).resolve() if Path(args.out_json).is_absolute() else (ROOT / args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve() if Path(args.out_csv).is_absolute() else (ROOT / args.out_csv).resolve()
    out_png = Path(args.out_png).resolve() if Path(args.out_png).is_absolute() else (ROOT / args.out_png).resolve()

    for path in [part3_audit_path, bell_pack_path, interference_path, condensed_path]:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            raise FileNotFoundError(f"required input not found: {_rel(path)}")

    payload = build_payload(
        part3_audit_json=part3_audit_path,
        bell_pack_json=bell_pack_path,
        interference_metrics_json=interference_path,
        condensed_holdout_summary_json=condensed_path,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, payload)
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_connection_shared_kpi",
                "phase": "7.21.1",
                "inputs": payload.get("inputs"),
                "outputs": {
                    "quantum_connection_shared_kpi_json": _rel(out_json),
                    "quantum_connection_shared_kpi_csv": _rel(out_csv),
                    "quantum_connection_shared_kpi_png": _rel(out_png),
                },
                "overall": payload.get("overall"),
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
