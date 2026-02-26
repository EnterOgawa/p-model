#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_connection_bridge_table.py

Step 7.21.2:
Bell の window/offset 感度と、干渉側の phase/visibility 感度を
同一テーブルへ統合し、selection由来と物理由来を行単位で分離して固定する。

出力:
  - output/public/quantum/quantum_connection_bridge_table.json
  - output/public/quantum/quantum_connection_bridge_table.csv
  - output/public/quantum/quantum_connection_bridge_table.png
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

SEVERITY = {"pass": 0, "watch": 1, "reject": 2}
STATUS_FROM_SEVERITY = {0: "pass", 1: "watch", 2: "reject"}


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
    # 条件分岐: `isinstance(value, (int, float))` を満たす経路を評価する。
    if isinstance(value, (int, float)):
        number = float(value)
        # 条件分岐: `math.isfinite(number)` を満たす経路を評価する。
        if math.isfinite(number):
            return number

    return None


def _normalized_score(value: Optional[float], threshold: Optional[float], operator: str) -> Optional[float]:
    # 条件分岐: `value is None or threshold is None or threshold == 0.0` を満たす経路を評価する。
    if value is None or threshold is None or threshold == 0.0:
        return None

    # 条件分岐: `operator == "<="` を満たす経路を評価する。

    if operator == "<=":
        return float(value) / float(threshold)

    # 条件分岐: `operator == ">="` を満たす経路を評価する。

    if operator == ">=":
        # 条件分岐: `value == 0.0` を満たす経路を評価する。
        if value == 0.0:
            return math.inf

        return float(threshold) / float(value)

    return None


def _pass_value(value: Optional[float], threshold: Optional[float], operator: str) -> Optional[bool]:
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


def _row_status(*, passed: Optional[bool], hard_gate: bool) -> str:
    # 条件分岐: `passed is True` を満たす経路を評価する。
    if passed is True:
        return "pass"

    # 条件分岐: `passed is False and hard_gate` を満たす経路を評価する。

    if passed is False and hard_gate:
        return "reject"

    # 条件分岐: `passed is False and not hard_gate` を満たす経路を評価する。

    if passed is False and not hard_gate:
        return "watch"

    return "watch"


def _group_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"pass": 0, "watch": 0, "reject": 0}
    severities: List[int] = []
    scores: List[float] = []
    for row in rows:
        status = str(row.get("status") or "watch")
        # 条件分岐: `status not in counts` を満たす経路を評価する。
        if status not in counts:
            status = "watch"

        counts[status] += 1
        severities.append(SEVERITY.get(status, 1))
        score = _as_float(row.get("normalized_score"))
        # 条件分岐: `score is not None` を満たす経路を評価する。
        if score is not None:
            scores.append(score)

    severity = max(severities) if severities else 1
    return {
        "rows_n": len(rows),
        "status": STATUS_FROM_SEVERITY.get(severity, "watch"),
        "status_counts": counts,
        "normalized_score_median": float(np.median(scores)) if scores else None,
        "normalized_score_max": max(scores) if scores else None,
    }


def _format_knob(knob: str) -> str:
    # 条件分岐: `knob == "window_ns"` を満たす経路を評価する。
    if knob == "window_ns":
        return "window half-width"

    # 条件分岐: `knob == "event_ready_offset_ps"` を満たす経路を評価する。

    if knob == "event_ready_offset_ps":
        return "event-ready start offset"

    return knob


def _is_fast_switching_delay_hard_gate(dataset_id: str) -> bool:
    return dataset_id.startswith("weihs1998_") or dataset_id.startswith("nist_")


def build_payload(
    *,
    part3_audit_json: Path,
    born_proxy_json: Path,
    kwiat_watch_audit_json: Optional[Path] = None,
    hom_noise_watch_audit_json: Optional[Path] = None,
) -> Dict[str, Any]:
    audit = _read_json(part3_audit_json)
    born = _read_json(born_proxy_json)
    kwiat_watch_audit: Dict[str, Any] = {}
    hom_noise_watch_audit: Dict[str, Any] = {}
    # 条件分岐: `kwiat_watch_audit_json is not None and kwiat_watch_audit_json.exists()` を満たす経路を評価する。
    if kwiat_watch_audit_json is not None and kwiat_watch_audit_json.exists():
        kwiat_watch_audit = _read_json(kwiat_watch_audit_json)

    # 条件分岐: `hom_noise_watch_audit_json is not None and hom_noise_watch_audit_json.exists()` を満たす経路を評価する。

    if hom_noise_watch_audit_json is not None and hom_noise_watch_audit_json.exists():
        hom_noise_watch_audit = _read_json(hom_noise_watch_audit_json)

    rows: List[Dict[str, Any]] = []

    bell_gate = (
        audit.get("gates", {}).get("bell", {})
        if isinstance(audit.get("gates"), dict)
        else {}
    )
    dataset_gates = bell_gate.get("dataset_gates") if isinstance(bell_gate.get("dataset_gates"), list) else []
    for gate in dataset_gates:
        # 条件分岐: `not isinstance(gate, dict)` を満たす経路を評価する。
        if not isinstance(gate, dict):
            continue

        dataset_id = str(gate.get("dataset_id") or "")
        knob = str(gate.get("selection_knob") or "")

        ratio = _as_float(gate.get("ratio_sys_stat"))
        ratio_thr = 1.0
        ratio_op = ">="
        ratio_pass = _pass_value(ratio, ratio_thr, ratio_op)
        rows.append(
            {
                "id": f"{dataset_id}:selection_ratio",
                "origin": "selection-driven",
                "domain": "bell",
                "observable": _format_knob(knob),
                "metric": "ratio_sys_stat",
                "value": ratio,
                "threshold": ratio_thr,
                "operator": ratio_op,
                "normalized_score": _normalized_score(ratio, ratio_thr, ratio_op),
                "status": _row_status(passed=ratio_pass, hard_gate=True),
                "hard_gate": True,
                "source": "part3_audit bell.dataset_gates",
                "note": "Selection sweep幅 / 統計幅（σ）",
            }
        )

        delay = _as_float(gate.get("delay_z"))
        # 条件分岐: `delay is not None` を満たす経路を評価する。
        if delay is not None:
            delay_thr = 3.0
            delay_op = ">="
            delay_pass = _pass_value(delay, delay_thr, delay_op)
            delay_hard_gate = _is_fast_switching_delay_hard_gate(dataset_id)
            delay_note = "Setting依存遅延の有意度（z）。fast-switch 系（Weihs/NIST）は hard gate、その他は watch 扱い。"
            extra = {}
            # 条件分岐: `"kwiat2013_" in dataset_id and isinstance(kwiat_watch_audit, dict)` を満たす経路を評価する。
            if "kwiat2013_" in dataset_id and isinstance(kwiat_watch_audit, dict):
                kw_summary = kwiat_watch_audit.get("summary") if isinstance(kwiat_watch_audit.get("summary"), dict) else {}
                kw_decision = str(kw_summary.get("decision") or "")
                kw_hard = kw_summary.get("hard_gate_applicable")
                # 条件分岐: `kw_decision` を満たす経路を評価する。
                if kw_decision:
                    delay_note += f" watch監査: {kw_decision}."

                # 条件分岐: `isinstance(kw_hard, bool)` を満たす経路を評価する。

                if isinstance(kw_hard, bool):
                    extra["watch_hard_gate_applicable"] = kw_hard

                max_any = _as_float(kw_summary.get("max_abs_z_any"))
                # 条件分岐: `max_any is not None` を満たす経路を評価する。
                if max_any is not None:
                    extra["watch_max_abs_z_any"] = max_any

            rows.append(
                {
                    "id": f"{dataset_id}:delay_signature",
                    "origin": "selection-driven",
                    "domain": "bell",
                    "observable": "time-tag delay signature",
                    "metric": "delay_z",
                    "value": delay,
                    "threshold": delay_thr,
                    "operator": delay_op,
                    "normalized_score": _normalized_score(delay, delay_thr, delay_op),
                    "status": _row_status(passed=delay_pass, hard_gate=delay_hard_gate),
                    "hard_gate": delay_hard_gate,
                    "source": "part3_audit bell.dataset_gates",
                    "note": delay_note,
                    **extra,
                }
            )

    criteria = born.get("criteria") if isinstance(born.get("criteria"), list) else []
    for criterion in criteria:
        # 条件分岐: `not isinstance(criterion, dict)` を満たす経路を評価する。
        if not isinstance(criterion, dict):
            continue

        proxy = str(criterion.get("proxy") or "")
        # 条件分岐: `proxy not in ("phase", "visibility")` を満たす経路を評価する。
        if proxy not in ("phase", "visibility"):
            continue

        value = _as_float(criterion.get("value"))
        threshold = _as_float(criterion.get("threshold"))
        operator = str(criterion.get("operator") or "")
        passed = _pass_value(value, threshold, operator)
        hard_gate = bool(criterion.get("gate"))
        rows.append(
            {
                "id": str(criterion.get("id") or ""),
                "origin": "physics-driven",
                "domain": "interference",
                "observable": proxy,
                "metric": str(criterion.get("metric") or ""),
                "value": value,
                "threshold": threshold,
                "operator": operator,
                "normalized_score": _normalized_score(value, threshold, operator),
                "status": _row_status(passed=passed, hard_gate=hard_gate),
                "hard_gate": hard_gate,
                "source": "born_route_a_proxy_constraints_pack",
                "note": str(criterion.get("note") or ""),
            }
        )

    # 条件分岐: `isinstance(hom_noise_watch_audit, dict) and hom_noise_watch_audit` を満たす経路を評価する。

    if isinstance(hom_noise_watch_audit, dict) and hom_noise_watch_audit:
        baseline = hom_noise_watch_audit.get("baseline") if isinstance(hom_noise_watch_audit.get("baseline"), dict) else {}
        summary = hom_noise_watch_audit.get("summary") if isinstance(hom_noise_watch_audit.get("summary"), dict) else {}
        thresholds_hom = (
            hom_noise_watch_audit.get("thresholds")
            if isinstance(hom_noise_watch_audit.get("thresholds"), dict)
            else {}
        )
        ratio_value = _as_float(baseline.get("ratio_interp_10k_over_100k"))
        ratio_thr = _as_float(thresholds_hom.get("lf_to_hf_ratio_min")) or 1.0
        ratio_pass = _pass_value(ratio_value, ratio_thr, ">=")
        decision = str(summary.get("decision") or "")
        note = "HOM noise PSD shape（低周波/高周波比）の帯域・drift感度監査。"
        # 条件分岐: `decision` を満たす経路を評価する。
        if decision:
            note += f" watch監査: {decision}."

        rows.append(
            {
                "id": "hom_noise_psd_shape",
                "origin": "physics-driven",
                "domain": "interference",
                "observable": "noise_psd_shape",
                "metric": "lf_to_hf_ratio",
                "value": ratio_value,
                "threshold": ratio_thr,
                "operator": ">=",
                "normalized_score": _normalized_score(ratio_value, ratio_thr, ">="),
                "status": _row_status(passed=ratio_pass, hard_gate=False),
                "hard_gate": False,
                "source": "hom_noise_psd_watch_audit",
                "note": note,
                "watch_hard_gate_applicable": summary.get("hard_gate_applicable"),
                "watch_median_ratio_detrended": _as_float(summary.get("detrended_ratio_median")),
            }
        )

    selection_rows = [row for row in rows if row.get("origin") == "selection-driven"]
    physics_rows = [row for row in rows if row.get("origin") == "physics-driven"]
    selection_summary = _group_summary(selection_rows)
    physics_summary = _group_summary(physics_rows)

    overall_severity = max(
        SEVERITY.get(str(selection_summary.get("status")), 1),
        SEVERITY.get(str(physics_summary.get("status")), 1),
    )
    overall = {
        "status": STATUS_FROM_SEVERITY.get(overall_severity, "watch"),
        "severity": overall_severity,
        "selection_summary": selection_summary,
        "physics_summary": physics_summary,
    }

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 7, "step": "7.21.2", "name": "Bell-interference bridge table packaging"},
        "intent": "Unify window/offset sensitivity and phase/visibility sensitivity in one table with origin tags.",
        "inputs": {
            "part3_audit_summary_json": _rel(part3_audit_json),
            "born_route_a_proxy_constraints_pack_json": _rel(born_proxy_json),
            "kwiat_delay_watch_audit_json": (
                _rel(kwiat_watch_audit_json) if kwiat_watch_audit_json is not None and kwiat_watch_audit_json.exists() else None
            ),
            "hom_noise_watch_audit_json": (
                _rel(hom_noise_watch_audit_json) if hom_noise_watch_audit_json is not None and hom_noise_watch_audit_json.exists() else None
            ),
        },
        "thresholds": {
            "selection_ratio_min": 1.0,
            "delay_z_min": 3.0,
            "phase_abs_z_max": 3.0,
            "visibility_ratio_max": 1.0,
            "overall_rule": "max(selection_status, physics_status)",
        },
        "rows": rows,
        "overall": overall,
        "watch_audit": {
            "kwiat_delay_signature": kwiat_watch_audit.get("summary") if isinstance(kwiat_watch_audit, dict) else {},
            "hom_noise_psd_shape": hom_noise_watch_audit.get("summary") if isinstance(hom_noise_watch_audit, dict) else {},
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "origin",
                "domain",
                "observable",
                "metric",
                "value",
                "threshold",
                "operator",
                "normalized_score",
                "status",
                "hard_gate",
                "source",
                "note",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, rows: List[Dict[str, Any]], overall_status: str) -> None:
    labels = [str(row.get("id") or "") for row in rows]
    scores = []
    colors = []
    for row in rows:
        score = _as_float(row.get("normalized_score"))
        # 条件分岐: `score is None` を満たす経路を評価する。
        if score is None:
            score = math.nan

        scores.append(score)
        status = str(row.get("status") or "watch")
        # 条件分岐: `status == "pass"` を満たす経路を評価する。
        if status == "pass":
            colors.append("#2f9e44")
        # 条件分岐: 前段条件が不成立で、`status == "reject"` を追加評価する。
        elif status == "reject":
            colors.append("#e03131")
        else:
            colors.append("#f2c94c")

    y = np.arange(len(labels))
    fig_height = max(4.8, 0.32 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(12.0, fig_height), dpi=180)
    ax.barh(y, scores, color=colors)
    ax.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("normalized score (<=1 means threshold satisfied)")
    ax.set_title(f"Quantum connection bridge table (overall={overall_status})")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build bridge table between Bell selection sensitivity and interference phase/visibility sensitivity.")
    parser.add_argument(
        "--part3-audit",
        default=str(ROOT / "output" / "public" / "summary" / "part3_audit_summary.json"),
        help="Input part3 audit summary JSON.",
    )
    parser.add_argument(
        "--born-proxy-pack",
        default=str(ROOT / "output" / "public" / "quantum" / "born_route_a_proxy_constraints_pack.json"),
        help="Input born route-A proxy constraints pack JSON.",
    )
    parser.add_argument(
        "--kwiat-watch-audit",
        default=str(ROOT / "output" / "public" / "quantum" / "bell_kwiat_delay_signature_watch_audit.json"),
        help="Optional Kwiat delay-signature watch audit JSON.",
    )
    parser.add_argument(
        "--hom-noise-watch-audit",
        default=str(ROOT / "output" / "public" / "quantum" / "hom_noise_psd_watch_audit.json"),
        help="Optional HOM noise-PSD watch audit JSON.",
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_bridge_table.json"),
        help="Output bridge table JSON.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_bridge_table.csv"),
        help="Output bridge table CSV.",
    )
    parser.add_argument(
        "--out-png",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_bridge_table.png"),
        help="Output bridge table PNG.",
    )
    args = parser.parse_args(argv)

    part3_audit_path = Path(args.part3_audit).resolve() if Path(args.part3_audit).is_absolute() else (ROOT / args.part3_audit).resolve()
    born_proxy_path = Path(args.born_proxy_pack).resolve() if Path(args.born_proxy_pack).is_absolute() else (ROOT / args.born_proxy_pack).resolve()
    kwiat_watch_path = Path(args.kwiat_watch_audit).resolve() if Path(args.kwiat_watch_audit).is_absolute() else (ROOT / args.kwiat_watch_audit).resolve()
    hom_noise_watch_path = Path(args.hom_noise_watch_audit).resolve() if Path(args.hom_noise_watch_audit).is_absolute() else (ROOT / args.hom_noise_watch_audit).resolve()
    out_json = Path(args.out_json).resolve() if Path(args.out_json).is_absolute() else (ROOT / args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve() if Path(args.out_csv).is_absolute() else (ROOT / args.out_csv).resolve()
    out_png = Path(args.out_png).resolve() if Path(args.out_png).is_absolute() else (ROOT / args.out_png).resolve()

    for input_path in [part3_audit_path, born_proxy_path]:
        # 条件分岐: `not input_path.exists()` を満たす経路を評価する。
        if not input_path.exists():
            raise FileNotFoundError(f"required input not found: {_rel(input_path)}")

    payload = build_payload(
        part3_audit_json=part3_audit_path,
        born_proxy_json=born_proxy_path,
        kwiat_watch_audit_json=kwiat_watch_path,
        hom_noise_watch_audit_json=hom_noise_watch_path,
    )
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    overall_status = str(payload.get("overall", {}).get("status") or "watch")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, rows)
    _plot(out_png, rows, overall_status)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_connection_bridge_table",
                "phase": "7.21.2",
                "inputs": payload.get("inputs"),
                "outputs": {
                    "quantum_connection_bridge_table_json": _rel(out_json),
                    "quantum_connection_bridge_table_csv": _rel(out_csv),
                    "quantum_connection_bridge_table_png": _rel(out_png),
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
