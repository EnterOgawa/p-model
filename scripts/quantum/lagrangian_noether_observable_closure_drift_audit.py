#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lagrangian_noether_observable_closure_drift_audit.py

Step 8.7.21.3:
8.7.21.1 で固定した L_total -> EL -> observables 閉包監査に対して、
運用上の drift（gate 逸脱）を機械判定し、再計算トリガーを固定出力する。

出力:
  - output/public/quantum/lagrangian_noether_observable_closure_drift_audit.json
  - output/public/quantum/lagrangian_noether_observable_closure_drift_audit.csv
  - output/public/quantum/lagrangian_noether_observable_closure_drift_audit.png
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

DEFAULT_BASELINE_NOETHER_GAUGE_MARGIN = 4.999993032890784e-08
DEFAULT_BASELINE_NOETHER_REALNESS_MARGIN = 5.0e-10


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
        v = float(value)
        # 条件分岐: `math.isfinite(v)` を満たす経路を評価する。
        if math.isfinite(v):
            return v

    return None


def _check_value(payload: Dict[str, Any], check_id: str) -> Any:
    rows = payload.get("checks")
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        return None

    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        # 条件分岐: `str(row.get("id") or "") == check_id` を満たす経路を評価する。

        if str(row.get("id") or "") == check_id:
            return row.get("value")

    return None


def _count_check_rows(payload: Dict[str, Any]) -> Tuple[int, int]:
    rows = payload.get("checks")
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        return 0, 0

    total = 0
    passed = 0
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        total += 1
        # 条件分岐: `row.get("pass") is True` を満たす経路を評価する。
        if row.get("pass") is True:
            passed += 1

    return total, passed


def _margin_ratio_status(
    *,
    current_margin: Optional[float],
    baseline_margin: float,
    pass_ratio: float,
    watch_ratio: float,
) -> Tuple[str, Optional[float]]:
    # 条件分岐: `current_margin is None` を満たす経路を評価する。
    if current_margin is None:
        return "reject", None

    # 条件分岐: `not math.isfinite(float(current_margin))` を満たす経路を評価する。

    if not math.isfinite(float(current_margin)):
        return "reject", None

    # 条件分岐: `baseline_margin <= 0.0` を満たす経路を評価する。

    if baseline_margin <= 0.0:
        return "reject", None

    ratio = float(current_margin) / float(baseline_margin)
    # 条件分岐: `ratio >= pass_ratio` を満たす経路を評価する。
    if ratio >= pass_ratio:
        return "pass", ratio

    # 条件分岐: `ratio >= watch_ratio` を満たす経路を評価する。

    if ratio >= watch_ratio:
        return "watch", ratio

    return "reject", ratio


def _score_from_status(status: str) -> float:
    # 条件分岐: `status == "pass"` を満たす経路を評価する。
    if status == "pass":
        return 1.0

    # 条件分岐: `status == "watch"` を満たす経路を評価する。

    if status == "watch":
        return 0.5

    return 0.0


def _make_row(
    *,
    cid: str,
    metric: str,
    value: Any,
    expected: Any,
    status: str,
    gate_level: str,
    source: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "id": cid,
        "metric": metric,
        "value": value,
        "expected": expected,
        "status": status,
        "score": _score_from_status(status),
        "gate_level": gate_level,
        "source": source,
        "note": note,
    }


def build_payload(
    *,
    closure_json: Path,
    baseline_noether_gauge_margin: float,
    baseline_noether_realness_margin: float,
    pass_ratio: float,
    watch_ratio: float,
) -> Dict[str, Any]:
    closure = _read_json(closure_json)
    decision = closure.get("decision") if isinstance(closure.get("decision"), dict) else {}
    diagnostics = closure.get("diagnostics") if isinstance(closure.get("diagnostics"), dict) else {}

    overall_status_value = str(decision.get("overall_status") or "")
    hard_fail_ids = list(decision.get("hard_fail_ids") or [])
    watch_ids = list(decision.get("watch_ids") or [])
    route_a_gate = str(decision.get("route_a_gate") or "")
    transition = str(decision.get("transition") or "")
    missing_equations_n = len(diagnostics.get("missing_equations") or [])
    missing_nonrel_channels_n = len(diagnostics.get("missing_nonrel_channels") or [])
    checks_total_n, checks_pass_n = _count_check_rows(closure)

    noether_gauge_margin = _as_float(_check_value(closure, "action::noether_gauge"))
    noether_realness_margin = _as_float(_check_value(closure, "action::noether_realness"))

    noether_gauge_ratio_status, noether_gauge_ratio = _margin_ratio_status(
        current_margin=noether_gauge_margin,
        baseline_margin=float(baseline_noether_gauge_margin),
        pass_ratio=float(pass_ratio),
        watch_ratio=float(watch_ratio),
    )
    noether_realness_ratio_status, noether_realness_ratio = _margin_ratio_status(
        current_margin=noether_realness_margin,
        baseline_margin=float(baseline_noether_realness_margin),
        pass_ratio=float(pass_ratio),
        watch_ratio=float(watch_ratio),
    )

    rows: List[Dict[str, Any]] = [
        _make_row(
            cid="closure_drift::overall_status",
            metric="overall_status",
            value=overall_status_value,
            expected="pass",
            status="pass" if overall_status_value == "pass" else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="閉包監査の全体判定が pass を維持していること。",
        ),
        _make_row(
            cid="closure_drift::hard_fail_ids_n",
            metric="hard_fail_ids_n",
            value=len(hard_fail_ids),
            expected=0,
            status="pass" if len(hard_fail_ids) == 0 else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="hard gate 逸脱が発生していないこと。",
        ),
        _make_row(
            cid="closure_drift::watch_ids_n",
            metric="watch_ids_n",
            value=len(watch_ids),
            expected=0,
            status="pass" if len(watch_ids) == 0 else "watch",
            gate_level="watch",
            source="lagrangian_noether_observable_closure_audit",
            note="watch 逸脱の発生数（運用監視）。",
        ),
        _make_row(
            cid="closure_drift::missing_equations_n",
            metric="missing_equations_n",
            value=missing_equations_n,
            expected=0,
            status="pass" if missing_equations_n == 0 else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="閉包必須式が欠落していないこと。",
        ),
        _make_row(
            cid="closure_drift::missing_nonrel_channels_n",
            metric="missing_nonrel_channels_n",
            value=missing_nonrel_channels_n,
            expected=0,
            status="pass" if missing_nonrel_channels_n == 0 else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="非相対論写像の必須channelが欠落していないこと。",
        ),
        _make_row(
            cid="closure_drift::route_a_gate",
            metric="route_a_gate",
            value=route_a_gate,
            expected="A_continue",
            status="pass" if route_a_gate == "A_continue" else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="route A が継続可能であること。",
        ),
        _make_row(
            cid="closure_drift::transition",
            metric="transition",
            value=transition,
            expected="A_stay",
            status="pass" if transition == "A_stay" else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="A->B 移行が要求されていないこと。",
        ),
        _make_row(
            cid="closure_drift::checks_all_pass",
            metric="checks_pass_n/checks_total_n",
            value=f"{checks_pass_n}/{checks_total_n}",
            expected="all pass",
            status="pass" if checks_total_n > 0 and checks_pass_n == checks_total_n else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="閉包監査内の checks が全件 pass であること。",
        ),
        _make_row(
            cid="closure_drift::noether_gauge_margin_positive",
            metric="noether_gauge_margin",
            value=noether_gauge_margin,
            expected="> 0",
            status="pass" if (noether_gauge_margin is not None and noether_gauge_margin > 0.0) else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="Noether gauge margin が正であること。",
        ),
        _make_row(
            cid="closure_drift::noether_realness_margin_positive",
            metric="noether_realness_margin",
            value=noether_realness_margin,
            expected="> 0",
            status="pass" if (noether_realness_margin is not None and noether_realness_margin > 0.0) else "reject",
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="Noether realness margin が正であること。",
        ),
        _make_row(
            cid="closure_drift::noether_gauge_margin_ratio",
            metric="noether_gauge_margin_ratio_vs_frozen",
            value=noether_gauge_ratio,
            expected=f">={pass_ratio} (watch >= {watch_ratio})",
            status=noether_gauge_ratio_status,
            gate_level="watch",
            source="lagrangian_noether_observable_closure_audit",
            note="Noether gauge margin の frozen 比率（drift 監視）。",
        ),
        _make_row(
            cid="closure_drift::noether_realness_margin_ratio",
            metric="noether_realness_margin_ratio_vs_frozen",
            value=noether_realness_ratio,
            expected=f">={pass_ratio} (watch >= {watch_ratio})",
            status=noether_realness_ratio_status,
            gate_level="watch",
            source="lagrangian_noether_observable_closure_audit",
            note="Noether realness margin の frozen 比率（drift 監視）。",
        ),
    ]

    hard_fail_row_ids = [str(r["id"]) for r in rows if r.get("gate_level") == "hard" and r.get("status") != "pass"]
    watch_row_ids = [str(r["id"]) for r in rows if r.get("status") == "watch"]
    drift_reject_row_ids = [str(r["id"]) for r in rows if r.get("gate_level") == "watch" and r.get("status") == "reject"]

    # 条件分岐: `hard_fail_row_ids` を満たす経路を評価する。
    if hard_fail_row_ids:
        overall_status = "reject"
    # 条件分岐: 前段条件が不成立で、`watch_row_ids or drift_reject_row_ids` を追加評価する。
    elif watch_row_ids or drift_reject_row_ids:
        overall_status = "watch"
    else:
        overall_status = "pass"

    recalc_required = overall_status != "pass"
    recalc_reasons = hard_fail_row_ids + drift_reject_row_ids + watch_row_ids
    recalc_commands = [
        "python -B scripts/quantum/action_principle_el_derivation_audit.py",
        "python -B scripts/quantum/nonrelativistic_reduction_schrodinger_mapping_audit.py",
        "python -B scripts/quantum/derivation_parameter_falsification_pack.py",
        "python -B scripts/quantum/derivation_observable_chain_lock_audit.py",
        "python -B scripts/quantum/lagrangian_noether_observable_closure_audit.py",
        "python -B scripts/quantum/lagrangian_noether_observable_closure_drift_audit.py",
        "python -B scripts/summary/part3_audit.py --no-regenerate",
    ]

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {
            "phase": 8,
            "step": "8.7.21.3",
            "name": "Lagrangian-Noether closure drift audit",
        },
        "intent": (
            "Monitor drift against frozen closure gates and fix the operational "
            "recalculation trigger for L_total -> EL -> observables."
        ),
        "inputs": {
            "lagrangian_noether_observable_closure_audit_json": _rel(closure_json),
        },
        "frozen_baseline": {
            "noether_gauge_margin": float(baseline_noether_gauge_margin),
            "noether_realness_margin": float(baseline_noether_realness_margin),
            "pass_ratio": float(pass_ratio),
            "watch_ratio": float(watch_ratio),
        },
        "checks": rows,
        "decision": {
            "overall_status": overall_status,
            "hard_fail_row_ids": hard_fail_row_ids,
            "watch_row_ids": watch_row_ids,
            "drift_reject_row_ids": drift_reject_row_ids,
            "recalc_required": recalc_required,
            "recalc_reason_row_ids": recalc_reasons,
            "recalc_commands": recalc_commands,
            "rule": (
                "Reject if any hard row is not pass; "
                "watch if only watch rows are degraded; pass otherwise."
            ),
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "metric",
                "value",
                "expected",
                "status",
                "score",
                "gate_level",
                "source",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, payload: Dict[str, Any]) -> None:
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    baseline = payload.get("frozen_baseline") if isinstance(payload.get("frozen_baseline"), dict) else {}

    ids: List[str] = []
    scores: List[float] = []
    colors: List[str] = []
    for row in checks:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        ids.append(str(row.get("id") or ""))
        score = row.get("score")
        scores.append(float(score) if isinstance(score, (int, float)) else math.nan)
        status = str(row.get("status") or "")
        # 条件分岐: `status == "pass"` を満たす経路を評価する。
        if status == "pass":
            colors.append("#2f9e44")
        # 条件分岐: 前段条件が不成立で、`status == "watch"` を追加評価する。
        elif status == "watch":
            colors.append("#eab308")
        else:
            colors.append("#dc2626")

    pass_ratio = float(baseline.get("pass_ratio", 0.5))
    watch_ratio = float(baseline.get("watch_ratio", 0.1))

    ratio_map = {
        "gauge": None,
        "realness": None,
    }
    for row in checks:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        rid = str(row.get("id") or "")
        # 条件分岐: `rid == "closure_drift::noether_gauge_margin_ratio"` を満たす経路を評価する。
        if rid == "closure_drift::noether_gauge_margin_ratio":
            ratio_map["gauge"] = _as_float(row.get("value"))
        # 条件分岐: 前段条件が不成立で、`rid == "closure_drift::noether_realness_margin_ratio"` を追加評価する。
        elif rid == "closure_drift::noether_realness_margin_ratio":
            ratio_map["realness"] = _as_float(row.get("value"))

    ratio_labels = ["noether gauge", "noether realness"]
    ratio_values = [
        float(ratio_map["gauge"]) if ratio_map["gauge"] is not None else 0.0,
        float(ratio_map["realness"]) if ratio_map["realness"] is not None else 0.0,
    ]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.4, 8.6), dpi=180, gridspec_kw={"height_ratios": [3.1, 1.5]})

    y = np.arange(len(ids))
    ax0.barh(y, scores, color=colors)
    ax0.set_yticks(y, ids)
    ax0.set_xlim(0.0, 1.05)
    ax0.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax0.set_xlabel("drift check score (1=pass, 0.5=watch, 0=reject)")
    ax0.set_title("Lagrangian-Noether closure drift audit (gate operation)")
    ax0.grid(axis="x", alpha=0.25, linestyle=":")

    x = np.arange(len(ratio_labels))
    ax1.bar(x, ratio_values, color="#2563eb")
    ax1.set_xticks(x, ratio_labels, rotation=0, ha="center")
    ymax = max(1.05, max(ratio_values) * 1.2 if ratio_values else 1.05)
    ax1.set_ylim(0.0, ymax)
    ax1.axhline(pass_ratio, linestyle="--", color="#2f9e44", linewidth=1.2, label=f"pass >= {pass_ratio:g}")
    ax1.axhline(watch_ratio, linestyle="--", color="#eab308", linewidth=1.2, label=f"watch >= {watch_ratio:g}")
    ax1.set_ylabel("margin ratio vs frozen")
    ax1.set_title("Noether margin drift monitor")
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate closure drift audit for Step 8.7.21 operation.")
    parser.add_argument(
        "--closure-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_audit.json"),
        help="Input closure audit JSON (Step 8.7.21.1).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_drift_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_drift_audit.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_drift_audit.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--baseline-noether-gauge-margin",
        type=float,
        default=DEFAULT_BASELINE_NOETHER_GAUGE_MARGIN,
        help="Frozen baseline margin for noether gauge monitor.",
    )
    parser.add_argument(
        "--baseline-noether-realness-margin",
        type=float,
        default=DEFAULT_BASELINE_NOETHER_REALNESS_MARGIN,
        help="Frozen baseline margin for noether realness monitor.",
    )
    parser.add_argument(
        "--pass-ratio",
        type=float,
        default=0.5,
        help="Pass threshold for margin ratio monitor.",
    )
    parser.add_argument(
        "--watch-ratio",
        type=float,
        default=0.1,
        help="Watch threshold for margin ratio monitor.",
    )
    args = parser.parse_args(argv)

    # 条件分岐: `args.pass_ratio <= 0.0 or args.watch_ratio <= 0.0 or args.watch_ratio >= args...` を満たす経路を評価する。
    if args.pass_ratio <= 0.0 or args.watch_ratio <= 0.0 or args.watch_ratio >= args.pass_ratio:
        print("[error] threshold rule violated: require pass-ratio > watch-ratio > 0")
        return 2

    closure_json = Path(args.closure_json)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    for name, path in [
        ("closure-json", closure_json),
        ("out-json", out_json),
        ("out-csv", out_csv),
        ("out-png", out_png),
    ]:
        # 条件分岐: `not path.is_absolute()` を満たす経路を評価する。
        if not path.is_absolute():
            resolved = (ROOT / path).resolve()
            # 条件分岐: `name == "closure-json"` を満たす経路を評価する。
            if name == "closure-json":
                closure_json = resolved
            # 条件分岐: 前段条件が不成立で、`name == "out-json"` を追加評価する。
            elif name == "out-json":
                out_json = resolved
            # 条件分岐: 前段条件が不成立で、`name == "out-csv"` を追加評価する。
            elif name == "out-csv":
                out_csv = resolved
            # 条件分岐: 前段条件が不成立で、`name == "out-png"` を追加評価する。
            elif name == "out-png":
                out_png = resolved

    # 条件分岐: `not closure_json.exists()` を満たす経路を評価する。

    if not closure_json.exists():
        print(f"[error] missing input: {_rel(closure_json)}")
        return 2

    payload = build_payload(
        closure_json=closure_json,
        baseline_noether_gauge_margin=float(args.baseline_noether_gauge_margin),
        baseline_noether_realness_margin=float(args.baseline_noether_realness_margin),
        pass_ratio=float(args.pass_ratio),
        watch_ratio=float(args.watch_ratio),
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    _write_csv(out_csv, rows if isinstance(rows, list) else [])
    _plot(out_png, payload)

    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")
    print(
        "[summary] overall_status="
        f"{decision.get('overall_status')}, recalc_required={decision.get('recalc_required')}, "
        f"hard_fail_rows={len(decision.get('hard_fail_row_ids') or [])}"
    )

    try:
        worklog.append_event(
            {
                "event_type": "quantum_lagrangian_noether_closure_drift_audit",
                "phase": "8",
                "step": "8.7.21.3",
                "outputs": {
                    "lagrangian_noether_observable_closure_drift_audit_json": _rel(out_json),
                    "lagrangian_noether_observable_closure_drift_audit_csv": _rel(out_csv),
                    "lagrangian_noether_observable_closure_drift_audit_png": _rel(out_png),
                },
                "metrics": {
                    "overall_status": decision.get("overall_status"),
                    "recalc_required": decision.get("recalc_required"),
                    "hard_fail_row_ids_n": len(decision.get("hard_fail_row_ids") or []),
                    "watch_row_ids_n": len(decision.get("watch_row_ids") or []),
                    "drift_reject_row_ids_n": len(decision.get("drift_reject_row_ids") or []),
                },
            }
        )
    except Exception as exc:
        print(f"[warn] worklog append skipped: {exc}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

