#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lagrangian_noether_closure_operational_cycle.py

Step 8.7.21.4:
8.7.22（weak-interaction closure watch）更新時に 8.7.21.3（closure drift）を
同一サイクルで再評価する運用監査を固定出力する。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_run` の入出力契約と処理意図を定義する。

def _run(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "cmd": cmd,
        "rc": int(proc.returncode),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
    }


# 関数: `_load_json` の入出力契約と処理意図を定義する。

def _load_json(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_status_score` の入出力契約と処理意図を定義する。

def _status_score(status: str) -> float:
    table = {"pass": 1.0, "watch": 0.5, "reject": 0.0}
    return table.get(str(status or "").lower(), 0.0)


# 関数: `_status_color` の入出力契約と処理意図を定義する。

def _status_color(status: str) -> str:
    table = {"pass": "#2f9e44", "watch": "#eab308", "reject": "#dc2626"}
    return table.get(str(status or "").lower(), "#6b7280")


# 関数: `_metric_row` の入出力契約と処理意図を定義する。

def _metric_row(metric_id: str, metric: str, value: Any, expected: Any, status: str, note: str) -> Dict[str, Any]:
    return {
        "id": metric_id,
        "metric": metric,
        "value": value,
        "expected": expected,
        "status": status,
        "score": _status_score(status),
        "note": note,
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "metric", "value", "expected", "status", "score", "note"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, rows: List[Dict[str, Any]], title: str) -> None:
    labels = [str(r.get("id") or "") for r in rows]
    scores = [float(r.get("score") or 0.0) for r in rows]
    colors = [_status_color(str(r.get("status") or "")) for r in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=180)
    ax.barh(y, scores, color=colors)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.05)
    ax.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.0)
    ax.set_xlabel("gate score (1=pass, 0.5=watch, 0=reject)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Step 8.7.21.4: run 8.7.22 closure checks and rerun 8.7.21.3 drift monitor in one operational cycle."
    )
    parser.add_argument(
        "--skip-upstream",
        action="store_true",
        help="Skip running upstream scripts and only evaluate existing output JSON files.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_closure_operational_cycle_8722_8721.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_closure_operational_cycle_8722_8721.csv"),
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_closure_operational_cycle_8722_8721.png"),
    )
    args = parser.parse_args(argv)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    # 条件分岐: `not out_png.is_absolute()` を満たす経路を評価する。

    if not out_png.is_absolute():
        out_png = (ROOT / out_png).resolve()

    runs: List[Dict[str, Any]] = []
    # 条件分岐: `not args.skip_upstream` を満たす経路を評価する。
    if not args.skip_upstream:
        commands = [
            [sys.executable, "-B", "scripts/quantum/weak_interaction_ckm_first_row_audit.py"],
            [sys.executable, "-B", "scripts/quantum/weak_interaction_pmns_first_row_audit.py"],
            [sys.executable, "-B", "scripts/quantum/weak_interaction_beta_decay_route_ab_audit.py"],
            [sys.executable, "-B", "scripts/quantum/lagrangian_noether_observable_closure_drift_audit.py"],
        ]
        runs = [_run(cmd) for cmd in commands]

    weak_json = ROOT / "output" / "public" / "quantum" / "weak_interaction_beta_decay_route_ab_audit.json"
    drift_json = ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_drift_audit.json"

    weak = _load_json(weak_json)
    drift = _load_json(drift_json)
    weak_decision = weak.get("decision") if isinstance(weak.get("decision"), dict) else {}
    drift_decision = drift.get("decision") if isinstance(drift.get("decision"), dict) else {}

    rc_status = "pass" if all(int(r.get("rc", 1)) == 0 for r in runs) else "reject"
    transition = str(weak_decision.get("transition") or "")
    transition_status = "pass" if transition == "A_stay_B_reject" else "reject"
    ckm_pmns_status = str((weak_decision.get("ckm_pmns_closure") or {}).get("status") or "reject")
    drift_status = str(drift_decision.get("overall_status") or "reject")
    recalc_required = bool(drift_decision.get("recalc_required"))
    recalc_status = "pass" if recalc_required is False else "reject"

    rows = [
        _metric_row(
            "op_cycle::upstream_rc",
            "8.7.22 + 8.7.21.3 command return codes",
            ",".join([str(int(r.get("rc", 1))) for r in runs]) if runs else "skipped",
            "all 0",
            rc_status if runs else "watch",
            "Operational cycle script return codes.",
        ),
        _metric_row(
            "op_cycle::weak_transition",
            "weak_interaction transition",
            transition,
            "A_stay_B_reject",
            transition_status,
            "Route-A/B transition consistency.",
        ),
        _metric_row(
            "op_cycle::ckm_pmns_closure",
            "weak_interaction ckm_pmns_closure status",
            ckm_pmns_status,
            "pass or watch",
            "pass" if ckm_pmns_status == "pass" else ("watch" if ckm_pmns_status == "watch" else "reject"),
            "CKM watch is currently allowed as operational watch.",
        ),
        _metric_row(
            "op_cycle::drift_overall",
            "closure drift overall_status",
            drift_status,
            "pass",
            "pass" if drift_status == "pass" else ("watch" if drift_status == "watch" else "reject"),
            "L_total->EL->observables drift gate result.",
        ),
        _metric_row(
            "op_cycle::drift_recalc_required",
            "closure drift recalc_required",
            recalc_required,
            False,
            recalc_status,
            "Operational recalc trigger must remain false.",
        ),
    ]

    hard_reject = any(str(r.get("status")) == "reject" for r in rows if str(r.get("id")) != "op_cycle::ckm_pmns_closure")
    # 条件分岐: `hard_reject` を満たす経路を評価する。
    if hard_reject:
        overall_status = "reject"
        decision = "operational_cycle_failed"
    # 条件分岐: 前段条件が不成立で、`ckm_pmns_status == "watch"` を追加評価する。
    elif ckm_pmns_status == "watch":
        overall_status = "watch"
        decision = "operational_cycle_stable_with_ckm_watch"
    # 条件分岐: 前段条件が不成立で、`ckm_pmns_status == "pass"` を追加評価する。
    elif ckm_pmns_status == "pass":
        overall_status = "pass"
        decision = "operational_cycle_stable_all_pass"
    else:
        overall_status = "reject"
        decision = "operational_cycle_ckm_pmns_reject"

    payload = {
        "generated_utc": _iso_utc_now(),
        "phase": 8,
        "step": "8.7.21.4",
        "title": "Lagrangian-Noether closure operational cycle (8.7.22 -> 8.7.21.3)",
        "inputs": {
            "weak_interaction_beta_decay_route_ab_audit_json": _rel(weak_json),
            "lagrangian_noether_observable_closure_drift_audit_json": _rel(drift_json),
        },
        "upstream_runs": runs,
        "checks": rows,
        "decision": {
            "overall_status": overall_status,
            "decision": decision,
            "watch_is_allowed_only_for": ["op_cycle::ckm_pmns_closure"],
            "rule": "Reject if any non-CKM row is reject; watch if only CKM closure is watch; pass if all pass.",
        },
        "outputs": {
            "json": _rel(out_json),
            "csv": _rel(out_csv),
            "png": _rel(out_png),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, rows)
    _plot(out_png, rows, "Closure operational cycle (8.7.22 -> 8.7.21.3)")

    print(f"[ok] json: {_rel(out_json)}")
    print(f"[ok] csv : {_rel(out_csv)}")
    print(f"[ok] png : {_rel(out_png)}")
    print(f"[summary] overall_status={overall_status}, decision={decision}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_lagrangian_noether_closure_operational_cycle",
                "phase": "8",
                "step": "8.7.21.4",
                "outputs": {
                    "lagrangian_noether_closure_operational_cycle_json": _rel(out_json),
                    "lagrangian_noether_closure_operational_cycle_csv": _rel(out_csv),
                    "lagrangian_noether_closure_operational_cycle_png": _rel(out_png),
                },
                "metrics": {
                    "overall_status": overall_status,
                    "decision": decision,
                    "ckm_pmns_closure_status": ckm_pmns_status,
                    "drift_overall_status": drift_status,
                    "drift_recalc_required": recalc_required,
                },
            }
        )
    except Exception as exc:
        print(f"[warn] worklog append skipped: {exc}")

    return 0 if overall_status in {"pass", "watch"} else 1


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

