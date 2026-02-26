#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
born_route_a_proxy_constraints.py

Step 8.7.2:
Born則ルートA（導出チャレンジ）を、観測proxy（位相・可視度・選別感度）で
同一ゲート評価し、A継続 / A棄却→B移行 を機械判定する。

出力:
  - output/public/quantum/born_route_a_proxy_constraints_pack.json
  - output/public/quantum/born_route_a_proxy_constraints_pack.csv
  - output/public/quantum/born_route_a_proxy_constraints_pack.png
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
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(v: Any) -> Optional[float]:
    # 条件分岐: `isinstance(v, (int, float))` を満たす経路を評価する。
    if isinstance(v, (int, float)):
        f = float(v)
        # 条件分岐: `math.isfinite(f)` を満たす経路を評価する。
        if math.isfinite(f):
            return f

    return None


def _criterion(
    *,
    cid: str,
    proxy: str,
    metric: str,
    value: Optional[float],
    threshold: float,
    operator: str,
    gate: bool,
    note: str,
) -> Dict[str, Any]:
    passed: Optional[bool] = None
    # 条件分岐: `value is not None` を満たす経路を評価する。
    if value is not None:
        # 条件分岐: `operator == "<="` を満たす経路を評価する。
        if operator == "<=":
            passed = bool(value <= threshold)
        # 条件分岐: 前段条件が不成立で、`operator == ">="` を追加評価する。
        elif operator == ">=":
            passed = bool(value >= threshold)

    return {
        "id": cid,
        "proxy": proxy,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "operator": operator,
        "pass": passed,
        "gate": gate,
        "note": note,
    }


def _extract_row(rows: List[Dict[str, Any]], channel: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        # 条件分岐: `str(row.get("channel") or "") == channel` を満たす経路を評価する。

        if str(row.get("channel") or "") == channel:
            return row

    return None


def build_pack() -> Dict[str, Any]:
    matter_path = ROOT / "output" / "public" / "quantum" / "matter_wave_interference_precision_audit_metrics.json"
    bell_pack_path = ROOT / "output" / "public" / "quantum" / "bell" / "falsification_pack.json"
    bell_sel_path = ROOT / "output" / "public" / "quantum" / "bell_selection_sensitivity_summary.json"

    matter = _read_json(matter_path) if matter_path.exists() else {}
    bell = _read_json(bell_pack_path) if bell_pack_path.exists() else {}
    bell_sel = _read_json(bell_sel_path) if bell_sel_path.exists() else {}

    rows = matter.get("rows") if isinstance(matter.get("rows"), list) else []

    row_alpha = _extract_row(rows, "atom_recoil_alpha")
    row_visibility = _extract_row(rows, "atom_interferometer_precision")
    row_molecular = _extract_row(rows, "molecular_isotopic_scaling")
    precision_gap_watch = matter.get("precision_gap_watch") if isinstance(matter.get("precision_gap_watch"), dict) else {}

    alpha_z = _as_float((row_alpha or {}).get("metric_value"))
    visibility_ratio_raw = _as_float((row_visibility or {}).get("metric_value"))
    # 条件分岐: `visibility_ratio_raw is None` を満たす経路を評価する。
    if visibility_ratio_raw is None:
        visibility_ratio_raw = _as_float(precision_gap_watch.get("visibility_reference_ratio"))

    # 条件分岐: `visibility_ratio_raw is None` を満たす経路を評価する。

    if visibility_ratio_raw is None:
        visibility_ratio_raw = _as_float(precision_gap_watch.get("median_ratio"))

    visibility_ratio_log10 = float(np.log10(visibility_ratio_raw)) if visibility_ratio_raw is not None and visibility_ratio_raw > 0 else None
    visibility_ref_channel = str(precision_gap_watch.get("visibility_reference_channel") or "atom_gravimeter")
    molecular_z = _as_float((row_molecular or {}).get("metric_value"))

    datasets = bell.get("datasets") if isinstance(bell.get("datasets"), list) else []
    fast_prefixes = ("weihs1998_", "nist_")
    fast_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        # 条件分岐: `not isinstance(ds, dict)` を満たす経路を評価する。
        if not isinstance(ds, dict):
            continue

        dataset_id = str(ds.get("dataset_id") or "")
        # 条件分岐: `not dataset_id.startswith(fast_prefixes)` を満たす経路を評価する。
        if not dataset_id.startswith(fast_prefixes):
            continue

        delay = ds.get("delay_signature") if isinstance(ds.get("delay_signature"), dict) else {}
        a = delay.get("Alice") if isinstance(delay.get("Alice"), dict) else {}
        b = delay.get("Bob") if isinstance(delay.get("Bob"), dict) else {}
        z_a = _as_float(a.get("z_delta_median"))
        z_b = _as_float(b.get("z_delta_median"))
        z_candidates = [z for z in (z_a, z_b) if z is not None]
        z_max = max(z_candidates) if z_candidates else None
        fast_rows.append(
            {
                "dataset_id": dataset_id,
                "ratio": _as_float(ds.get("ratio")),
                "delay_z_alice": z_a,
                "delay_z_bob": z_b,
                "delay_z_max": z_max,
            }
        )

    fast_z_max = [r["delay_z_max"] for r in fast_rows if isinstance(r.get("delay_z_max"), (int, float))]
    fast_ratios = [r["ratio"] for r in fast_rows if isinstance(r.get("ratio"), (int, float))]
    min_fast_zmax = min(float(v) for v in fast_z_max) if fast_z_max else None
    min_fast_ratio = min(float(v) for v in fast_ratios) if fast_ratios else None

    criteria: List[Dict[str, Any]] = [
        _criterion(
            cid="phase_alpha_consistency",
            proxy="phase",
            metric="atom_recoil_alpha_abs_z",
            value=alpha_z,
            threshold=3.0,
            operator="<=",
            gate=True,
            note="位相整合の最小ゲート（abs_z<=3）。",
        ),
        _criterion(
            cid="phase_molecular_scaling",
            proxy="phase",
            metric="molecular_isotopic_scaling_zmax",
            value=molecular_z,
            threshold=3.0,
            operator="<=",
            gate=True,
            note="分子スケーリング整合の最小ゲート（z<=3）。",
        ),
        _criterion(
            cid="selection_delay_signature_fast",
            proxy="selection",
            metric="min_fast_switching_delay_zmax",
            value=min_fast_zmax,
            threshold=3.0,
            operator=">=",
            gate=True,
            note="fast-switch/time-tag 系で setting依存遅延 z が 3以上。",
        ),
        _criterion(
            cid="selection_sweep_sensitivity_fast",
            proxy="selection",
            metric="min_fast_switching_selection_ratio",
            value=min_fast_ratio,
            threshold=1.0,
            operator=">=",
            gate=True,
            note="selection sweep の変動幅が統計幅1σ以上（ratio>=1）。",
        ),
        _criterion(
            cid="visibility_atom_precision_gap",
            proxy="visibility",
            metric="log10_atom_interferometer_current_over_required_ratio",
            value=visibility_ratio_log10,
            threshold=1.0,
            operator="<=",
            gate=False,
            note="可視度差分の決着力監視（log10正規化；1桁以内をwatch閾値）。未達でも即棄却には使わない。",
        ),
    ]

    hard_fail = [c["id"] for c in criteria if c.get("gate") and c.get("pass") is False]
    hard_unknown = [c["id"] for c in criteria if c.get("gate") and c.get("pass") is None]

    # 条件分岐: `hard_fail or hard_unknown` を満たす経路を評価する。
    if hard_fail or hard_unknown:
        decision = "A_to_B"
    else:
        decision = "A_continue"

    visibility_ok = next((c.get("pass") for c in criteria if c.get("id") == "visibility_atom_precision_gap"), None)
    watchlist: List[str] = []
    # 条件分岐: `visibility_ok is False` を満たす経路を評価する。
    if visibility_ok is False:
        watchlist.append("visibility_precision_gap")

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.2", "name": "Born route-A proxy constraints packaging"},
        "intent": "Freeze an operational machine gate for Born route-A using phase/visibility/selection proxies.",
        "inputs": {
            "matter_wave_interference_precision_audit_metrics_json": _rel(matter_path),
            "bell_falsification_pack_json": _rel(bell_pack_path),
            "bell_selection_sensitivity_summary_json": _rel(bell_sel_path),
        },
        "criteria": criteria,
        "decision": {
            "route_a_gate": decision,
            "hard_fail_ids": hard_fail,
            "hard_unknown_ids": hard_unknown,
            "watchlist": watchlist,
            "rule": "A_to_B if any hard gate fails/unknown; otherwise A_continue.",
        },
        "diagnostics": {
            "fast_switching_datasets": fast_rows,
            "bell_selection_summary_available": bool(bell_sel),
            "matter_rows_n": len(rows),
            "visibility_reference": {
                "channel": visibility_ref_channel,
                "ratio_raw": visibility_ratio_raw,
                "ratio_log10": visibility_ratio_log10,
                "threshold_log10": 1.0,
            },
        },
    }


def _write_csv(path: Path, criteria: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "proxy", "metric", "value", "threshold", "operator", "pass", "gate", "note"],
        )
        writer.writeheader()
        for row in criteria:
            writer.writerow(row)


def _plot(path: Path, payload: Dict[str, Any]) -> None:
    crit = payload.get("criteria") if isinstance(payload.get("criteria"), list) else []
    score_rows = []
    for row in crit:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        cid = str(row.get("id") or "")
        val = _as_float(row.get("value"))
        thr = _as_float(row.get("threshold"))
        op = str(row.get("operator") or "")
        # 条件分岐: `val is None or thr is None or thr == 0.0` を満たす経路を評価する。
        if val is None or thr is None or thr == 0.0:
            score = math.nan
        # 条件分岐: 前段条件が不成立で、`op == "<="` を追加評価する。
        elif op == "<=":
            score = float(val / thr)
        # 条件分岐: 前段条件が不成立で、`op == ">="` を追加評価する。
        elif op == ">=":
            score = float(thr / val) if val != 0.0 else math.inf
        else:
            score = math.nan

        score_rows.append((cid, score, row.get("pass"), bool(row.get("gate"))))

    labels = [r[0] for r in score_rows]
    scores = [r[1] for r in score_rows]
    colors = []
    for _, _, passed, gate in score_rows:
        # 条件分岐: `passed is None` を満たす経路を評価する。
        if passed is None:
            colors.append("#9ca3af")
        # 条件分岐: 前段条件が不成立で、`passed` を追加評価する。
        elif passed:
            colors.append("#2f9e44" if gate else "#1d4ed8")
        else:
            colors.append("#dc2626")

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)
    ax.barh(y, scores, color=colors)
    ax.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("normalized score (<=1 is pass)")
    ax.set_title(f"Born route-A proxy gate: {payload.get('decision', {}).get('route_a_gate', 'unknown')}")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze Born route-A proxy constraints and machine gate (A_continue vs A_to_B).")
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_route_a_proxy_constraints_pack.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_route_a_proxy_constraints_pack.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_route_a_proxy_constraints_pack.png"),
        help="Output PNG path.",
    )
    args = ap.parse_args(argv)

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

    payload = build_pack()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, payload.get("criteria") if isinstance(payload.get("criteria"), list) else [])
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_born_route_a_proxy_constraints",
                "phase": "8.7.2",
                "inputs": payload.get("inputs"),
                "outputs": {
                    "born_route_a_proxy_constraints_pack_json": _rel(out_json),
                    "born_route_a_proxy_constraints_pack_csv": _rel(out_csv),
                    "born_route_a_proxy_constraints_pack_png": _rel(out_png),
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
