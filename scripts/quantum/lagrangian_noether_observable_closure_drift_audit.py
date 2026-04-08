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
from scripts.utils.plot_style import get_wavep_font_size  # noqa: E402

try:
    import matplotlib as mpl
    from scripts.utils.plot_style import install_wavep_cjk_font_override  # noqa: E402

    install_wavep_cjk_font_override(preferred_name="Noto Sans CJK JP")
    mpl.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

DEFAULT_BASELINE_NOETHER_GAUGE_MARGIN = 4.999993032890784e-08
DEFAULT_BASELINE_NOETHER_REALNESS_MARGIN = 5.0e-10


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
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(value: Any) -> Optional[float]:
    # 条件分岐: `isinstance(value, (int, float))` を満たす経路を評価する。
    if isinstance(value, (int, float)):
        v = float(value)
        # 条件分岐: `math.isfinite(v)` を満たす経路を評価する。
        if math.isfinite(v):
            return v

    return None


# 関数: `_check_value` の入出力契約と処理意図を定義する。

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


# 関数: `_count_check_rows` の入出力契約と処理意図を定義する。

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


# 関数: `_margin_ratio_status` の入出力契約と処理意図を定義する。

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


# 関数: `_score_from_status` の入出力契約と処理意図を定義する。

def _score_from_status(status: str) -> float:
    # 条件分岐: `status == "pass"` を満たす経路を評価する。
    if status == "pass":
        return 1.0

    # 条件分岐: `status == "watch"` を満たす経路を評価する。

    if status == "watch":
        return 0.5

    return 0.0


# 関数: `_make_row` の入出力契約と処理意図を定義する。

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


# 関数: `build_payload` の入出力契約と処理意図を定義する。

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
    closure_shared_gate_policy = str(decision.get("shared_gate_policy") or "unknown")
    allow_watch_closure = closure_shared_gate_policy == "watch_if_bell_pairing_only"
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

    overall_row_status = (
        "pass"
        if overall_status_value == "pass"
        else ("watch" if allow_watch_closure and overall_status_value == "watch" else "reject")
    )
    overall_row_expected = "pass or watch(policy)" if allow_watch_closure else "pass"
    overall_row_gate_level = "watch" if allow_watch_closure else "hard"

    checks_all_pass_status = (
        "pass"
        if checks_total_n > 0 and checks_pass_n == checks_total_n
        else (
            "watch"
            if allow_watch_closure and checks_total_n > 0 and len(hard_fail_ids) == 0
            else "reject"
        )
    )
    checks_all_pass_expected = "all pass (or watch under watch-policy)" if allow_watch_closure else "all pass"
    checks_all_pass_gate_level = "watch" if allow_watch_closure else "hard"

    rows: List[Dict[str, Any]] = [
        _make_row(
            cid="closure_drift::overall_status",
            metric="overall_status",
            value=overall_status_value,
            expected=overall_row_expected,
            status=overall_row_status,
            gate_level=overall_row_gate_level,
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
            cid="closure_drift::shared_gate_policy",
            metric="shared_gate_policy(closure)",
            value=closure_shared_gate_policy,
            expected="strict_hard or watch_if_bell_pairing_only",
            status=(
                "pass"
                if closure_shared_gate_policy in {"strict_hard", "watch_if_bell_pairing_only"}
                else "watch"
            ),
            gate_level="watch",
            source="lagrangian_noether_observable_closure_audit",
            note="closure 側 shared gate policy を drift 監査へ引き継いでいること。",
        ),
        _make_row(
            cid="closure_drift::checks_all_pass",
            metric="checks_pass_n/checks_total_n",
            value=f"{checks_pass_n}/{checks_total_n}",
            expected=checks_all_pass_expected,
            status=checks_all_pass_status,
            gate_level=checks_all_pass_gate_level,
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

    recalc_required = bool(hard_fail_row_ids or drift_reject_row_ids)
    recalc_reasons = hard_fail_row_ids + drift_reject_row_ids
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
            "closure_shared_gate_policy": closure_shared_gate_policy,
            "recalc_required": recalc_required,
            "recalc_reason_row_ids": recalc_reasons,
            "recalc_commands": recalc_commands,
            "rule": (
                "Reject if any hard row is not pass; "
                "watch if only watch rows are degraded; pass otherwise."
            ),
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

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


# 関数: `_wrap_two_line_label` の入出力契約と処理意図を定義する。

def _wrap_two_line_label(text: str) -> str:
    words = [
        token
        for token in str(text).replace("::", " ").replace(":", " ").replace("_", " ").split()
        if token
    ]
    if len(words) <= 1:
        return " ".join(words) if words else str(text)

    best_index = 1
    best_score: Optional[int] = None
    for idx in range(1, len(words)):
        left = " ".join(words[:idx])
        right = " ".join(words[idx:])
        score = max(len(left), len(right))
        if best_score is None or score < best_score:
            best_score = score
            best_index = idx

    return " ".join(words[:best_index]) + "\n" + " ".join(words[best_index:])


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, payload: Dict[str, Any]) -> None:
    display_labels = {
        "closure_drift::overall_status": "ドリフト: 全体判定",
        "closure_drift::hard_fail_ids_n": "ドリフト: hard失敗件数",
        "closure_drift::watch_ids_n": "ドリフト: 監視件数",
        "closure_drift::missing_equations_n": "ドリフト: 欠落式件数",
        "closure_drift::missing_nonrel_channels_n": "ドリフト: 欠落channel件数",
        "closure_drift::route_a_gate": "ドリフト: 経路A判定",
        "closure_drift::transition": "ドリフト: 遷移判定",
        "closure_drift::shared_gate_policy": "ドリフト: 共有ゲート方針",
        "closure_drift::checks_all_pass": "ドリフト: 判定通過数",
        "closure_drift::noether_gauge_margin_positive": "ドリフト: Noether\nゲージ余裕の正値性",
        "closure_drift::noether_realness_margin_positive": "ドリフト: Noether\n実数性余裕の正値性",
        "closure_drift::noether_gauge_margin_ratio": "ドリフト: Noether\nゲージ余裕比",
        "closure_drift::noether_realness_margin_ratio": "ドリフト: Noether\n実数性余裕比",
    }
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    baseline = payload.get("frozen_baseline") if isinstance(payload.get("frozen_baseline"), dict) else {}

    ids: List[str] = []
    scores: List[float] = []
    colors: List[str] = []
    for row in checks:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        ids.append(display_labels.get(str(row.get("id") or ""), str(row.get("id") or "")))
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

    ratio_labels = ["Noether ゲージ", "Noether 実数性"]
    ratio_values = [
        float(ratio_map["gauge"]) if ratio_map["gauge"] is not None else 0.0,
        float(ratio_map["realness"]) if ratio_map["realness"] is not None else 0.0,
    ]

    title_size = max(get_wavep_font_size("title"), 16.0)
    axis_size = max(get_wavep_font_size("axis"), 14.0)
    tick_size = max(get_wavep_font_size("tick"), 13.0)
    upper_y_tick_size = max(tick_size + 1.8, 14.8)
    legend_size = max(get_wavep_font_size("legend"), 12.8)

    upper_height = max(14.4, 0.78 * len(ids) + 5.0)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13.0, upper_height), dpi=180, gridspec_kw={"height_ratios": [5.55, 1.55]})

    y = np.arange(len(ids))
    wrapped_ids = [_wrap_two_line_label(label) for label in ids]
    ax0.barh(y, scores, color=colors, height=0.72)
    ax0.set_yticks(y)
    ax0.set_yticklabels(wrapped_ids, fontsize=upper_y_tick_size)
    for tick in ax0.get_yticklabels():
        tick.set_linespacing(1.15)

    ax0.set_xlim(0.0, 1.05)
    ax0.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax0.set_xlabel("ドリフト監査スコア（1で通過、0.5で監視、0で棄却）", fontsize=axis_size)
    ax0.set_title("Lagrangian-Noether 閉包ドリフト監査（ゲート運用）", fontsize=title_size, pad=9.0)
    ax0.tick_params(axis="x", labelsize=tick_size)
    ax0.tick_params(axis="y", pad=7.0)
    ax0.grid(axis="x", alpha=0.25, linestyle=":")

    x = np.arange(len(ratio_labels))
    ax1.bar(x, ratio_values, color="#2563eb")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ratio_labels, rotation=0, ha="center", fontsize=tick_size)
    ymax = max(1.05, max(ratio_values) * 1.2 if ratio_values else 1.05)
    ax1.set_ylim(0.0, ymax)
    ax1.axhline(pass_ratio, linestyle="--", color="#2f9e44", linewidth=1.2, label=f"通過 >= {pass_ratio:g}")
    ax1.axhline(watch_ratio, linestyle="--", color="#eab308", linewidth=1.2, label=f"監視 >= {watch_ratio:g}")
    ax1.set_ylabel("固定基準に対する余裕比", fontsize=axis_size)
    ax1.set_title("Noether 余裕比ドリフト監視", fontsize=title_size, pad=8.0)
    ax1.tick_params(axis="y", labelsize=tick_size)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.legend(loc="upper right", frameon=False, fontsize=legend_size)

    fig.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.11, hspace=0.58)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

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
    out_pdf = Path(args.out_png).with_suffix(".pdf")

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

    # 条件分岐: `not out_pdf.is_absolute()` を満たす経路を評価する。

    if not out_pdf.is_absolute():
        out_pdf = (ROOT / out_pdf).resolve()

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
    _plot(out_pdf, payload)

    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")
    print(f"[ok] wrote: {_rel(out_pdf)}")
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
