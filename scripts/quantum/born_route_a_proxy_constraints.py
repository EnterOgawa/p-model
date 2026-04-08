#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
born_route_a_proxy_constraints.py

Step 8.7.2 / 8.7.49.5:
Born則ルートA（導出チャレンジ）を、観測proxy（位相・可視度・選別感度）と
位相ランダム化 + 線形検出応答の statistical bridge で同一パック化し、
A継続 / A棄却→B移行 と「完全導出か、条件付きbridge止まりか」を
再現可能な形で固定する。

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

try:
    import matplotlib as mpl
    from scripts.utils.plot_style import install_wavep_cjk_font_override  # noqa: E402

    install_wavep_cjk_font_override(preferred_name="Noto Sans CJK JP")
    mpl.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


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

def _as_float(v: Any) -> Optional[float]:
    # 条件分岐: `isinstance(v, (int, float))` を満たす経路を評価する。
    if isinstance(v, (int, float)):
        f = float(v)
        # 条件分岐: `math.isfinite(f)` を満たす経路を評価する。
        if math.isfinite(f):
            return f

    return None


# 関数: `_criterion` の入出力契約と処理意図を定義する。

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


# 関数: `_extract_row` の入出力契約と処理意図を定義する。

def _extract_row(rows: List[Dict[str, Any]], channel: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        # 条件分岐: `str(row.get("channel") or "") == channel` を満たす経路を評価する。

        if str(row.get("channel") or "") == channel:
            return row

    return None


# 関数: `_normalize_weights` の入出力契約と処理意図を定義する。

def _normalize_weights(weights: List[float]) -> List[float]:
    positives = [float(max(w, 0.0)) for w in weights]
    total = float(sum(positives))
    # 条件分岐: `total <= 0.0` を満たす経路を評価する。
    if total <= 0.0:
        raise ValueError("weights must contain at least one positive entry")

    return [w / total for w in positives]


# 関数: `_simulate_phase_randomized_case` の入出力契約と処理意図を定義する。

def _simulate_phase_randomized_case(
    *,
    case_id: str,
    target_weights: List[float],
    submode_counts: List[int],
    shots: int,
    seed: int,
) -> Dict[str, Any]:
    weights = np.asarray(_normalize_weights(target_weights), dtype=float)
    counts = np.asarray(submode_counts, dtype=int)
    # 条件分岐: `weights.size != counts.size` を満たす経路を評価する。
    if weights.size != counts.size:
        raise ValueError("target_weights and submode_counts must have the same length")

    # 条件分岐: `np.any(counts <= 0)` を満たす経路を評価する。

    if np.any(counts <= 0):
        raise ValueError("submode_counts must all be positive")

    amplitudes = [np.full(int(n), float(np.sqrt(w / n)), dtype=float) for w, n in zip(weights, counts)]
    rng = np.random.default_rng(seed)
    mean_intensity = np.zeros_like(weights)

    for _ in range(shots):
        shot_intensity = np.zeros_like(weights)
        for idx, amp in enumerate(amplitudes):
            phases = rng.uniform(0.0, 2.0 * np.pi, size=amp.size)
            complex_amplitude = np.sum(amp * np.exp(1j * phases))
            shot_intensity[idx] = float(np.abs(complex_amplitude) ** 2)

        mean_intensity += shot_intensity

    mean_intensity /= float(shots)
    predicted_frequency = mean_intensity / float(np.sum(mean_intensity))
    residual = predicted_frequency - weights
    cross_term_residual = mean_intensity - weights

    return {
        "case_id": case_id,
        "shots": int(shots),
        "seed": int(seed),
        "target_frequency": weights.tolist(),
        "submode_counts": counts.tolist(),
        "mean_intensity": mean_intensity.tolist(),
        "predicted_frequency": predicted_frequency.tolist(),
        "max_abs_frequency_error": float(np.max(np.abs(residual))),
        "l1_frequency_error": float(np.sum(np.abs(residual))),
        "max_abs_cross_term_residual": float(np.max(np.abs(cross_term_residual))),
    }


# 関数: `_build_statistical_bridge` の入出力契約と処理意図を定義する。

def _build_statistical_bridge() -> Dict[str, Any]:
    cases = [
        _simulate_phase_randomized_case(
            case_id="two_path_balanced",
            target_weights=[0.5, 0.5],
            submode_counts=[8, 8],
            shots=8192,
            seed=2026031301,
        ),
        _simulate_phase_randomized_case(
            case_id="three_bin_skewed",
            target_weights=[0.15, 0.35, 0.5],
            submode_counts=[3, 5, 7],
            shots=12288,
            seed=2026031302,
        ),
        _simulate_phase_randomized_case(
            case_id="gaussian_profile_7bin",
            target_weights=[0.009172, 0.069913, 0.239062, 0.363706, 0.239062, 0.069913, 0.009172],
            submode_counts=[4, 4, 4, 4, 4, 4, 4],
            shots=16384,
            seed=2026031303,
        ),
    ]
    max_abs_frequency_error = max(float(case["max_abs_frequency_error"]) for case in cases)
    max_abs_cross_term_residual = max(float(case["max_abs_cross_term_residual"]) for case in cases)
    tolerance = 0.02
    monte_carlo_pass = bool(max_abs_frequency_error <= tolerance)

    return {
        "status": "partial_bridge_fixed_operational_rule_retained",
        "assumptions": [
            "psi is the normalized positive-frequency envelope fluctuation fixed in 8.7.49.2.",
            "Micro-phases are sufficiently mixed across sub-events so ensemble cross-terms average out.",
            "Single-shot detector response is linear in the transported local energy density.",
            "Single-shot backreaction on the P-field is negligible in the same regime where the operational Born rule was adopted.",
        ],
        "what_is_fixed": [
            "|psi|^2 can be read as normalized envelope-energy density in the frequentist limit under the assumptions above.",
            "The bridge explains why count frequencies track |psi|^2 without claiming a full microscopic derivation.",
        ],
        "what_remains_open": [
            "Why phase mixing follows generically from P dynamics rather than from an external ergodic assumption.",
            "Why detector linearity should emerge microscopically for arbitrary measurement devices.",
            "Why conditioning/state update reduces to the Lueders/Kraus form.",
        ],
        "monte_carlo": {
            "tolerance_max_abs_frequency_error": tolerance,
            "max_abs_frequency_error": max_abs_frequency_error,
            "max_abs_cross_term_residual": max_abs_cross_term_residual,
            "pass": monte_carlo_pass,
            "cases": cases,
        },
        "decision": {
            "born_route": "partial_bridge_only",
            "full_first_principles_derivation": False,
            "operational_born_rule_retained": True,
        },
    }


# 関数: `build_pack` の入出力契約と処理意図を定義する。

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
    statistical_bridge = _build_statistical_bridge()
    statistical_bridge_max_abs_frequency_error = _as_float(
        ((statistical_bridge.get("monte_carlo") if isinstance(statistical_bridge.get("monte_carlo"), dict) else {}).get("max_abs_frequency_error"))
    )

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
        _criterion(
            cid="statistical_bridge_mc_max_abs_frequency_error",
            proxy="born_bridge",
            metric="phase_randomized_frequency_max_abs_error",
            value=statistical_bridge_max_abs_frequency_error,
            threshold=0.02,
            operator="<=",
            gate=False,
            note="位相ランダム化 + 線形検出応答の toy Monte Carlo が target frequency を 2e-2 以内で再現することを確認する。",
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
        "phase": {"phase": 8, "step": "8.7.49.5", "name": "Born route-A statistical bridge + proxy constraints packaging"},
        "lineage": {"origin_step": "8.7.2"},
        "intent": "Freeze the conditional statistical bridge for Born rule plus the operational machine gate using phase/visibility/selection proxies.",
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
            "born_statistical_bridge": statistical_bridge.get("decision"),
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
            "statistical_bridge": statistical_bridge,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

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


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, payload: Dict[str, Any]) -> None:
    display_labels = {
        "phase_alpha_consistency": "位相: 原子反跳 α\n整合",
        "phase_molecular_scaling": "位相: 分子同位体\nスケーリング",
        "selection_delay_signature_fast": "選別: 高速切替\n遅延指標",
        "selection_sweep_sensitivity_fast": "選別: 高速切替\n感度掃引",
        "visibility_atom_precision_gap": "可視度: 原子干渉計\n精度差",
        "statistical_bridge_mc_max_abs_frequency_error": "Born橋渡し:\nモンテカルロ最大誤差",
    }
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

    labels = [display_labels.get(r[0], r[0]) for r in score_rows]
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
    decision_label = {
        "A_continue": "A継続",
        "A_reject": "A棄却",
        "unknown": "不明",
    }.get(str(payload.get("decision", {}).get("route_a_gate", "unknown")), str(payload.get("decision", {}).get("route_a_gate", "unknown")))
    fig, ax = plt.subplots(figsize=(11.8, 5.8), dpi=180)
    ax.barh(y, scores, color=colors)
    ax.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", labelsize=14.8)
    ax.set_xlabel("正規化スコア（1以下で通過）", fontsize=15.2)
    ax.set_title(
        f"Born 経路A プロキシ判定（{decision_label}）",
        fontsize=16.0,
        pad=8.0,
    )
    ax.tick_params(axis="x", labelsize=14.4)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze Born route-A statistical bridge and proxy gate (A_continue vs A_to_B).")
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
    out_pdf = Path(args.out_png).with_suffix(".pdf")
    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    # 条件分岐: `not out_png.is_absolute()` を満たす経路を評価する。

    if not out_png.is_absolute():
        out_png = (ROOT / out_png).resolve()

    # 条件分岐: `not out_pdf.is_absolute()` を満たす経路を評価する。

    if not out_pdf.is_absolute():
        out_pdf = (ROOT / out_pdf).resolve()

    payload = build_pack()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, payload.get("criteria") if isinstance(payload.get("criteria"), list) else [])
    _plot(out_png, payload)
    _plot(out_pdf, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")
    print(f"[ok] wrote: {_rel(out_pdf)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_born_route_a_proxy_constraints",
                "phase": "8.7.49.5",
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
