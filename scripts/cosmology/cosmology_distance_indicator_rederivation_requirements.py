#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_requirements.py

Step 14.2.21（距離指標の再導出要件の整理）:
静的背景P（最小: ε0=-1）で DDR 制約を回避するために、
距離指標（SNe Ia / BAO / DDR）の「何を捨てる/置き換える必要があるか」を
既存の固定 metrics（到達限界・誤差予算感度・再接続の張力）から1枚にまとめて固定する。

狙い:
  - “誤差で隠す”の上限（z_limit）と、“機構で埋める”場合に必要な規模（Δε, Δμ, τ）を同じ土俵で示す。
  - その上で、距離指標が暗黙に使っている仮定（標準光源/標準定規/光子保存/校正）を明示し、
    P-model側で再導出すべき対象を固定する。

入力（固定: 既存出力を参照）:
  - output/private/cosmology/cosmology_distance_indicator_reach_limit_metrics.json
  - output/private/cosmology/cosmology_distance_indicator_error_budget_sensitivity_metrics.json
  - output/private/cosmology/cosmology_reconnection_plausibility_metrics.json

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_requirements.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_requirements_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。
def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_fmt_float` の入出力契約と処理意図を定義する。

def _fmt_float(x: Optional[float], *, digits: int = 6) -> str:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return ""

    # 条件分岐: `not math.isfinite(float(x))` を満たす経路を評価する。

    if not math.isfinite(float(x)):
        return ""

    x = float(x)
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_find_value_at_z` の入出力契約と処理意図を定義する。

def _find_value_at_z(values_at_z: List[Dict[str, Any]], z: float) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_d = float("inf")
    for v in values_at_z:
        try:
            zz = float(v.get("z"))
        except Exception:
            continue

        d = abs(zz - float(z))
        # 条件分岐: `d < best_d` を満たす経路を評価する。
        if d < best_d:
            best_d = d
            best = dict(v)

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        return None

    # 条件分岐: `best_d > 1e-6` を満たす経路を評価する。

    if best_d > 1e-6:
        return None

    return best


# 関数: `_compute_delta_mu_curve` の入出力契約と処理意図を定義する。

def _compute_delta_mu_curve(delta_eps: float, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    # extra_dl_factor = (1+z)^Δε, Δμ = 5 log10(extra_dl_factor)
    return 5.0 * np.log10(np.maximum(1e-300, (1.0 + z) ** float(delta_eps)))


# 関数: `_plot_figure` の入出力契約と処理意図を定義する。

def _plot_figure(
    *,
    out_png: Path,
    reach: Dict[str, Any],
    z_limit: Dict[str, Any],
    plausibility: Dict[str, Any],
    scan_best: Optional[Dict[str, Any]] = None,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    # Representative labels/values
    rep_bao = reach["bao"]
    rep_no_bao = reach["no_bao"]
    delta_eps_bao = float(rep_bao["delta_eps_needed"])
    delta_eps_no_bao = float(rep_no_bao["delta_eps_needed"])

    v_bao_z1 = _find_value_at_z(list(rep_bao.get("values_at_z") or []), 1.0) or {}
    v_no_z1 = _find_value_at_z(list(rep_no_bao.get("values_at_z") or []), 1.0) or {}

    zlim_bao_3s = float(z_limit["bao"]["3.0sigma"])
    zlim_no_3s = float(z_limit["no_bao"]["3.0sigma"])

    scan_no_bao_note = ""
    try:
        # 条件分岐: `isinstance(scan_best, dict)` を満たす経路を評価する。
        if isinstance(scan_best, dict):
            bw = scan_best.get("bin_width")
            nmin = scan_best.get("sn_min_points")
            focus = scan_best.get("focus")
            bits: List[str] = []
            # 条件分岐: `isinstance(focus, str) and focus.strip()` を満たす経路を評価する。
            if isinstance(focus, str) and focus.strip():
                # Keep only the dataset label (avoid long/garbled suffixes in some environments).
                label = focus.strip().split(" / ")[0].strip()
                # 条件分岐: `label` を満たす経路を評価する。
                if label:
                    bits.append(label)

            # 条件分岐: `bw is not None` を満たす経路を評価する。

            if bw is not None:
                bits.append(f"bin幅={_fmt_float(float(bw), digits=3)}")

            # 条件分岐: `nmin is not None` を満たす経路を評価する。

            if nmin is not None:
                bits.append(f"n≥{int(nmin)}")

            # 条件分岐: `bits` を満たす経路を評価する。

            if bits:
                scan_no_bao_note = "（最楽観スキャン: " + "; ".join(bits) + "）"
    except Exception:
        scan_no_bao_note = ""

    # Prepare curves

    z = np.linspace(0.0, 2.3, 240)
    dmu_bao = _compute_delta_mu_curve(delta_eps_bao, z)
    dmu_no = _compute_delta_mu_curve(delta_eps_no_bao, z)

    fig = plt.figure(figsize=(16, 9.2))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.05, 0.95), height_ratios=(0.62, 0.38))

    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[:, 1])
    ax_text.axis("off")

    # Top-left: required Δμ(z) curves + z_limit markers
    ax.plot(z, dmu_bao, color="#1f77b4", linewidth=2.2, label=f"代表（BAO含む）：Δε={_fmt_float(delta_eps_bao, digits=3)}")
    ax.plot(
        z,
        dmu_no,
        color="#ff7f0e",
        linewidth=2.2,
        label=f"代表（BAOなし）：Δε={_fmt_float(delta_eps_no_bao, digits=3)}",
    )
    ax.axvline(zlim_bao_3s, color="#1f77b4", linestyle="--", alpha=0.6)
    ax.axvline(zlim_no_3s, color="#ff7f0e", linestyle="--", alpha=0.6)
    ax.text(
        zlim_bao_3s + 0.02,
        0.05,
        f"z_limit≈{_fmt_float(zlim_bao_3s, digits=3)}（3σ）",
        color="#1f77b4",
        fontsize=9,
        rotation=90,
        va="bottom",
        ha="left",
        alpha=0.85,
    )
    ax.text(
        zlim_no_3s + 0.02,
        0.05,
        f"z_limit≈{_fmt_float(zlim_no_3s, digits=3)}（3σ）",
        color="#ff7f0e",
        fontsize=9,
        rotation=90,
        va="bottom",
        ha="left",
        alpha=0.85,
    )

    ax.set_title("DDR再接続に必要な距離モジュラス補正 |Δμ(z)| と “誤差で隠せる” 上限", fontsize=12)
    ax.set_xlabel("赤方偏移 z", fontsize=11)
    ax.set_ylabel("|Δμ(z)| [mag]", fontsize=11)
    ax.set_xlim(0.0, 2.3)
    ax.set_ylim(0.0, max(float(np.nanmax(dmu_bao)), float(np.nanmax(dmu_no))) * 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.text(
        0.99,
        0.02,
        "注: z_limit は距離指標の誤差予算（opt_total）から見積もった“隠せる”上限（3σ）。\n"
        "z>z_limit では、指標の定義/校正そのものの再導出が必要。",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#dddddd", alpha=0.9),
    )

    # Bottom-left: compact numeric summary (table-like bars)
    ax2.axis("off")
    ax2.set_title("代表点（z=1）と単独機構の張力（一次ソース）", fontsize=12)

    # Single-mechanism requirements (computed for BAO representative in plausibility metrics)
    req_single = plausibility.get("required_single_mechanism", {})
    z_scores = plausibility.get("z_scores", {})

    alpha_req = (req_single.get("opacity_only") or {}).get("alpha_opacity")
    sL_req = (req_single.get("candle_only") or {}).get("s_L")
    rdrag_req = (req_single.get("ruler_only") or {}).get("r_drag_required_z1_mpc")

    z_opacity = z_scores.get("opacity_only_alpha_primary")
    z_candle = z_scores.get("candle_only_s_L_primary")
    z_rdrag = z_scores.get("ruler_only_r_drag_z1_primary")

    rows = [
        [
            "BAO含む（代表）",
            f"Δε={_fmt_float(delta_eps_bao, digits=3)}",
            f"Δμ(z=1)={_fmt_float(v_bao_z1.get('delta_mu_mag'), digits=3)} mag",
            f"τ(z=1)={_fmt_float(v_bao_z1.get('tau_equivalent_dimming'), digits=3)}",
        ],
        [
            "BAOなし（代表）",
            f"Δε={_fmt_float(delta_eps_no_bao, digits=3)}",
            f"Δμ(z=1)={_fmt_float(v_no_z1.get('delta_mu_mag'), digits=3)} mag",
            f"τ(z=1)={_fmt_float(v_no_z1.get('tau_equivalent_dimming'), digits=3)}",
        ],
    ]

    header = ["代表制約", "必要補正", "距離補正（z=1）", "等価減光（z=1）"]
    table = ax2.table(
        cellText=rows,
        colLabels=header,
        loc="upper center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.18, 0.30, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        # 条件分岐: `r == 0` を満たす経路を評価する。
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold", color="#222222")
        else:
            cell.set_facecolor("#ffffff")

    text2 = (
        "単独機構で Δε を埋める（BAO含む代表）:\n"
        f"  不透明度のみ: α≈{_fmt_float(alpha_req, digits=3)}（約{_fmt_float(abs(z_opacity), digits=3)}σ）\n"
        f"  標準光源のみ: s_L≈{_fmt_float(sL_req, digits=3)}（約{_fmt_float(abs(z_candle), digits=3)}σ）\n"
        f"  標準定規のみ: r_d(z=1)≈{_fmt_float(rdrag_req, digits=3)} Mpc（約{_fmt_float(abs(z_rdrag), digits=3)}σ）"
    )
    ax2.text(
        0.02,
        0.02,
        text2,
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#222222",
    )

    # Right panel: “what to re-derive / replace” checklist
    assumptions = [
        ("光子数保存（透明）", "有効不透明度 τ(z) を一次ソースと整合する形で再導出（散乱・吸収・選択効果の切り分け）。"),
        ("標準光源（SNe Ia）", "標準化（SALT2等）と光度進化 L(z) の扱いを P-model で再導出（距離スケール H0 と退化）。"),
        ("標準定規（BAO）", "物理スケール r_d の“固定”を置き換えるなら、r_d(z) の生成機構と銀河統計への影響を再導出。"),
        ("距離指標の幾何", "D_L/D_A の定義（DDR）を、観測で使う距離の構成と合わせて再定義（FRW前提の混入を点検）。"),
        ("独立プローブ", "SN time dilation / CMB T(z) / Alcock-Paczynski 等と同時に整合することが必要条件。"),
    ]

    lines = [
        "距離指標を静的背景Pで再導出する要件（固定）",
        "",
        "1) “誤差で隠す”の上限（opt_total 3σ, envelope）:",
        f"   - BAO含む代表: z_limit≈{_fmt_float(zlim_bao_3s, digits=3)}",
        f"   - BAOなし代表: z_limit≈{_fmt_float(zlim_no_3s, digits=3)} {scan_no_bao_note}".rstrip(),
        "",
        "2) z=1 で必要になる補正の規模（代表）:",
        f"   - BAO含む: |Δμ|≈{_fmt_float(v_bao_z1.get('delta_mu_mag'), digits=3)} mag",
        f"   - BAOなし: |Δμ|≈{_fmt_float(v_no_z1.get('delta_mu_mag'), digits=3)} mag",
        "",
        "3) 再導出/置換が必要な前提（距離指標の構成）:",
    ]
    for title, desc in assumptions:
        lines.append(f"   - {title}: {desc}")

    lines.extend(
        [
            "",
            "結論:",
            "  z>z_limit では、現行の距離指標（標準化・校正）のまま“追加補正”で埋めるのは困難。",
            "  静的無限空間として成立させるには、距離指標そのものを P-model 側で再導出し、独立プローブと同時に満たす必要がある。",
        ]
    )
    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        color="#111111",
        linespacing=1.35,
    )

    fig.suptitle("宇宙論（距離指標）：静的背景Pで DDR 制約を回避するための再導出要件", fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.parse_args(argv)

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_png = out_dir / "cosmology_distance_indicator_rederivation_requirements.png"
    out_metrics = out_dir / "cosmology_distance_indicator_rederivation_requirements_metrics.json"

    in_reach = _ROOT / "output" / "private" / "cosmology" / "cosmology_distance_indicator_reach_limit_metrics.json"
    in_sens = (
        _ROOT / "output" / "private" / "cosmology" / "cosmology_distance_indicator_error_budget_sensitivity_metrics.json"
    )
    in_plaus = _ROOT / "output" / "private" / "cosmology" / "cosmology_reconnection_plausibility_metrics.json"

    # 条件分岐: `not in_reach.exists()` を満たす経路を評価する。
    if not in_reach.exists():
        raise FileNotFoundError(f"missing input: {in_reach}")

    # 条件分岐: `not in_sens.exists()` を満たす経路を評価する。

    if not in_sens.exists():
        raise FileNotFoundError(f"missing input: {in_sens}")

    # 条件分岐: `not in_plaus.exists()` を満たす経路を評価する。

    if not in_plaus.exists():
        raise FileNotFoundError(f"missing input: {in_plaus}")

    reach_all = _read_json(in_reach)
    sens_all = _read_json(in_sens)
    plaus_all = _read_json(in_plaus)

    reach = dict(reach_all.get("reach") or {})
    # 条件分岐: `not (isinstance(reach, dict) and "bao" in reach and "no_bao" in reach)` を満たす経路を評価する。
    if not (isinstance(reach, dict) and "bao" in reach and "no_bao" in reach):
        raise ValueError("unexpected reach metrics schema (need reach.bao/no_bao)")

    envelope = dict((sens_all.get("envelope") or {}).get("opt_total") or {})
    # 条件分岐: `not (isinstance(envelope, dict) and "bao" in envelope and "no_bao" in envelope)` を満たす経路を評価する。
    if not (isinstance(envelope, dict) and "bao" in envelope and "no_bao" in envelope):
        raise ValueError("unexpected sensitivity metrics schema (need envelope.opt_total.bao/no_bao)")

    scan_best: Optional[Dict[str, Any]] = None
    try:
        scan = sens_all.get("scan")
        # 条件分岐: `isinstance(scan, dict)` を満たす経路を評価する。
        if isinstance(scan, dict):
            best = scan.get("best")
            # 条件分岐: `isinstance(best, dict)` を満たす経路を評価する。
            if isinstance(best, dict):
                scan_best = dict(best)
                scan_best["focus"] = scan.get("focus")
    except Exception:
        scan_best = None

    plaus_metrics = dict(plaus_all.get("metrics") or {})
    # 条件分岐: `"required_single_mechanism" not in plaus_metrics or "z_scores" not in plaus_m...` を満たす経路を評価する。
    if "required_single_mechanism" not in plaus_metrics or "z_scores" not in plaus_metrics:
        raise ValueError("unexpected plausibility metrics schema (need metrics.required_single_mechanism / z_scores)")

    _plot_figure(
        out_png=out_png,
        reach={"bao": reach["bao"], "no_bao": reach["no_bao"]},
        z_limit={"bao": envelope["bao"], "no_bao": envelope["no_bao"]},
        plausibility={
            "required_single_mechanism": plaus_metrics.get("required_single_mechanism") or {},
            "z_scores": plaus_metrics.get("z_scores") or {},
        },
        scan_best=scan_best,
    )

    # Build metrics JSON for reproducibility
    def _extract_z1(r: Dict[str, Any]) -> Dict[str, Any]:
        v = _find_value_at_z(list(r.get("values_at_z") or []), 1.0) or {}
        return {
            "delta_mu_mag": v.get("delta_mu_mag"),
            "tau_equivalent_dimming": v.get("tau_equivalent_dimming"),
            "extra_dl_factor": v.get("extra_dl_factor"),
        }

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "goal": "distance-indicator re-derivation requirements (static background P, DDR reconnection)",
            "z_limit": "max z where required |Δμ(z)| can be hidden within distance-indicator error budget (opt_total, envelope)",
            "delta_eps": "Δε ≡ ε0_obs - (-1) (static minimal) for representative DDR constraints",
        },
        "inputs": {
            "reach_limit_metrics": str(in_reach.relative_to(_ROOT)).replace("\\", "/"),
            "error_budget_sensitivity_metrics": str(in_sens.relative_to(_ROOT)).replace("\\", "/"),
            "reconnection_plausibility_metrics": str(in_plaus.relative_to(_ROOT)).replace("\\", "/"),
        },
        "representatives": {
            "bao": {
                "id": reach["bao"].get("id"),
                "short_label": reach["bao"].get("short_label"),
                "epsilon0_obs": reach["bao"].get("epsilon0_obs"),
                "epsilon0_sigma": reach["bao"].get("epsilon0_sigma"),
                "delta_eps_needed": reach["bao"].get("delta_eps_needed"),
                "z1": _extract_z1(reach["bao"]),
            },
            "no_bao": {
                "id": reach["no_bao"].get("id"),
                "short_label": reach["no_bao"].get("short_label"),
                "epsilon0_obs": reach["no_bao"].get("epsilon0_obs"),
                "epsilon0_sigma": reach["no_bao"].get("epsilon0_sigma"),
                "delta_eps_needed": reach["no_bao"].get("delta_eps_needed"),
                "z1": _extract_z1(reach["no_bao"]),
            },
        },
        "z_limit_envelope_opt_total": envelope,
        "z_limit_scan_best": scan_best,
        "single_mechanism_tension_primary": {
            "opacity_only": {
                **(plaus_metrics.get("required_single_mechanism") or {}).get("opacity_only", {}),
                "z_score": (plaus_metrics.get("z_scores") or {}).get("opacity_only_alpha_primary"),
            },
            "candle_only": {
                **(plaus_metrics.get("required_single_mechanism") or {}).get("candle_only", {}),
                "z_score": (plaus_metrics.get("z_scores") or {}).get("candle_only_s_L_primary"),
            },
            "ruler_only": {
                **(plaus_metrics.get("required_single_mechanism") or {}).get("ruler_only", {}),
                "z_score": (plaus_metrics.get("z_scores") or {}).get("ruler_only_r_drag_z1_primary"),
            },
        },
        "assumptions_to_rederive": [
            {
                "id": "photon_conservation",
                "label": "光子数保存（透明）",
                "replace_with": "有効不透明度 τ(z) の導出（散乱/吸収/選択効果の切り分け）",
            },
            {
                "id": "standard_candle_snia",
                "label": "標準光源（SNe Ia）",
                "replace_with": "標準化・校正・光度進化 L(z) の再導出（H0スケールとの退化を含む）",
            },
            {
                "id": "standard_ruler_bao",
                "label": "標準定規（BAO）",
                "replace_with": "r_d(z) の生成機構と観測量（D_M/r_d, H r_d）への影響の再導出",
            },
            {
                "id": "distance_geometry",
                "label": "距離指標の幾何（DDR）",
                "replace_with": "D_L/D_A の定義と観測で使う距離構成の再点検（FRW前提の混入排除）",
            },
            {
                "id": "independent_probes",
                "label": "独立プローブ",
                "replace_with": "SN time dilation / CMB T(z) / AP などとの同時整合（必要条件）",
            },
        ],
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_metrics.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_metrics, metrics)

    worklog.append_event(
        {
            "kind": "cosmology",
            "step": "14.2.21",
            "task": "distance_indicator_rederivation_requirements",
            "inputs": [
                str(in_reach.relative_to(_ROOT)).replace("\\", "/"),
                str(in_sens.relative_to(_ROOT)).replace("\\", "/"),
                str(in_plaus.relative_to(_ROOT)).replace("\\", "/"),
            ],
            "outputs": {"png": out_png, "metrics_json": out_metrics},
            "metrics": {
                "z_limit_opt_total_3sigma": {
                    "bao": envelope["bao"].get("3.0sigma"),
                    "no_bao": envelope["no_bao"].get("3.0sigma"),
                },
                "z_limit_scan_best": scan_best,
                "delta_eps_needed": {
                    "bao": reach["bao"].get("delta_eps_needed"),
                    "no_bao": reach["no_bao"].get("delta_eps_needed"),
                },
            },
        }
    )

    print(f"[OK] wrote: {out_png}")
    print(f"[OK] wrote: {out_metrics}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
