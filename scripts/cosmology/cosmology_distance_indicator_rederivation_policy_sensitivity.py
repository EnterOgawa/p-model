#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_policy_sensitivity.py

Phase 16（宇宙論）/ Step 16.4：
再接続候補探索（candidate_search）の結論が「一次ソース拘束の選び方」にどれだけ依存するかを可視化する。

ここでは最小の比較として、
  - scan（best_independent）：候補（independent-only）の中から min max|z| を選ぶ
  - tightest_fixed：independent-only の中で最小σ（tightest）の α と s_L を固定して WLS
を並べる。

入力（既存の固定出力）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_candidate_search_metrics.json
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_metrics.json
  - output/private/cosmology/cosmology_distance_duality_systematics_envelope_metrics.json（policy=category_sys のとき）
  - data/cosmology/*.json（constraints）

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_policy_sensitivity.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_policy_sensitivity_metrics.json
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

from scripts.cosmology import cosmology_distance_indicator_rederivation_candidate_search as cs  # noqa: E402
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


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(x: Any) -> Optional[float]:
    try:
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        v = float(x)
        # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。
        if not math.isfinite(v):
            return None

        return v
    except Exception:
        return None


# 関数: `_fmt` の入出力契約と処理意図を定義する。

def _fmt(x: Optional[float], *, digits: int = 3) -> str:
    # 条件分岐: `x is None or not math.isfinite(float(x))` を満たす経路を評価する。
    if x is None or not math.isfinite(float(x)):
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


# 関数: `_classify_color` の入出力契約と処理意図を定義する。

def _classify_color(abs_z: Optional[float]) -> str:
    # 条件分岐: `abs_z is None or not math.isfinite(float(abs_z))` を満たす経路を評価する。
    if abs_z is None or not math.isfinite(float(abs_z)):
        return "#cccccc"

    # 条件分岐: `abs_z < 3.0` を満たす経路を評価する。

    if abs_z < 3.0:
        return "#2ca02c"

    # 条件分岐: `abs_z < 5.0` を満たす経路を評価する。

    if abs_z < 5.0:
        return "#ffbf00"

    return "#d62728"


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Policy sensitivity of candidate_search (scan vs tightest-fixed) for Phase 16 / Step 16.4."
    )
    ap.add_argument(
        "--candidate-metrics",
        default=str(
            _ROOT
            / "output"
            / "private"
            / "cosmology"
            / "cosmology_distance_indicator_rederivation_candidate_search_metrics.json"
        ),
        help="Input candidate_search metrics JSON (default: output/private/cosmology/...candidate_search_metrics.json)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Output directory (default: output/private/cosmology)",
    )
    args = ap.parse_args(argv)

    in_metrics = Path(args.candidate_metrics)
    # 条件分岐: `not in_metrics.exists()` を満たす経路を評価する。
    if not in_metrics.exists():
        legacy = _ROOT / "output" / "cosmology" / "cosmology_distance_indicator_rederivation_candidate_search_metrics.json"
        # 条件分岐: `legacy.exists()` を満たす経路を評価する。
        if legacy.exists():
            in_metrics = legacy
        else:
            raise FileNotFoundError(
                f"missing required metrics: {in_metrics} (run scripts/summary/run_all.py --offline first)"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_japanese_font()

    cand = _read_json(in_metrics)
    ddr_sigma_policy = cand.get("ddr_sigma_policy") if isinstance(cand.get("ddr_sigma_policy"), dict) else {}
    policy = str(ddr_sigma_policy.get("policy") or "raw")

    # Load constraints (same as candidate_search) to pick tightest independent α and s_L.
    data_dir = _ROOT / "data" / "cosmology"
    in_ddr = data_dir / "distance_duality_constraints.json"
    in_opacity = data_dir / "cosmic_opacity_constraints.json"
    in_candle = data_dir / "sn_standard_candle_evolution_constraints.json"
    in_pt = data_dir / "sn_time_dilation_constraints.json"
    in_pe = data_dir / "cmb_temperature_scaling_constraints.json"
    in_bao_fit = _ROOT / "output" / "private" / "cosmology" / "cosmology_bao_scaled_distance_fit_metrics.json"

    for p in (in_ddr, in_opacity, in_candle, in_pt, in_pe, in_bao_fit):
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    ddr_rows = _read_json(in_ddr).get("constraints") or []
    opacity_rows = _read_json(in_opacity).get("constraints") or []
    candle_rows = _read_json(in_candle).get("constraints") or []
    pt_rows = _read_json(in_pt).get("constraints") or []
    pe_rows = _read_json(in_pe).get("constraints") or []
    bao_fit = _read_json(in_bao_fit)

    env = {}
    # 条件分岐: `policy == "category_sys"` を満たす経路を評価する。
    if policy == "category_sys":
        env_path = _ROOT / "output" / "private" / "cosmology" / "cosmology_distance_duality_systematics_envelope_metrics.json"
        env = cs._load_ddr_systematics_envelope(env_path) if env_path.exists() else {}

    ddr_all = [
        cs._apply_ddr_sigma_policy(cs.DDRConstraint.from_json(r), policy=policy, envelope=env) for r in ddr_rows
    ]
    opacity_all = cs._as_gaussian_list(opacity_rows, mean_key="alpha_opacity", sigma_key="alpha_opacity_sigma")
    candle_all = cs._as_gaussian_list(candle_rows, mean_key="s_L", sigma_key="s_L_sigma")
    pt_all = cs._as_gaussian_list(pt_rows, mean_key="p_t", sigma_key="p_t_sigma")
    pe_all_from_beta = cs._as_pT_constraints(pe_rows)
    # 条件分岐: `not pt_all` を満たす経路を評価する。
    if not pt_all:
        raise ValueError("no SN time dilation constraint found")

    # 条件分岐: `not pe_all_from_beta` を満たす経路を評価する。

    if not pe_all_from_beta:
        raise ValueError("no CMB temperature scaling constraint found")

    p_t = pt_all[0]
    p_e = pe_all_from_beta[0]

    try:
        sR_bao = float(bao_fit["fit"]["best_fit"]["s_R"])
        sR_bao_sigma = float(bao_fit["fit"]["best_fit"]["s_R_sigma_1d"])
    except Exception as e:
        raise ValueError("unexpected BAO fit metrics schema") from e

    # Match the sigma_scale used by candidate_search metrics (soft constraint).

    sigma_used = _safe_float(((cand.get("fixed_constraints") or {}).get("bao_s_R") or {}).get("sigma_used"))
    # 条件分岐: `sigma_used is None` を満たす経路を評価する。
    if sigma_used is None:
        sigma_used = sR_bao_sigma

    opacity_ind = [c for c in opacity_all if c.is_independent()]
    candle_ind = [c for c in candle_all if c.is_independent()]
    # 条件分岐: `not opacity_ind` を満たす経路を評価する。
    if not opacity_ind:
        raise ValueError("no independent opacity candidates found")

    # 条件分岐: `not candle_ind` を満たす経路を評価する。

    if not candle_ind:
        raise ValueError("no independent candle candidates found")

    op_tight = min(opacity_ind, key=lambda c: float(c.sigma))
    ca_tight = min(candle_ind, key=lambda c: float(c.sigma))

    # Pull scan(best_independent) from candidate_search metrics to ensure consistency.
    per_ddr_metrics = ((cand.get("results") or {}).get("per_ddr")) if isinstance(cand.get("results"), dict) else None
    # 条件分岐: `not isinstance(per_ddr_metrics, list) or not per_ddr_metrics` を満たす経路を評価する。
    if not isinstance(per_ddr_metrics, list) or not per_ddr_metrics:
        raise ValueError("invalid candidate_search metrics: results.per_ddr missing or empty")

    scan_by_id: Dict[str, Dict[str, Any]] = {}
    for r in per_ddr_metrics:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        d = r.get("ddr")
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        r_id = str(d.get("id") or "")
        # 条件分岐: `not r_id` を満たす経路を評価する。
        if not r_id:
            continue

        scan_by_id[r_id] = r.get("best_independent") if isinstance(r.get("best_independent"), dict) else {}

    rows_out: List[Dict[str, Any]] = []
    labels: List[str] = []
    scan_vals: List[float] = []
    tight_vals: List[float] = []
    scan_colors: List[str] = []
    tight_colors: List[str] = []

    for d in ddr_all:
        scan = scan_by_id.get(d.id) or {}
        scan_fit = scan.get("fit") if isinstance(scan.get("fit"), dict) else {}
        scan_z = _safe_float(scan_fit.get("max_abs_z"))
        scan_lim = str(scan_fit.get("limiting_observation") or "na")

        tight_fit = cs._wls_fit(
            ddr=d,
            sR_bao=sR_bao,
            sR_bao_sigma=float(sigma_used),
            opacity=op_tight,
            candle=ca_tight,
            p_t=p_t,
            p_e=p_e,
        )
        tight_z = _safe_float(tight_fit.get("max_abs_z"))
        tight_lim = str(tight_fit.get("limiting_observation") or "na")

        label = str(d.short_label or d.id)
        # 条件分岐: `bool(d.uses_bao)` を満たす経路を評価する。
        if bool(d.uses_bao):
            label = f"{label}（BAO含む）"

        labels.append(label)
        scan_vals.append(float(scan_z) if scan_z is not None else float("nan"))
        tight_vals.append(float(tight_z) if tight_z is not None else float("nan"))
        scan_colors.append(_classify_color(scan_z))
        tight_colors.append(_classify_color(tight_z))

        rows_out.append(
            {
                "ddr": {
                    "id": d.id,
                    "short_label": d.short_label,
                    "uses_bao": d.uses_bao,
                    "epsilon0_obs": d.epsilon0,
                    "epsilon0_sigma": d.epsilon0_sigma,
                    "epsilon0_sigma_raw": d.epsilon0_sigma_raw,
                    "sigma_sys_category": d.sigma_sys_category,
                    "sigma_policy": d.sigma_policy,
                    "category": d.category,
                },
                "scan_best_independent": {
                    "max_abs_z": scan_z,
                    "limiting_observation": scan_lim,
                    "opacity": scan.get("opacity"),
                    "candle": scan.get("candle"),
                },
                "tightest_fixed_independent": {
                    "max_abs_z": tight_z,
                    "limiting_observation": tight_lim,
                    "opacity": {"id": op_tight.id, "short_label": op_tight.short_label, "mean": op_tight.mean, "sigma": op_tight.sigma},
                    "candle": {"id": ca_tight.id, "short_label": ca_tight.short_label, "mean": ca_tight.mean, "sigma": ca_tight.sigma},
                    "fit": tight_fit,
                },
            }
        )

    # Sort by scan(best_independent) max|z| ascending for readability.

    def _z_sort(v: float) -> float:
        return v if math.isfinite(v) else float("inf")

    order = sorted(range(len(labels)), key=lambda i: (_z_sort(scan_vals[i]), _z_sort(tight_vals[i]), labels[i]))
    labels = [labels[i] for i in order]
    scan_vals = [scan_vals[i] for i in order]
    tight_vals = [tight_vals[i] for i in order]
    scan_colors = [scan_colors[i] for i in order]
    tight_colors = [tight_colors[i] for i in order]
    rows_out = [rows_out[i] for i in order]

    # --- Plot
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    y = np.arange(len(labels), dtype=float)
    h = 0.38
    fig, ax = plt.subplots(figsize=(16, 8.5))

    ax.barh(
        y - h / 2,
        scan_vals,
        height=h,
        color=scan_colors,
        alpha=0.85,
        edgecolor="#333333",
        linewidth=0.8,
        label="scan: best_independent（min max|z|）",
    )
    ax.barh(
        y + h / 2,
        tight_vals,
        height=h,
        color=tight_colors,
        alpha=0.65,
        edgecolor="#333333",
        linewidth=0.8,
        hatch="///",
        label="tightest_fixed（αとs_Lを最小σで固定）",
    )

    for x0, label in ((1.0, "1σ"), (3.0, "3σ"), (5.0, "5σ")):
        ax.axvline(x0, color="#999999", lw=1.0, ls="--", alpha=0.6)
        ax.text(x0, -0.9, label, ha="center", va="bottom", fontsize=9, color="#666666")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    x_max = 0.0
    for v in scan_vals + tight_vals:
        # 条件分岐: `math.isfinite(v)` を満たす経路を評価する。
        if math.isfinite(v):
            x_max = max(x_max, float(v))

    x_max = max(6.0, x_max * 1.15 + 0.25)
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("max|z|（DDR + BAO(s_R) + α + s_L + p_t + p_e の同時整合; WLS）")
    ax.set_title(f"宇宙論（再接続候補探索）：一次ソース選択ルールの感度（DDR σ: {policy}）", pad=18)

    legend_handles = [
        Patch(facecolor="#e0e0e0", edgecolor="#333333", label="scan: best_independent"),
        Patch(facecolor="#e0e0e0", edgecolor="#333333", hatch="///", label="tightest_fixed"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    fig.text(
        0.5,
        0.02,
        f"tightest_fixed で固定する拘束：α={op_tight.short_label} / s_L={ca_tight.short_label}（independent-only の最小σ）",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])

    out_png = out_dir / "cosmology_distance_indicator_rederivation_policy_sensitivity.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    out_json = out_dir / "cosmology_distance_indicator_rederivation_policy_sensitivity_metrics.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "candidate_search_metrics": str(in_metrics).replace("\\", "/"),
            "ddr": str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
            "opacity": str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
            "candle": str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
            "sn_time_dilation": str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
            "cmb_temperature_scaling": str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
            "bao_fit": str(in_bao_fit.relative_to(_ROOT)).replace("\\", "/"),
        },
        "ddr_sigma_policy": ddr_sigma_policy,
        "fixed_constraints": {
            "p_t": {"id": p_t.id, "short_label": p_t.short_label, "mean": p_t.mean, "sigma": p_t.sigma},
            "p_e": {"id": p_e.id, "short_label": p_e.short_label, "mean": p_e.mean, "sigma": p_e.sigma},
            "bao_s_R": {"mean": sR_bao, "sigma_used": float(sigma_used)},
            "tightest_fixed": {
                "opacity": {"id": op_tight.id, "short_label": op_tight.short_label, "mean": op_tight.mean, "sigma": op_tight.sigma},
                "candle": {"id": ca_tight.id, "short_label": ca_tight.short_label, "mean": ca_tight.mean, "sigma": ca_tight.sigma},
            },
        },
        "rows": rows_out,
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_json.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_json, payload)

    worklog.append_event(
        {
            "kind": "cosmology_distance_indicator_rederivation_policy_sensitivity",
            "script": "scripts/cosmology/cosmology_distance_indicator_rederivation_policy_sensitivity.py",
            "inputs": [in_metrics, in_ddr, in_opacity, in_candle, in_pt, in_pe, in_bao_fit],
            "outputs": [out_png, out_json],
            "ddr_sigma_policy": ddr_sigma_policy,
            "tightest_fixed_opacity": {"id": op_tight.id, "short_label": op_tight.short_label},
            "tightest_fixed_candle": {"id": ca_tight.id, "short_label": ca_tight.short_label},
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

