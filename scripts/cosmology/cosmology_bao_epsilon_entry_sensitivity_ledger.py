#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_epsilon_entry_sensitivity_ledger.py

Phase 4 / Step 4.14.4:
BAO一次統計 ε（AP warping）の「入口（手続きノブ）」がどの程度 ε を動かすかを、
既存の感度スキャン出力から **台帳（ledger）**として統合する。

狙い：
- 「理論差」ではなく「手続き差」で ε が動く混入を、定量（Δε と Δε/σ）で固定する。
- BOSS screening / DESI promotion の両方について、freeze 点の根拠（どのノブが支配的か）を
  1つの JSON に集約する。

出力（固定）：
- output/private/cosmology/cosmology_bao_epsilon_entry_sensitivity_ledger.json
- output/private/cosmology/cosmology_bao_epsilon_entry_sensitivity_ledger.png

注：
- 本スクリプトは「再計算」ではなく「既存metricsの統合」である。
  重い Corrfunc/Jackknife/RascalC 計算は、個別スクリプト（*_sensitivity.py）側で固定済み。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

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
    except Exception:
        return


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# 関数: `_sigma_from_err` の入出力契約と処理意図を定義する。

def _sigma_from_err(err_lo: float, err_hi: float) -> Optional[float]:
    # 条件分岐: `not isinstance(err_lo, (int, float)) or not isinstance(err_hi, (int, float))` を満たす経路を評価する。
    if not isinstance(err_lo, (int, float)) or not isinstance(err_hi, (int, float)):
        return None

    # 条件分岐: `not (math.isfinite(float(err_lo)) and math.isfinite(float(err_hi)))` を満たす経路を評価する。

    if not (math.isfinite(float(err_lo)) and math.isfinite(float(err_hi))):
        return None

    s = (abs(float(err_lo)) + abs(float(err_hi))) / 2.0
    # 条件分岐: `s <= 0` を満たす経路を評価する。
    if s <= 0:
        return None

    return s


# 関数: `_combine_sigma` の入出力契約と処理意図を定義する。

def _combine_sigma(s1: Optional[float], s2: Optional[float]) -> Optional[float]:
    # 条件分岐: `s1 is None and s2 is None` を満たす経路を評価する。
    if s1 is None and s2 is None:
        return None

    # 条件分岐: `s1 is None` を満たす経路を評価する。

    if s1 is None:
        return s2

    # 条件分岐: `s2 is None` を満たす経路を評価する。

    if s2 is None:
        return s1

    s = math.sqrt(float(s1) ** 2 + float(s2) ** 2)
    # 条件分岐: `not math.isfinite(s) or s <= 0` を満たす経路を評価する。
    if not math.isfinite(s) or s <= 0:
        return None

    return s


# クラス: `MaxDelta` の責務と境界条件を定義する。

@dataclass(frozen=True)
class MaxDelta:
    abs_sigma: Optional[float]
    abs_delta_eps: Optional[float]
    where: Dict[str, Any]


# 関数: `_max_delta_init` の入出力契約と処理意図を定義する。

def _max_delta_init() -> MaxDelta:
    return MaxDelta(abs_sigma=None, abs_delta_eps=None, where={})


# 関数: `_max_delta_update` の入出力契約と処理意図を定義する。

def _max_delta_update(cur: MaxDelta, *, abs_sigma: Optional[float], abs_delta_eps: Optional[float], where: Dict[str, Any]) -> MaxDelta:
    # Prefer abs_sigma as the primary ordering; fall back to abs_delta_eps.
    if cur.abs_sigma is None and abs_sigma is None:
        # 条件分岐: `cur.abs_delta_eps is None and abs_delta_eps is None` を満たす経路を評価する。
        if cur.abs_delta_eps is None and abs_delta_eps is None:
            return cur

        # 条件分岐: `cur.abs_delta_eps is None` を満たす経路を評価する。

        if cur.abs_delta_eps is None:
            return MaxDelta(abs_sigma=None, abs_delta_eps=abs_delta_eps, where=where)

        # 条件分岐: `abs_delta_eps is None` を満たす経路を評価する。

        if abs_delta_eps is None:
            return cur

        # 条件分岐: `abs_delta_eps > cur.abs_delta_eps` を満たす経路を評価する。

        if abs_delta_eps > cur.abs_delta_eps:
            return MaxDelta(abs_sigma=None, abs_delta_eps=abs_delta_eps, where=where)

        return cur

    # 条件分岐: `cur.abs_sigma is None` を満たす経路を評価する。

    if cur.abs_sigma is None:
        return MaxDelta(abs_sigma=abs_sigma, abs_delta_eps=abs_delta_eps, where=where)

    # 条件分岐: `abs_sigma is None` を満たす経路を評価する。

    if abs_sigma is None:
        return cur

    # 条件分岐: `abs_sigma > cur.abs_sigma` を満たす経路を評価する。

    if abs_sigma > cur.abs_sigma:
        return MaxDelta(abs_sigma=abs_sigma, abs_delta_eps=abs_delta_eps, where=where)

    return cur


# 関数: `_summarize_eps_map_sensitivity` の入出力契約と処理意図を定義する。

def _summarize_eps_map_sensitivity(
    metrics: Dict[str, Any],
    *,
    name: str,
    baseline_key: str,
    skip_keys: Iterable[str],
) -> Dict[str, Any]:
    eps_map = metrics.get("results", {}).get("eps", {})
    # 条件分岐: `not isinstance(eps_map, dict) or baseline_key not in eps_map` を満たす経路を評価する。
    if not isinstance(eps_map, dict) or baseline_key not in eps_map:
        return {
            "name": name,
            "status": "missing",
            "note": f"baseline_key={baseline_key} not found in results.eps",
        }

    baseline = eps_map.get(baseline_key, {})
    variants = [k for k in eps_map.keys() if k not in set(skip_keys) and k != baseline_key]

    max_delta = _max_delta_init()
    by_variant: Dict[str, Any] = {}

    for variant_key in variants:
        v = eps_map.get(variant_key, {})
        rows: List[Dict[str, Any]] = []
        for zbin_str, b_entry in baseline.items():
            # 条件分岐: `zbin_str not in v` を満たす経路を評価する。
            if zbin_str not in v:
                continue

            v_entry = v.get(zbin_str, {})

            b_eps = float(b_entry.get("eps"))
            v_eps = float(v_entry.get("eps"))
            delta = v_eps - b_eps

            b_sigma = _sigma_from_err(float(b_entry.get("err_lo", float("nan"))), float(b_entry.get("err_hi", float("nan"))))
            v_sigma = _sigma_from_err(float(v_entry.get("err_lo", float("nan"))), float(v_entry.get("err_hi", float("nan"))))
            sigma = _combine_sigma(b_sigma, v_sigma)
            abs_sigma = abs(delta) / sigma if sigma is not None else None

            row = {
                "zbin": int(b_entry.get("zbin", zbin_str)),
                "z_eff_baseline": b_entry.get("z_eff"),
                "eps_baseline": b_eps,
                "eps_variant": v_eps,
                "delta_eps": delta,
                "sigma_combined": sigma,
                "abs_sigma": abs_sigma,
            }
            rows.append(row)

            max_delta = _max_delta_update(
                max_delta,
                abs_sigma=abs_sigma,
                abs_delta_eps=abs(delta),
                where={"variant": variant_key, "zbin": row["zbin"], "delta_eps": delta, "abs_sigma": abs_sigma},
            )

        by_variant[variant_key] = {
            "n_bins": len(rows),
            "rows": rows,
        }

    return {
        "name": name,
        "status": "ok",
        "baseline_key": baseline_key,
        "variant_keys": variants,
        "max_delta": {
            "abs_sigma": max_delta.abs_sigma,
            "abs_delta_eps": max_delta.abs_delta_eps,
            "where": max_delta.where,
        },
        "by_variant": by_variant,
    }


# 関数: `_summarize_coordinate_spec_sensitivity` の入出力契約と処理意図を定義する。

def _summarize_coordinate_spec_sensitivity(metrics: Dict[str, Any]) -> Dict[str, Any]:
    base_tag = metrics.get("inputs", {}).get("base_out_tag")
    variant_tags = metrics.get("inputs", {}).get("variant_out_tags", [])
    points = metrics.get("points", [])
    # 条件分岐: `not isinstance(base_tag, str) or not isinstance(variant_tags, list) or not is...` を満たす経路を評価する。
    if not isinstance(base_tag, str) or not isinstance(variant_tags, list) or not isinstance(points, list):
        return {"name": "DESI coordinate spec sensitivity", "status": "missing", "note": "unexpected schema"}

    # 関数: `key` の入出力契約と処理意図を定義する。

    def key(p: Dict[str, Any]) -> Tuple[str, float, float, str]:
        return (str(p.get("dist")), float(p.get("z_min")), float(p.get("z_max")), str(p.get("out_tag")))

    # Index by (dist,zmin,zmax,out_tag) to allow direct matching.

    idx: Dict[Tuple[str, float, float, str], Dict[str, Any]] = {}
    for p in points:
        try:
            idx[key(p)] = p
        except Exception:
            continue

    # Base points by (dist,zmin,zmax)

    base_points: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    for p in points:
        # 条件分岐: `p.get("out_tag") != base_tag` を満たす経路を評価する。
        if p.get("out_tag") != base_tag:
            continue

        try:
            base_points[(str(p.get("dist")), float(p.get("z_min")), float(p.get("z_max")))] = p
        except Exception:
            continue

    max_delta = _max_delta_init()
    by_variant: Dict[str, Any] = {}

    for vtag in variant_tags:
        rows: List[Dict[str, Any]] = []
        for k_base, b in base_points.items():
            dist, zmin, zmax = k_base
            v = idx.get((dist, zmin, zmax, str(vtag)))
            # 条件分岐: `v is None` を満たす経路を評価する。
            if v is None:
                continue

            b_eps = float(b.get("eps"))
            v_eps = float(v.get("eps"))
            delta = v_eps - b_eps

            b_sigma = float(b.get("sigma_eps_1sigma")) if b.get("sigma_eps_1sigma") is not None else None
            v_sigma = float(v.get("sigma_eps_1sigma")) if v.get("sigma_eps_1sigma") is not None else None
            sigma = _combine_sigma(b_sigma, v_sigma)
            abs_sigma = abs(delta) / sigma if sigma is not None else None

            row = {
                "dist": dist,
                "z_min": zmin,
                "z_max": zmax,
                "eps_baseline": b_eps,
                "eps_variant": v_eps,
                "delta_eps": delta,
                "sigma_combined": sigma,
                "abs_sigma": abs_sigma,
            }
            rows.append(row)

            max_delta = _max_delta_update(
                max_delta,
                abs_sigma=abs_sigma,
                abs_delta_eps=abs(delta),
                where={"variant": vtag, "dist": dist, "z_min": zmin, "z_max": zmax, "delta_eps": delta, "abs_sigma": abs_sigma},
            )

        by_variant[str(vtag)] = {"n_points": len(rows), "rows": rows}

    return {
        "name": "DESI coordinate spec sensitivity",
        "status": "ok",
        "baseline_out_tag": base_tag,
        "variant_out_tags": [str(x) for x in variant_tags],
        "max_delta": {
            "abs_sigma": max_delta.abs_sigma,
            "abs_delta_eps": max_delta.abs_delta_eps,
            "where": max_delta.where,
        },
        "by_variant": by_variant,
    }


# 関数: `_summarize_peakfit_settings_sensitivity` の入出力契約と処理意図を定義する。

def _summarize_peakfit_settings_sensitivity(metrics: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = metrics.get("results", [])
    # 条件分岐: `not isinstance(scenarios, list)` を満たす経路を評価する。
    if not isinstance(scenarios, list):
        return {"name": "DESI peakfit settings sensitivity", "status": "missing", "note": "unexpected schema"}

    base_scenario_id = "base"
    base_case: Optional[Dict[str, Any]] = None
    scenario_by_id: Dict[str, Dict[str, Any]] = {}
    for s in scenarios:
        sid = s.get("scenario", {}).get("id")
        # 条件分岐: `isinstance(sid, str)` を満たす経路を評価する。
        if isinstance(sid, str):
            scenario_by_id[sid] = s

        # 条件分岐: `sid == base_scenario_id` を満たす経路を評価する。

        if sid == base_scenario_id:
            base_case = s

    # 条件分岐: `base_case is None and scenarios` を満たす経路を評価する。

    if base_case is None and scenarios:
        base_case = scenarios[0]
        base_scenario_id = str(base_case.get("scenario", {}).get("id", "base?"))

    # 条件分岐: `base_case is None` を満たす経路を評価する。

    if base_case is None:
        return {"name": "DESI peakfit settings sensitivity", "status": "missing", "note": "no scenarios"}

    # 関数: `bkey` の入出力契約と処理意図を定義する。

    def bkey(r: Dict[str, Any]) -> Tuple[str, str]:
        # (dist, z_range_key)
        zr = r.get("z_range", {})
        return (str(r.get("dist")), str(zr.get("key")))

    base_rows = {bkey(r): r for r in base_case.get("results", []) if isinstance(r, dict)}

    max_delta = _max_delta_init()
    by_variant: Dict[str, Any] = {}

    for sid, s in scenario_by_id.items():
        # 条件分岐: `sid == base_scenario_id` を満たす経路を評価する。
        if sid == base_scenario_id:
            continue

        rows: List[Dict[str, Any]] = []
        for k, b in base_rows.items():
            dist, zr_key = k
            match: Optional[Dict[str, Any]] = None
            for r in s.get("results", []):
                # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
                if not isinstance(r, dict):
                    continue

                # 条件分岐: `bkey(r) == k` を満たす経路を評価する。

                if bkey(r) == k:
                    match = r
                    break

            # 条件分岐: `match is None` を満たす経路を評価する。

            if match is None:
                continue

            b_eps = float(b.get("fit", {}).get("free", {}).get("eps"))
            v_eps = float(match.get("fit", {}).get("free", {}).get("eps"))
            delta = v_eps - b_eps

            b_sigma = float(b.get("screening", {}).get("sigma_eps_1sigma")) if b.get("screening", {}).get("sigma_eps_1sigma") is not None else None
            v_sigma = float(match.get("screening", {}).get("sigma_eps_1sigma")) if match.get("screening", {}).get("sigma_eps_1sigma") is not None else None
            sigma = _combine_sigma(b_sigma, v_sigma)
            abs_sigma = abs(delta) / sigma if sigma is not None else None

            zr = b.get("z_range", {})
            row = {
                "dist": dist,
                "z_range": {"key": zr_key, "label": zr.get("label"), "z_min": zr.get("z_min"), "z_max": zr.get("z_max")},
                "eps_baseline": b_eps,
                "eps_variant": v_eps,
                "delta_eps": delta,
                "sigma_combined": sigma,
                "abs_sigma": abs_sigma,
            }
            rows.append(row)

            max_delta = _max_delta_update(
                max_delta,
                abs_sigma=abs_sigma,
                abs_delta_eps=abs(delta),
                where={"variant": sid, "dist": dist, "z_range_key": zr_key, "delta_eps": delta, "abs_sigma": abs_sigma},
            )

        by_variant[sid] = {"n_points": len(rows), "rows": rows}

    return {
        "name": "DESI peakfit settings sensitivity",
        "status": "ok",
        "baseline_scenario_id": base_scenario_id,
        "variant_scenario_ids": [k for k in scenario_by_id.keys() if k != base_scenario_id],
        "max_delta": {
            "abs_sigma": max_delta.abs_sigma,
            "abs_delta_eps": max_delta.abs_delta_eps,
            "where": max_delta.where,
        },
        "by_variant": by_variant,
    }


# 関数: `_summarize_desi_promotion_check` の入出力契約と処理意図を定義する。

def _summarize_desi_promotion_check(d: Dict[str, Any]) -> Dict[str, Any]:
    result = d.get("result", {})
    return {
        "name": "DESI promotion gate (multi-tracer)",
        "status": "ok" if isinstance(result, dict) else "missing",
        "promoted": result.get("promoted") if isinstance(result, dict) else None,
        "passing_tracers": result.get("passing_tracers") if isinstance(result, dict) else None,
        "passing_tracers_n": result.get("passing_tracers_n") if isinstance(result, dict) else None,
        "threshold_abs": d.get("params", {}).get("threshold_abs"),
        "target_dist": d.get("params", {}).get("target_dist"),
    }


# 関数: `_plot_max_sigmas` の入出力契約と処理意図を定義する。

def _plot_max_sigmas(studies: List[Dict[str, Any]], out_png: Path) -> Optional[str]:
    # Keep only rows with finite abs_sigma.
    rows: List[Tuple[str, float]] = []
    for s in studies:
        md = s.get("max_delta", {})
        v = md.get("abs_sigma")
        # 条件分岐: `isinstance(v, (int, float)) and math.isfinite(float(v))` を満たす経路を評価する。
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            rows.append((str(s.get("name")), float(v)))

    # 条件分岐: `not rows` を満たす経路を評価する。

    if not rows:
        return None

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig_h = max(3.5, 0.45 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = list(range(len(rows)))
    ax.barh(y, vals, color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("max |Δε| / σ (combined)")
    ax.set_title("BAO ε entry sensitivity ledger (procedure knobs)")
    ax.axvline(1.0, color="#888", ls="--", lw=1.0)
    ax.axvline(3.0, color="#888", ls="--", lw=1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return str(out_png)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weight-scheme-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_catalog_weight_scheme_sensitivity_metrics.json"),
        help="Path to cosmology_bao_catalog_weight_scheme_sensitivity_metrics.json",
    )
    p.add_argument(
        "--random-max-rows-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_catalog_random_max_rows_sensitivity_metrics.json"),
        help="Path to cosmology_bao_catalog_random_max_rows_sensitivity_metrics.json",
    )
    p.add_argument(
        "--desi-peakfit-settings-metrics",
        default=str(
            _ROOT
            / "output"
            / "cosmology"
            / "cosmology_bao_catalog_peakfit_settings_sensitivity__lrg_combined__w_desi_default_ms_off_y1bins_metrics.json"
        ),
        help="Path to cosmology_bao_catalog_peakfit_settings_sensitivity__*.json",
    )
    p.add_argument(
        "--desi-coordinate-spec-metrics",
        default=str(
            _ROOT
            / "output"
            / "cosmology"
            / "cosmology_bao_catalog_coordinate_spec_sensitivity__lrg_combined__w_desi_default_ms_off_y1bins_metrics.json"
        ),
        help="Path to cosmology_bao_catalog_coordinate_spec_sensitivity__*.json",
    )
    p.add_argument(
        "--desi-promotion-check",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_desi_dr1_bao_promotion_check.json"),
        help="Path to cosmology_desi_dr1_bao_promotion_check.json",
    )
    p.add_argument(
        "--out-json",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_epsilon_entry_sensitivity_ledger.json"),
        help="Output json path",
    )
    p.add_argument(
        "--out-png",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_epsilon_entry_sensitivity_ledger.png"),
        help="Output png path",
    )
    args = p.parse_args()

    _set_japanese_font()

    inputs: Dict[str, Any] = {}
    studies: List[Dict[str, Any]] = []

    # BOSS: weight scheme
    ws_path = Path(args.weight_scheme_metrics).resolve()
    # 条件分岐: `ws_path.exists()` を満たす経路を評価する。
    if ws_path.exists():
        inputs["weight_scheme_metrics"] = str(ws_path)
        ws = _read_json(ws_path)
        studies.append(
            _summarize_eps_map_sensitivity(
                ws,
                name="BOSS weight scheme sensitivity (boss_default vs fkp_only/none)",
                baseline_key="boss_default",
                skip_keys=["published_post"],
            )
        )
    else:
        studies.append({"name": "BOSS weight scheme sensitivity", "status": "missing", "path": str(ws_path)})

    # BOSS: random sampling

    rm_path = Path(args.random_max_rows_metrics).resolve()
    # 条件分岐: `rm_path.exists()` を満たす経路を評価する。
    if rm_path.exists():
        inputs["random_max_rows_metrics"] = str(rm_path)
        rm = _read_json(rm_path)
        studies.append(
            _summarize_eps_map_sensitivity(
                rm,
                name="BOSS random sampling sensitivity (reservoir vs prefix_rows)",
                baseline_key="baseline",
                skip_keys=["published_post"],
            )
        )
    else:
        studies.append({"name": "BOSS random sampling sensitivity", "status": "missing", "path": str(rm_path)})

    # DESI: peakfit settings

    ps_path = Path(args.desi_peakfit_settings_metrics).resolve()
    # 条件分岐: `ps_path.exists()` を満たす経路を評価する。
    if ps_path.exists():
        inputs["desi_peakfit_settings_metrics"] = str(ps_path)
        ps = _read_json(ps_path)
        studies.append(_summarize_peakfit_settings_sensitivity(ps))
    else:
        studies.append({"name": "DESI peakfit settings sensitivity", "status": "missing", "path": str(ps_path)})

    # DESI: coordinate spec

    cs_path = Path(args.desi_coordinate_spec_metrics).resolve()
    # 条件分岐: `cs_path.exists()` を満たす経路を評価する。
    if cs_path.exists():
        inputs["desi_coordinate_spec_metrics"] = str(cs_path)
        cs = _read_json(cs_path)
        studies.append(_summarize_coordinate_spec_sensitivity(cs))
    else:
        studies.append({"name": "DESI coordinate spec sensitivity", "status": "missing", "path": str(cs_path)})

    # DESI: promotion gate

    promo_path = Path(args.desi_promotion_check).resolve()
    # 条件分岐: `promo_path.exists()` を満たす経路を評価する。
    if promo_path.exists():
        inputs["desi_promotion_check"] = str(promo_path)
        promo = _read_json(promo_path)
        studies.append(_summarize_desi_promotion_check(promo))
    else:
        studies.append({"name": "DESI promotion gate", "status": "missing", "path": str(promo_path)})

    out_json = Path(args.out_json).resolve()
    out_png = Path(args.out_png).resolve()

    ledger: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "Phase 4 / Step 4.14.4 (BAO epsilon entry sensitivity ledger)",
        "inputs": inputs,
        "policy": {
            "metric": "max |Δε| / σ (combined; sqrt(σ1^2+σ2^2)) per study",
            "thresholds": {"ok_max": 1.0, "mixed_max": 3.0},
            "note": "This ledger quantifies procedure-induced ε shifts. It is not a claim that any particular baseline is correct.",
        },
        "studies": studies,
        "outputs": {"json": str(out_json), "png": str(out_png)},
    }

    _write_json(out_json, ledger)
    png_written = _plot_max_sigmas([s for s in studies if isinstance(s, dict)], out_png)
    # 条件分岐: `png_written is None` を満たす経路を評価する。
    if png_written is None:
        ledger["outputs"]["png"] = None
        _write_json(out_json, ledger)

    worklog.append_event(
        {
            "event_type": "cosmology_bao_epsilon_entry_sensitivity_ledger",
            "generated_utc": ledger["generated_utc"],
            "inputs": inputs,
            "outputs": ledger["outputs"],
            "note": "Ledger synthesized from existing sensitivity metrics to freeze BAO ε entry procedure dependence.",
        }
    )


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

