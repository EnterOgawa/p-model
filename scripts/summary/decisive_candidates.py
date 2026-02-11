from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_num(x: Any, *, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "" if x is None else str(x)
    if not math.isfinite(v):
        return ""
    if v == 0.0:
        return "0"
    av = abs(v)
    if av < 1e-3 or av >= 1e5:
        return f"{v:.{digits}g}"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def _load_decisive_falsification(root: Path) -> Dict[str, Any]:
    path = root / "output" / "private" / "summary" / "decisive_falsification.json"
    if not path.exists():
        return {}
    try:
        j = _read_json(path)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}


def _load_eht_paper5_m3_rescue_metrics(root: Path) -> Dict[str, Any]:
    path = root / "output" / "private" / "eht" / "eht_sgra_paper5_m3_nir_reconnection_conditions_metrics.json"
    if not path.exists():
        return {}
    try:
        j = _read_json(path)
    except Exception:
        return {}
    if not isinstance(j, dict):
        return {}

    m3 = ((j.get("derived") or {}).get("m3") or {})
    ks = (m3.get("historical_distribution_values_ks") or {})
    if not (isinstance(ks, dict) and bool(ks.get("ok"))):
        return {}

    margin = ks.get("margin") if isinstance(ks.get("margin"), dict) else {}
    d_disc = ks.get("d_discreteness") if isinstance(ks.get("d_discreteness"), dict) else {}
    dig = ks.get("digitize_scale_thresholds") if isinstance(ks.get("digitize_scale_thresholds"), dict) else {}
    eff = ks.get("effective_n_scenarios") if isinstance(ks.get("effective_n_scenarios"), dict) else {}

    p_neff_curves = None
    try:
        p_neff_curves = (eff.get("wielgus2022_pre_eht_curves_n") or {}).get("p_asymptotic")
    except Exception:
        p_neff_curves = None

    return {
        "ok": True,
        "source_json": str(path.relative_to(root)).replace("\\", "/"),
        "alpha": ks.get("alpha"),
        "d": ks.get("d"),
        "p_asymptotic": ks.get("p_asymptotic"),
        "p_minus_alpha": margin.get("p_minus_alpha"),
        "d_step": d_disc.get("d_step"),
        "p_next_up_asymptotic": d_disc.get("p_next_up_asymptotic"),
        "digitize_scale_up_threshold": dig.get("scale_up_to_decrease_hist_le_count_by_1"),
        "digitize_delta_abs_to_flip_one_step": dig.get("delta_abs_to_move_hist_le_max_to_t"),
        "digitize_delta_rel_to_flip_one_step": dig.get("delta_rel_to_move_hist_le_max_to_t"),
        "p_if_effective_n_curves_30": p_neff_curves,
    }


def _extract_eht_candidates(fals: Dict[str, Any], *, paper5_m3_rescue: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    eht = fals.get("eht") if isinstance(fals.get("eht"), dict) else {}
    rows = eht.get("rows") if isinstance(eht.get("rows"), list) else []
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or r.get("key") or "EHT")

        kappa_systematics_source = str(r.get("kappa_systematics_source") or "")
        kappa_budget_sigma = float(r.get("kappa_budget_sigma", float("nan")))
        kappa_obs_sigma = float(r.get("kappa_obs_sigma", float("nan")))
        kappa_sigma_assumed_kerr = float(r.get("kappa_sigma_assumed_kerr", float("nan")))
        kappa_sigma_systematics_now = float("nan")
        if kappa_systematics_source == "budget" and math.isfinite(kappa_budget_sigma):
            kappa_sigma_systematics_now = kappa_budget_sigma
        elif kappa_systematics_source == "obs" and math.isfinite(kappa_obs_sigma):
            kappa_sigma_systematics_now = kappa_obs_sigma
        elif kappa_systematics_source == "kerr" and math.isfinite(kappa_sigma_assumed_kerr):
            kappa_sigma_systematics_now = kappa_sigma_assumed_kerr

        kappa_req0 = float(r.get("kappa_sigma_required_3sigma_if_ring_sigma_zero", float("nan")))
        kappa_sigma_systematics_improvement_factor_to_3sigma_if_ring_sigma_zero = float("nan")
        if (
            math.isfinite(kappa_sigma_systematics_now)
            and kappa_sigma_systematics_now > 0
            and math.isfinite(kappa_req0)
            and kappa_req0 > 0
        ):
            kappa_sigma_systematics_improvement_factor_to_3sigma_if_ring_sigma_zero = (
                kappa_sigma_systematics_now / kappa_req0
            )

        out.append(
            {
                "key": f"eht_{name}".replace(" ", "_"),
                "topic": "EHT（ブラックホール影）",
                "target": name,
                "observable": "リング直径→影直径係数（θ_shadow/(GM/c^2D)）",
                "diff_percent": float(eht.get("shadow_diameter_coeff_diff_percent", float("nan"))),
                "diff_uas": float(r.get("diff_uas", float("nan"))),
                "sigma_now_uas": float(r.get("sigma_obs_now_uas", float("nan"))),
                "sigma_now_with_kappa_uas": float(r.get("sigma_obs_now_with_kappa_uas", float("nan"))),
                "sigma_now_with_kappa_scattering_uas": float(
                    r.get("sigma_obs_now_with_kappa_scattering_uas", float("nan"))
                ),
                "sigma_needed_3sigma_uas": float(r.get("sigma_obs_needed_3sigma_uas", float("nan"))),
                "kappa_sigma_required_3sigma_if_ring_sigma_zero": kappa_req0,
                "kappa_sigma_required_3sigma_if_ring_sigma_current": float(
                    r.get("kappa_sigma_required_3sigma_if_ring_sigma_current", float("nan"))
                ),
                "kappa_systematics_source": kappa_systematics_source,
                "kappa_budget_sigma": kappa_budget_sigma,
                "kappa_obs_sigma": kappa_obs_sigma,
                "kappa_sigma_assumed_kerr": kappa_sigma_assumed_kerr,
                "kappa_sigma_systematics_now": kappa_sigma_systematics_now,
                "kappa_sigma_systematics_improvement_factor_to_3sigma_if_ring_sigma_zero": kappa_sigma_systematics_improvement_factor_to_3sigma_if_ring_sigma_zero,
                "gap_factor_now_over_needed": float(r.get("gap_factor_now_over_needed", float("nan"))),
                "gap_factor_now_over_needed_with_kappa": float(r.get("gap_factor_now_over_needed_with_kappa", float("nan"))),
                "gap_factor_now_over_needed_with_kappa_scattering": float(
                    r.get("gap_factor_now_over_needed_with_kappa_scattering", float("nan"))
                ),
                "theta_unit_rel_sigma_now_pct": float(r.get("theta_unit_rel_sigma_now_pct", float("nan"))),
                "theta_unit_rel_sigma_needed_pct": float(r.get("theta_unit_rel_sigma_needed_pct", float("nan"))),
                "status": "implemented",
                "dominant_systematics": ["κ（リング/シャドウ比）", "Kerrスピン/傾斜", "散乱/放射モデル"],
                "source_keys": str(r.get("source_keys") or ""),
                "paper5_m3_near_passing_rescue": paper5_m3_rescue
                if (name.strip() == "Sgr A*" and isinstance(paper5_m3_rescue, dict) and bool(paper5_m3_rescue.get("ok")))
                else None,
            }
        )
    return out


def _extract_delta_candidate(fals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    delta = fals.get("delta") if isinstance(fals.get("delta"), dict) else {}
    rows = delta.get("rows") if isinstance(delta.get("rows"), list) else []

    delta_adopted = float(delta.get("delta_adopted", float("nan")))
    gamma_needed = float(delta.get("gamma_needed_to_probe_delta_adopted", float("nan")))

    gamma_obs_max = float("nan")
    tightest_delta_upper = float("nan")
    tightest_label = ""
    for r in rows:
        if not isinstance(r, dict):
            continue
        g = float(r.get("gamma_obs", float("nan")))
        if math.isfinite(g):
            gamma_obs_max = max(gamma_obs_max, g) if math.isfinite(gamma_obs_max) else g

        du = float(r.get("delta_upper_from_gamma", float("nan")))
        if math.isfinite(du) and du > 0:
            if (not math.isfinite(tightest_delta_upper)) or (du < tightest_delta_upper):
                tightest_delta_upper = du
                tightest_label = str(r.get("label") or r.get("key") or "")

    gap_gamma = float("nan")
    if math.isfinite(gamma_needed) and gamma_needed > 0 and math.isfinite(gamma_obs_max) and gamma_obs_max > 0:
        gap_gamma = gamma_needed / gamma_obs_max

    return {
        "key": "delta_saturation",
        "topic": "速度飽和 δ（特殊相対論の差分）",
        "target": "高γ観測（宇宙線/ν等）",
        "observable": "γ_max ≈ 1/√δ（v→c で dτ/dt が0にならない）",
        "delta_adopted": delta_adopted,
        "gamma_needed_to_probe_delta_adopted": gamma_needed,
        "gamma_obs_max_in_sources": gamma_obs_max,
        "gap_gamma_needed_over_obs": gap_gamma,
        "tightest_delta_upper_from_sources": tightest_delta_upper,
        "tightest_delta_upper_label": tightest_label,
        "status": "implemented_constraint_only",
        "dominant_systematics": ["粒子質量の仮定（特にν）", "エネルギー推定の系統", "サンプル選定（概算）"],
    }


def _extract_s2_candidate(root: Path) -> Optional[Dict[str, Any]]:
    path = root / "output" / "private" / "eht" / "gravity_s2_pmodel_projection.json"
    if not path.exists():
        return None

    try:
        j = _read_json(path)
    except Exception:
        return None

    if not isinstance(j, dict) or not bool(j.get("ok")):
        return None

    red = (j.get("derived") or {}).get("redshift_f") or {}
    pre = (j.get("derived") or {}).get("precession_f_sp") or {}

    return {
        "key": "s2_gravity",
        "topic": "S2（GRAVITY）強場テスト",
        "target": "銀河中心 S2 星軌道",
        "observable": "f（重力赤方偏移; 2018）/ f_SP（Schwarzschild歳差; 2020）",
        "delta_f_pmodel_minus_gr": float(red.get("delta_f", float("nan"))),
        "sigma_f_needed_3sigma": float(red.get("sigma_f_required_3sigma", float("nan"))),
        "sigma_f_now": float(((red.get("obs") or {}).get("sigma_total_quadrature")) or float("nan")),
        "gap_sigma_f_now_over_needed": float(red.get("gap_sigma_now_over_required", float("nan"))),
        "delta_f_sp_order_estimate": float(pre.get("delta_f_sp_order_estimate_2pn_over_1pn", float("nan"))),
        "sigma_f_sp_needed_3sigma_order": float(pre.get("sigma_f_sp_required_3sigma_order", float("nan"))),
        "sigma_f_sp_now": float(((pre.get("obs") or {}).get("sigma")) or float("nan")),
        "gap_sigma_f_sp_now_over_needed_order": float(pre.get("gap_sigma_now_over_required_order", float("nan"))),
        "status": "implemented_constraint_only",
        "dominant_systematics": ["2PN差は微小（係数差）", "視線速度・軌道モデル", "データ系統（校正/参照系）"],
        "source_keys": "GRAVITY2018 (f), GRAVITY2020 (f_SP)",
    }


def _render_table(candidates: List[Dict[str, Any]], *, out_png: Path) -> None:
    _set_japanese_font()

    fig = plt.figure(figsize=(13.5, 6.2), dpi=170)
    ax = fig.add_subplot(111)
    ax.axis("off")

    title = "差分予測候補（Phase 8.1）：必要精度と現状ギャップ（概要）"
    ax.text(0.5, 1.02, title, ha="center", va="bottom", fontsize=14, fontweight="bold", transform=ax.transAxes)

    headers = [
        "候補",
        "対象",
        "観測量（要約）",
        "必要精度（3σ判別）",
        "現状精度（代表）",
        "ギャップ",
        "状態",
    ]

    rows: List[List[str]] = []
    for c in candidates:
        topic = str(c.get("topic") or "")
        target = str(c.get("target") or "")
        obs = str(c.get("observable") or "")
        status = str(c.get("status") or "")

        need = ""
        now = ""
        gap = ""

        if topic.startswith("EHT"):
            sigma_need = c.get("sigma_needed_3sigma_uas")
            if math.isfinite(float(sigma_need or float("nan"))):
                need = f"σ_obs(shadow) ≤ {_format_num(sigma_need, digits=3)} μas"
            else:
                # For cases like M87*, mass/distance uncertainty (θ_unit) dominates → discrimination is n/a.
                now_pct = c.get("theta_unit_rel_sigma_now_pct")
                need_pct = c.get("theta_unit_rel_sigma_needed_pct")
                if math.isfinite(float(now_pct or float("nan"))) and math.isfinite(float(need_pct or float("nan"))):
                    need = (
                        f"n/a（θ_unit={_format_num(now_pct, digits=2)}% > 要求={_format_num(need_pct, digits=2)}%）"
                    )
                else:
                    need = "n/a"

            kappa_req0 = c.get("kappa_sigma_required_3sigma_if_ring_sigma_zero")
            if math.isfinite(float(kappa_req0 or float("nan"))) and float(kappa_req0) > 0:
                need += f" / κσ ≤ {_format_num(float(kappa_req0) * 100, digits=2)}%（ringσ→0）"
            now_sigma = c.get("sigma_now_uas")
            now_sigma_k = c.get("sigma_now_with_kappa_uas")
            now_sigma_ks = c.get("sigma_now_with_kappa_scattering_uas")
            if math.isfinite(float(now_sigma or float("nan"))):
                now = f"σ_obs ≈ {_format_num(now_sigma, digits=3)} μas"
            if math.isfinite(float(now_sigma_k or float("nan"))):
                now += f"（参考:+κ={_format_num(now_sigma_k, digits=3)}）"
            if math.isfinite(float(now_sigma_ks or float("nan"))):
                now += f"（参考:+κ+散乱={_format_num(now_sigma_ks, digits=3)}）"
            kappa_now = c.get("kappa_sigma_systematics_now")
            if math.isfinite(float(kappa_now or float('nan'))):
                now += f" / κσ≈{_format_num(float(kappa_now) * 100, digits=2)}%"
            g = c.get("gap_factor_now_over_needed_with_kappa")
            if not math.isfinite(float(g or float("nan"))):
                g = c.get("gap_factor_now_over_needed")
            g2 = c.get("gap_factor_now_over_needed_with_kappa_scattering")
            if math.isfinite(float(g or float("nan"))):
                gap = f"{_format_num(g, digits=3)}×"
            if not gap and math.isfinite(float(g2 or float("nan"))):
                gap = f"{_format_num(g2, digits=3)}×"
        elif topic.startswith("S2"):
            s_need = c.get("sigma_f_needed_3sigma")
            s_now = c.get("sigma_f_now")
            g = c.get("gap_sigma_f_now_over_needed")
            if math.isfinite(float(s_need or float("nan"))):
                need = f"σ(f) ≤ {_format_num(s_need, digits=2)}"
            if math.isfinite(float(s_now or float("nan"))):
                now = f"σ(f)_now ≈ {_format_num(s_now, digits=2)}"
            if math.isfinite(float(g or float("nan"))):
                gap = f"{_format_num(g, digits=2)}×"
        elif topic.startswith("速度飽和"):
            need = f"γ ≳ {_format_num(c.get('gamma_needed_to_probe_delta_adopted'), digits=2)}"
            now = f"γ_max(既知) ≈ {_format_num(c.get('gamma_obs_max_in_sources'), digits=2)}"
            g = c.get("gap_gamma_needed_over_obs")
            if math.isfinite(float(g or float("nan"))):
                gap = f"{_format_num(g, digits=2)}×"

        rows.append([topic, target, obs, need, now, gap, status])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)

    # Column widths (hand-tuned for readability)
    col_widths = [0.17, 0.10, 0.34, 0.16, 0.18, 0.07, 0.10]
    for i, w in enumerate(col_widths):
        for (r, c), cell in table.get_celld().items():
            if c == i:
                cell.set_width(w)
            # header row
            if r == 0:
                cell.set_facecolor("#f1f1f1")
                cell.set_text_props(weight="bold")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _build_payload(root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fals = _load_decisive_falsification(root)

    candidates: List[Dict[str, Any]] = []
    paper5_m3_rescue = _load_eht_paper5_m3_rescue_metrics(root)
    candidates.extend(_extract_eht_candidates(fals, paper5_m3_rescue=paper5_m3_rescue))

    delta = _extract_delta_candidate(fals)
    if isinstance(delta, dict):
        candidates.append(delta)

    s2 = _extract_s2_candidate(root)
    if isinstance(s2, dict):
        candidates.append(s2)

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "inputs": {
            "decisive_falsification_json": "output/private/summary/decisive_falsification.json",
            "gravity_s2_pmodel_projection_json": "output/eht/gravity_s2_pmodel_projection.json",
            "eht_sgra_paper5_m3_nir_reconnection_conditions_metrics_json": "output/eht/eht_sgra_paper5_m3_nir_reconnection_conditions_metrics.json",
        },
        "policy": {
            "selection_goal": "GRとP-modelの差が出る観測量を候補として列挙し、一次ソース・支配誤差・必要精度でスクリーニングする。",
            "status_scale": {
                "implemented": "スクリプト＋固定名出力があり、オフライン再現可能",
                "implemented_constraint_only": "制約の可視化はできているが“測定”としては未達（概算/仮定あり）",
            },
        },
        "candidates": candidates,
        "notes": [
            "この一覧は Phase 8.1 の棚卸し（第一版）。EHTは差分予測の中心、δは“飽和”の反証条件（概算）として扱う。",
            "EHTのギャップは『参考:+κ』を優先して表示する（κはリング→シャドウ変換の系統誤差）。",
            "S2（GRAVITY）の f / f_SP は現状は整合性チェック（1PN）であり、P-model と GR の 2PN 差を検出するには桁違いの精度が必要（order estimate）。",
            "Sgr A* の放射モデル側（Paper V; M3/2.2 μm）の near-passing 救済条件は eht_Sgr_A* の paper5_m3_near_passing_rescue に添付する（p−α、D-step、digitize scale閾値など）。",
        ],
    }
    return payload, candidates


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Phase 8.1: list decisive differential-prediction candidates and gaps.")
    ap.add_argument("--out-png", type=str, default=str(root / "output" / "private" / "summary" / "decisive_candidates.png"))
    ap.add_argument("--out-json", type=str, default=str(root / "output" / "private" / "summary" / "decisive_candidates.json"))
    args = ap.parse_args()

    out_png = Path(args.out_png)
    out_json = Path(args.out_json)

    payload, candidates = _build_payload(root)
    payload["outputs"] = {
        "decisive_candidates_png": str(out_png).replace("\\", "/"),
        "decisive_candidates_json": str(out_json).replace("\\", "/"),
    }
    _write_json(out_json, payload)
    _render_table(candidates, out_png=out_png)

    try:
        worklog.append_event(
            {
                "event_type": "decisive_candidates",
                "argv": list(sys.argv),
                "inputs": {
                    "decisive_falsification_json": root / "output" / "private" / "summary" / "decisive_falsification.json"
                },
                "outputs": {"decisive_candidates_png": out_png, "decisive_candidates_json": out_json},
            }
        )
    except Exception:
        pass

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
