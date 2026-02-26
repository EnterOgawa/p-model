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


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(x: Any) -> Optional[float]:
    try:
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        return float(x)
    except Exception:
        return None


# 関数: `_classify_sigma` の入出力契約と処理意図を定義する。

def _classify_sigma(abs_z: float) -> Tuple[str, str]:
    # 条件分岐: `not math.isfinite(abs_z)` を満たす経路を評価する。
    if not math.isfinite(abs_z):
        return ("info", "#999999")

    # 条件分岐: `abs_z < 3.0` を満たす経路を評価する。

    if abs_z < 3.0:
        return ("ok", "#2ca02c")

    # 条件分岐: `abs_z < 5.0` を満たす経路を評価する。

    if abs_z < 5.0:
        return ("mixed", "#ffbf00")

    return ("ng", "#d62728")


# 関数: `_load_required` の入出力契約と処理意図を定義する。

def _load_required(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise FileNotFoundError(
            f"missing required metrics: {path} (run scripts/summary/run_all.py --offline first)"
        )

    return _read_json(path)


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


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cosmology tension attribution (Step 16.5): summarize which probes drive mismatch and next checks."
    )
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Output directory (default: output/private/cosmology)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_japanese_font()

    # --- Inputs (precomputed metrics from primary scripts)
    pack = _load_required(_ROOT / "output" / "private" / "cosmology" / "cosmology_static_infinite_hypothesis_pack_metrics.json")
    ddr = _load_required(_ROOT / "output" / "private" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json")
    ddr_sys = _load_required(_ROOT / "output" / "private" / "cosmology" / "cosmology_distance_duality_systematics_envelope_metrics.json")
    bao_fit = _load_required(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_distance_ratio_fit_metrics.json")
    bao_global_prior_survey_f = _load_required(
        _ROOT
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan_metrics.json"
    )
    bao_global_prior_loo_f = _load_required(
        _ROOT
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan_metrics.json"
    )
    bao_sigma_scan = _load_required(
        _ROOT
        / "output"
        / "private"
        / "cosmology"
        / "cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan_metrics.json"
    )
    tolman = _load_required(
        _ROOT / "output" / "private" / "cosmology" / "cosmology_tolman_surface_brightness_constraints_metrics.json"
    )
    sn_td = _load_required(_ROOT / "output" / "private" / "cosmology" / "cosmology_sn_time_dilation_constraints_metrics.json")
    cmb_tz = _load_required(
        _ROOT / "output" / "private" / "cosmology" / "cosmology_cmb_temperature_scaling_constraints_metrics.json"
    )

    # --- Extract key numbers
    pack_rows = pack.get("rows") or []
    # 条件分岐: `not isinstance(pack_rows, list) or not pack_rows` を満たす経路を評価する。
    if not isinstance(pack_rows, list) or not pack_rows:
        raise ValueError("invalid pack metrics (rows missing): cosmology_static_infinite_hypothesis_pack_metrics.json")

    ddr_rows = ddr.get("rows") or []
    # 条件分岐: `not isinstance(ddr_rows, list) or not ddr_rows` を満たす経路を評価する。
    if not isinstance(ddr_rows, list) or not ddr_rows:
        raise ValueError("invalid DDR metrics (rows missing): cosmology_distance_duality_constraints_metrics.json")

    ddr_rep = next((r for r in ddr_rows if isinstance(r, dict) and r.get("id") == "martinelli2021_snIa_bao"), None)
    # 条件分岐: `not isinstance(ddr_rep, dict)` を満たす経路を評価する。
    if not isinstance(ddr_rep, dict):
        ddr_rep = next((r for r in ddr_rows if isinstance(r, dict) and r.get("uses_bao") is True), None)

    # 条件分岐: `not isinstance(ddr_rep, dict)` を満たす経路を評価する。

    if not isinstance(ddr_rep, dict):
        ddr_rep = ddr_rows[0] if isinstance(ddr_rows[0], dict) else {}

    eps_obs = _safe_float(ddr_rep.get("epsilon0_obs"))
    eps_sig = _safe_float(ddr_rep.get("epsilon0_sigma"))
    z_ddr = _safe_float(ddr_rep.get("z_pbg_static"))
    dmu_z1 = _safe_float(ddr_rep.get("delta_distance_modulus_mag_z1"))
    sig_mult_3s = _safe_float(ddr_rep.get("sigma_multiplier_to_not_reject_pbg_static_3sigma"))

    # Also incorporate category-level systematics proxy (σ_cat) if available (Step 16.5.3/16.5.4).
    ddr_sys_rows = ddr_sys.get("rows") if isinstance(ddr_sys.get("rows"), list) else []
    ddr_sys_row = next(
        (r for r in ddr_sys_rows if isinstance(r, dict) and str(r.get("id") or "") == str(ddr_rep.get("id") or "")),
        None,
    )
    ddr_abs_z_raw = _safe_float((ddr_sys_row or {}).get("abs_z_raw"))
    # 条件分岐: `ddr_abs_z_raw is None and z_ddr is not None` を満たす経路を評価する。
    if ddr_abs_z_raw is None and z_ddr is not None:
        ddr_abs_z_raw = abs(float(z_ddr))

    ddr_abs_z_sys = _safe_float((ddr_sys_row or {}).get("abs_z_with_category_sys"))
    ddr_sigma_cat = _safe_float((ddr_sys_row or {}).get("sigma_sys_category"))
    sig_mult_3s_sys = (None if ddr_abs_z_sys is None else float(ddr_abs_z_sys) / 3.0)

    bao_comb = bao_fit.get("results", {}).get("combined", {}) if isinstance(bao_fit, dict) else {}
    bao_boss = bao_fit.get("results", {}).get("boss_only", {}) if isinstance(bao_fit, dict) else {}
    bao_eboss = bao_fit.get("results", {}).get("eboss_only", {}) if isinstance(bao_fit, dict) else {}
    bao_desi = bao_fit.get("results", {}).get("desi_only", {}) if isinstance(bao_fit, dict) else {}
    try:
        sR_boss = _safe_float((bao_boss.get("best_fit") or {}).get("s_R"))
        sR_boss_sig = _safe_float((bao_boss.get("best_fit") or {}).get("s_R_sigma_1d"))
        sR_eboss = _safe_float((bao_eboss.get("best_fit") or {}).get("s_R"))
        sR_eboss_sig = _safe_float((bao_eboss.get("best_fit") or {}).get("s_R_sigma_1d"))
        sR_desi = _safe_float((bao_desi.get("best_fit") or {}).get("s_R"))
        sR_desi_sig = _safe_float((bao_desi.get("best_fit") or {}).get("s_R_sigma_1d"))
        sR_comb = _safe_float((bao_comb.get("best_fit") or {}).get("s_R"))
        sR_comb_sig = _safe_float((bao_comb.get("best_fit") or {}).get("s_R_sigma_1d"))
        bao_chi2_dof = _safe_float(bao_comb.get("chi2_dof"))
    except Exception:
        sR_boss = sR_boss_sig = sR_eboss = sR_eboss_sig = sR_desi = sR_desi_sig = sR_comb = sR_comb_sig = bao_chi2_dof = None

    bao_f_1s = _safe_float((bao_sigma_scan.get("results", {}).get("thresholds") or {}).get("rep_bao_f_for_max_abs_z_le_1sigma"))

    # 関数: `_pick_f` の入出力契約と処理意図を定義する。
    def _pick_f(payload: Dict[str, Any], variant_id: str) -> Optional[float]:
        try:
            rows = (payload.get("scan") or {}).get("by_variant") or []
            for r in rows:
                v = r.get("variant") or {}
                # 条件分岐: `str(v.get("id") or "") == variant_id` を満たす経路を評価する。
                if str(v.get("id") or "") == variant_id:
                    return _safe_float(r.get("estimated_f_1sigma_all_candidates"))
        except Exception:
            return None

        return None

    f_boss_baseline_all = _pick_f(bao_global_prior_survey_f, "boss_dr12_dm_h_baseline")
    f_boss_ratio_all = _pick_f(bao_global_prior_survey_f, "bao_ratio_boss_only")
    f_eboss_ratio_all = _pick_f(bao_global_prior_survey_f, "bao_ratio_eboss_only")
    f_desi_ratio_all = _pick_f(bao_global_prior_survey_f, "bao_ratio_desi_only")
    f_comb_ratio_all = _pick_f(bao_global_prior_survey_f, "bao_ratio_combined")
    f_comb_ratio_survey_sys_all = _pick_f(bao_global_prior_survey_f, "bao_ratio_combined_with_survey_sys")
    f_comb_ratio_loo_sys_all = _pick_f(bao_global_prior_survey_f, "bao_ratio_combined_with_loo_sys")

    f_comb_all = _pick_f(bao_global_prior_loo_f, "bao_ratio_combined_baseline")
    f_comb_drop_lya = None
    f_comb_drop_eboss_lya = None
    f_comb_drop_desi_lya = None
    try:
        loo_rows = (bao_global_prior_loo_f.get("scan") or {}).get("by_variant") or []
        for r in loo_rows:
            v = r.get("variant") or {}
            om = v.get("omitted") or {}
            om_id = str(om.get("id") or "")
            # 条件分岐: `om_id == "eboss_dr16_lya_z233"` を満たす経路を評価する。
            if om_id == "eboss_dr16_lya_z233":
                f_comb_drop_eboss_lya = _safe_float(r.get("estimated_f_1sigma_all_candidates"))

            # 条件分岐: `om_id == "desi_dr1_lya_z2330"` を満たす経路を評価する。

            if om_id == "desi_dr1_lya_z2330":
                f_comb_drop_desi_lya = _safe_float(r.get("estimated_f_1sigma_all_candidates"))

            # 条件分岐: `str(om.get("short_label") or "").strip().startswith("Lyα")` を満たす経路を評価する。

            if str(om.get("short_label") or "").strip().startswith("Lyα"):
                f_comb_drop_lya = _safe_float(r.get("estimated_f_1sigma_all_candidates"))
                # keep scanning to also catch DESI/eBOSS specific ids if present
    except Exception:
        f_comb_drop_lya = None

    tol_rows = tolman.get("rows") or []
    # 条件分岐: `not isinstance(tol_rows, list) or not tol_rows` を満たす経路を評価する。
    if not isinstance(tol_rows, list) or not tol_rows:
        raise ValueError("invalid Tolman metrics (rows missing): cosmology_tolman_surface_brightness_constraints_metrics.json")

    tol_r = next((r for r in tol_rows if isinstance(r, dict) and str(r.get("id") or "").endswith("_r")), None)
    tol_i = next((r for r in tol_rows if isinstance(r, dict) and str(r.get("id") or "").endswith("_i")), None)
    tol_r_z = _safe_float((tol_r or {}).get("z_pbg_static"))
    tol_i_z = _safe_float((tol_i or {}).get("z_pbg_static"))
    tol_r_evol = _safe_float((tol_r or {}).get("evolution_exponent_needed_pbg_static"))
    tol_i_evol = _safe_float((tol_i or {}).get("evolution_exponent_needed_pbg_static"))

    sn_rows = sn_td.get("rows") or []
    cmb_rows = cmb_tz.get("rows") or []
    sn_row = sn_rows[0] if (isinstance(sn_rows, list) and sn_rows and isinstance(sn_rows[0], dict)) else {}
    cmb_row = cmb_rows[0] if (isinstance(cmb_rows, list) and cmb_rows and isinstance(cmb_rows[0], dict)) else {}
    p_t_obs = _safe_float(sn_row.get("p_t_obs"))
    p_t_sig = _safe_float(sn_row.get("p_t_sigma"))
    z_p_t = _safe_float(sn_row.get("z_frw"))
    p_T_obs = _safe_float(cmb_row.get("p_T_obs"))
    p_T_sig = _safe_float(cmb_row.get("p_T_sigma"))
    z_p_T = _safe_float(cmb_row.get("z_std"))

    # --- Figure
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    labels: List[str] = []
    abs_zs: List[float] = []
    colors: List[str] = []
    ann: List[str] = []
    for r in pack_rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        label = str(r.get("label") or "").strip()
        z = _safe_float(r.get("z_static"))
        # 条件分岐: `not label or z is None or not math.isfinite(float(z))` を満たす経路を評価する。
        if not label or z is None or not math.isfinite(float(z)):
            continue

        az = abs(float(z))
        status, color = _classify_sigma(az)
        labels.append(label)
        abs_zs.append(az)
        colors.append(color)
        ann.append(f"{az:.2f}" if az < 20 else f">20 ({az:.2f})")

    # 条件分岐: `not labels` を満たす経路を評価する。

    if not labels:
        raise ValueError("no valid rows in static hypothesis pack")

    y = np.arange(len(labels))
    x = np.minimum(np.array(abs_zs, dtype=float), 20.0)
    ax.barh(y, x, color=colors, alpha=0.90)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 20.0)
    ax.set_xlabel("|z|（静的背景P最小：棄却度）", fontsize=11)
    ax.set_title("宇宙論：張力の原因切り分け（Step 16.5 入口）", fontsize=13)
    ax.axvline(3.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.axvline(5.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.text(3.0, -0.8, "3σ", ha="center", va="bottom", fontsize=9, color="#666666")
    ax.text(5.0, -0.8, "5σ", ha="center", va="bottom", fontsize=9, color="#666666")

    for yi, xi, t in zip(y, x, ann):
        ax.text(min(19.8, float(xi) + 0.2), yi, t, va="center", ha="left", fontsize=9, color="#333333")

    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax_info.axis("off")

    # 関数: `_box` の入出力契約と処理意図を定義する。
    def _box(y_top: float, title: str, lines: List[str], *, fc: str = "#ffffff") -> float:
        txt = title + "\n" + "\n".join(lines)
        ax_info.text(
            0.02,
            y_top,
            txt,
            transform=ax_info.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.45", "fc": fc, "ec": "#cfcfcf"},
        )
        # rough height estimate (keeps layout stable without measuring pixels)
        return y_top - (0.055 * (2 + len(lines)))

    y0 = 0.98
    y0 = _box(
        y0,
        "独立プローブ（距離指標に依存しにくい）",
        [
            f"SN time dilation: p_t={_fmt(p_t_obs)}±{_fmt(p_t_sig)} → p_t=1 と整合（z={_fmt(z_p_t, digits=2)}）",
            f"CMB T(z): p_T={_fmt(p_T_obs)}±{_fmt(p_T_sig)} → p_T=1 と整合（z={_fmt(z_p_T, digits=2)}）",
            "背景P（膨張なし）の“赤方偏移機構”そのものは、この2本だけでは棄却されない。",
        ],
        fc="#f8fbff",
    )
    y0 -= 0.02

    tol_note = ""
    # 条件分岐: `tol_r_evol is not None and tol_r_evol < 0` を満たす経路を評価する。
    if tol_r_evol is not None and tol_r_evol < 0:
        tol_note = "（R band は整合に逆符号の進化補正が必要）"

    y0 = _box(
        y0,
        "張力の中心（距離指標依存：DDR / BAO / Tolman）",
        [
            (
                "DDR（SNIa+BAO代表）: "
                f"ε0={_fmt(eps_obs)}±{_fmt(eps_sig)} に対し、静的最小は ε0=-1（|z|≈{_fmt(ddr_abs_z_raw, digits=2)}"
                + (
                    f"→{_fmt(ddr_abs_z_sys, digits=2)}（σ_cat={_fmt(ddr_sigma_cat, digits=3)}）"
                    if ddr_abs_z_sys is not None
                    else ""
                )
                + "）"
            ),
            (
                f"  z=1での差: Δμ≈{_fmt(dmu_z1)} mag（3σで棄却しないには σ×{_fmt(sig_mult_3s, digits=2)}"
                + (f" / σ_cat込みなら σ×{_fmt(sig_mult_3s_sys, digits=2)}" if sig_mult_3s_sys is not None else "")
                + " が必要）"
            ),
            f"BAO距離比 fit: s_R(併合)≈{_fmt(sR_comb)}±{_fmt(sR_comb_sig)}（χ²/dof≈{_fmt(bao_chi2_dof, digits=2)}）",
            f"  代表（BAO含むDDR）を 1σ に入れる目安: BAOのσを f≈{_fmt(bao_f_1s, digits=2)} 倍（best_independent）",
            (
                "  単一priorで全DDR行を1σ同時整合: "
                + f"f≈{_fmt(f_boss_baseline_all, digits=2)}（BOSS基準） / "
                + f"{_fmt(f_eboss_ratio_all, digits=2)}（eBOSSのみ） / "
                + f"{_fmt(f_comb_ratio_all, digits=2)}（BOSS+eBOSS）"
                + (
                    f" / {_fmt(f_comb_ratio_survey_sys_all, digits=2)}（併合+σ_sys（survey差））"
                    if f_comb_ratio_survey_sys_all is not None
                    else ""
                )
                + (
                    f" / {_fmt(f_comb_ratio_loo_sys_all, digits=2)}（併合+σ_sys（LOO））"
                    if f_comb_ratio_loo_sys_all is not None
                    else ""
                )
            ),
            (
                "  併合fit leave-one-out: "
                + f"全点 f≈{_fmt(f_comb_all, digits=2)} → "
                + f"Lyα(z=2.33)除外で f≈{_fmt(f_comb_drop_lya, digits=2)}"
            ),
            f"Tolman: |z|（静的 n=2）=R:{_fmt(abs(tol_r_z) if tol_r_z is not None else None, digits=2)}, I:{_fmt(abs(tol_i_z) if tol_i_z is not None else None, digits=2)} {tol_note}",
        ],
        fc="#fffaf2",
    )
    y0 -= 0.02

    next_lines = [
        "切り分け方針：",
        "- 距離指標（標準光源/標準定規）の定義・校正・共分散・進化補正のどこが効くかを、一次ソース単位で分解する。",
        "次の検証（候補）：",
        "- DDR：BAOを使わない制約（クラスター+SNe / 強重力レンズ / 標準サイレン等）の増強・相互比較。",
        "- BAO：BOSS/eBOSSの系統差（共分散/解析手法）を固定し、同一枠組みで再評価。",
        "- Tolman：進化補正が支配的なので、波長/母集団選別を変えた一次ソースで符号とスケールを再検証。",
    ]
    _box(y0, "Step 16.5（観測起因 vs モデル起因）への落とし込み", next_lines, fc="#f7fff7")

    out_png = out_dir / "cosmology_tension_attribution.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "z_thresholds": {"ok": 3.0, "mixed": 5.0},
            "notes": "この図は『静的背景P（最小仮定）』と一次ソース制約の張力が、どの観測（前提）に集中しているかを要約する。",
        },
        "inputs": {
            "static_pack_metrics": "output/private/cosmology/cosmology_static_infinite_hypothesis_pack_metrics.json",
            "ddr_metrics": "output/private/cosmology/cosmology_distance_duality_constraints_metrics.json",
            "ddr_systematics_metrics": "output/private/cosmology/cosmology_distance_duality_systematics_envelope_metrics.json",
            "bao_distance_ratio_fit_metrics": "output/private/cosmology/cosmology_bao_distance_ratio_fit_metrics.json",
            "bao_global_prior_survey_f_metrics": "output/private/cosmology/cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan_metrics.json",
            "bao_global_prior_leave_one_out_f_metrics": "output/private/cosmology/cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan_metrics.json",
            "bao_sigma_scan_metrics": "output/private/cosmology/cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan_metrics.json",
            "tolman_metrics": "output/private/cosmology/cosmology_tolman_surface_brightness_constraints_metrics.json",
            "sn_time_dilation_metrics": "output/private/cosmology/cosmology_sn_time_dilation_constraints_metrics.json",
            "cmb_temperature_metrics": "output/private/cosmology/cosmology_cmb_temperature_scaling_constraints_metrics.json",
        },
        "results": {
            "ddr_representative": {
                "id": str(ddr_rep.get("id") or ""),
                "epsilon0_obs": eps_obs,
                "epsilon0_sigma": eps_sig,
                "z_pbg_static": z_ddr,
                "abs_z_raw": ddr_abs_z_raw,
                "abs_z_with_category_sys": ddr_abs_z_sys,
                "sigma_sys_category": ddr_sigma_cat,
                "delta_distance_modulus_mag_z1": dmu_z1,
                "sigma_multiplier_to_not_reject_pbg_static_3sigma": sig_mult_3s,
                "sigma_multiplier_to_not_reject_pbg_static_3sigma_with_category_sys": sig_mult_3s_sys,
            },
            "bao_distance_ratio_fit": {
                "boss_only": {"s_R": sR_boss, "s_R_sigma_1d": sR_boss_sig},
                "eboss_only": {"s_R": sR_eboss, "s_R_sigma_1d": sR_eboss_sig},
                "desi_only": {"s_R": sR_desi, "s_R_sigma_1d": sR_desi_sig},
                "combined": {"s_R": sR_comb, "s_R_sigma_1d": sR_comb_sig, "chi2_dof": bao_chi2_dof},
            },
            "bao_sigma_relax_needed": {
                "rep_bao_f_for_max_abs_z_le_1sigma": bao_f_1s,
            },
            "bao_global_prior_sigma_relax_needed": {
                "single_prior_worst_max_abs_z_le_1sigma": {
                    "boss_dr12_dm_h_baseline": f_boss_baseline_all,
                    "bao_ratio_boss_only": f_boss_ratio_all,
                    "bao_ratio_eboss_only": f_eboss_ratio_all,
                    "bao_ratio_desi_only": f_desi_ratio_all,
                    "bao_ratio_combined": f_comb_ratio_all,
                    "bao_ratio_combined_with_survey_sys": f_comb_ratio_survey_sys_all,
                    "bao_ratio_combined_with_loo_sys": f_comb_ratio_loo_sys_all,
                }
            },
            "bao_global_prior_leave_one_out": {
                "combined_all_points": f_comb_all,
                "combined_drop_lya_z233": f_comb_drop_lya,
                "combined_drop_eboss_lya_z233": f_comb_drop_eboss_lya,
                "combined_drop_desi_lya_z2330": f_comb_drop_desi_lya,
            },
            "tolman": {
                "r_band": {"z_pbg_static": tol_r_z, "evolution_exponent_needed_pbg_static": tol_r_evol},
                "i_band": {"z_pbg_static": tol_i_z, "evolution_exponent_needed_pbg_static": tol_i_evol},
            },
            "independent_probes": {
                "sn_time_dilation": {"p_t_obs": p_t_obs, "p_t_sigma": p_t_sig, "z_std": z_p_t},
                "cmb_temperature": {"p_T_obs": p_T_obs, "p_T_sigma": p_T_sig, "z_std": z_p_T},
            },
            "next_actions": [
                "DDR：BAOを使わない制約（クラスター+SNe / 強重力レンズ / 標準サイレン等）の増強・相互比較。",
                "BAO：BOSS/eBOSSの系統差（共分散/解析手法）を固定し、同一枠組みで再評価。",
                "Tolman：進化補正が支配的なので、波長/母集団選別を変えた一次ソースで符号とスケールを再検証。",
            ],
        },
        "outputs": {"png": str(out_png.relative_to(_ROOT)).replace("\\", "/")},
    }

    out_json = out_dir / "cosmology_tension_attribution_metrics.json"
    _write_json(out_json, metrics)

    try:
        worklog.append_event(
            {
                "kind": "cosmology_tension_attribution",
                "outputs": [out_png, out_json],
                "ddr_z_abs": None if z_ddr is None else abs(float(z_ddr)),
                "ddr_z_abs_with_category_sys": ddr_abs_z_sys,
                "bao_sigma_relax_f": bao_f_1s,
                "bao_global_prior_f_boss_baseline": f_boss_baseline_all,
                "bao_global_prior_f_combined": f_comb_ratio_all,
                "bao_global_prior_f_combined_drop_lya": f_comb_drop_lya,
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
