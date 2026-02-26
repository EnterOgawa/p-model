from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

def _fmt_float(x: float, *, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(x: Any) -> Optional[float]:
    try:
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        return float(x)
    except Exception:
        return None


# 関数: `_arxiv_id_from_url` の入出力契約と処理意図を定義する。

def _arxiv_id_from_url(url: str) -> str:
    s = (url or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return ""

    for key in ("arxiv.org/abs/", "arxiv.org/pdf/"):
        i = s.find(key)
        # 条件分岐: `i < 0` を満たす経路を評価する。
        if i < 0:
            continue

        tail = s[i + len(key) :]
        tail = tail.split("?")[0].split("#")[0]
        # 条件分岐: `key.endswith("/pdf/")` を満たす経路を評価する。
        if key.endswith("/pdf/"):
            tail = tail.split(".pdf")[0]

        return tail.strip("/").strip()

    return ""


# 関数: `_source_short` の入出力契約と処理意図を定義する。

def _source_short(source: Any) -> str:
    # 条件分岐: `not isinstance(source, dict)` を満たす経路を評価する。
    if not isinstance(source, dict):
        return ""

    arxiv = str(source.get("arxiv_id") or "").strip()
    # 条件分岐: `not arxiv` を満たす経路を評価する。
    if not arxiv:
        url = str(source.get("url") or "").strip()
        arxiv = _arxiv_id_from_url(url)

    # 条件分岐: `arxiv` を満たす経路を評価する。

    if arxiv:
        return f"arXiv:{arxiv}"

    doi = str(source.get("doi") or "").strip()
    # 条件分岐: `doi` を満たす経路を評価する。
    if doi:
        return f"doi:{doi}"

    year = str(source.get("year") or "").strip()
    return year


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


# 関数: `_select_ddr_representatives` の入出力契約と処理意図を定義する。

def _select_ddr_representatives(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    best_bao: Optional[Dict[str, Any]] = None
    best_bao_sig = float("inf")
    best_no_bao: Optional[Dict[str, Any]] = None
    best_no_bao_abs_z = float("inf")

    for r in rows:
        sig = _safe_float(r.get("epsilon0_sigma"))
        # 条件分岐: `sig is None or not (sig > 0.0)` を満たす経路を評価する。
        if sig is None or not (sig > 0.0):
            continue

        uses_bao = bool(r.get("uses_bao", False))
        # 条件分岐: `uses_bao and sig < best_bao_sig` を満たす経路を評価する。
        if uses_bao and sig < best_bao_sig:
            best_bao_sig = sig
            best_bao = dict(r)

        # 条件分岐: `not uses_bao` を満たす経路を評価する。

        if not uses_bao:
            z_pbg = _safe_float(r.get("z_pbg_static"))
            # 条件分岐: `z_pbg is None` を満たす経路を評価する。
            if z_pbg is None:
                continue

            az = abs(float(z_pbg))
            # 条件分岐: `az < best_no_bao_abs_z` を満たす経路を評価する。
            if az < best_no_bao_abs_z:
                best_no_bao_abs_z = az
                best_no_bao = dict(r)

    return {"bao": best_bao, "no_bao": best_no_bao}


# クラス: `PackRow` の責務と境界条件を定義する。

@dataclass(frozen=True)
class PackRow:
    label: str
    kind: str
    depends_on_distance_indicators: str
    z_static: float
    note: str
    sources: str


# 関数: `_plot_pack` の入出力契約と処理意図を定義する。

def _plot_pack(rows: Sequence[PackRow], *, out_png: Path, cap_sigma: float) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    labels = [r.label for r in rows]
    abs_z = [abs(r.z_static) if math.isfinite(r.z_static) else float("nan") for r in rows]
    statuses = [_classify_sigma(a) for a in abs_z]
    colors = [c for _, c in statuses]

    plotted = [min(float(a), float(cap_sigma)) if math.isfinite(a) else 0.0 for a in abs_z]
    y = np.arange(len(rows))

    fig = plt.figure(figsize=(18, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0))
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax.barh(y, plotted, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("|z|（静的無限空間：最小仮定の棄却度）", fontsize=11)
    ax.set_xlim(0.0, float(cap_sigma))
    ax.grid(True, axis="x", linestyle="--", alpha=0.45)
    for xline, txt in [(1.0, "1σ"), (3.0, "3σ"), (5.0, "5σ")]:
        ax.axvline(xline, color="#333333", linewidth=1.0, alpha=0.25)
        ax.text(xline + 0.15, -0.6, txt, fontsize=9, color="#333333", alpha=0.8)

    for i, a in enumerate(abs_z):
        # 条件分岐: `not math.isfinite(a)` を満たす経路を評価する。
        if not math.isfinite(a):
            continue

        shown = min(a, cap_sigma)
        text = f"{_fmt_float(a, digits=3)}σ"
        # 条件分岐: `a > cap_sigma` を満たす経路を評価する。
        if a > cap_sigma:
            text = f">{_fmt_float(cap_sigma, digits=3)}σ（{_fmt_float(a, digits=3)}σ）"

        ax.text(
            shown + 0.2,
            float(i),
            text,
            va="center",
            ha="left",
            fontsize=9,
            color="#111111",
        )

    # Right side: lightweight “assumption / dependency” table

    ax2.axis("off")
    ax2.set_title("前提と依存（例外規定の整理）", fontsize=12)
    header = ["検証", "距離指標依存", "注意（系統/前提）", "一次ソース"]
    table_rows = []
    for r in rows:
        table_rows.append([r.label, r.depends_on_distance_indicators, r.note, r.sources])

    table = ax2.table(
        cellText=table_rows,
        colLabels=header,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.24, 0.19, 0.39, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.35)
    for (row_i, col_i), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        # 条件分岐: `row_i == 0` を満たす経路を評価する。
        if row_i == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold", color="#222222")
        else:
            cell.set_facecolor("#ffffff")

    fig.suptitle("静的無限空間仮説（最小仮定）の検証パック：主要プローブの棄却度と前提", fontsize=14)
    fig.text(
        0.5,
        0.012,
        "色: |z|<3（OK） / 3〜5（要改善） / >5（不一致）。DDRは距離指標（SNIa/BAOなど）の前提に依存し、Tolmanは進化系統が支配的になり得る。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: static-infinite-space hypothesis verification pack.")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Output directory (default: output/private/cosmology).",
    )
    ap.add_argument(
        "--cap-sigma",
        type=float,
        default=20.0,
        help="Max sigma shown in the bar plot (default: 20). Larger values are clipped and annotated.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir)
    cap_sigma = float(args.cap_sigma)
    # 条件分岐: `not (cap_sigma > 0.0)` を満たす経路を評価する。
    if not (cap_sigma > 0.0):
        raise ValueError("--cap-sigma must be > 0")

    # Inputs (generated by other cosmology scripts in run_all).

    p_sn = out_dir / "cosmology_sn_time_dilation_constraints_metrics.json"
    p_cmb = out_dir / "cosmology_cmb_temperature_scaling_constraints_metrics.json"
    p_ap = out_dir / "cosmology_alcock_paczynski_constraints_metrics.json"
    p_ddr = out_dir / "cosmology_distance_duality_constraints_metrics.json"
    p_tol = out_dir / "cosmology_tolman_surface_brightness_constraints_metrics.json"
    p_reach = out_dir / "cosmology_distance_indicator_reach_limit_metrics.json"
    p_err_budget_sens = out_dir / "cosmology_distance_indicator_error_budget_sensitivity_metrics.json"
    p_bao = out_dir / "cosmology_bao_scaled_distance_fit_metrics.json"

    sn = _read_json(p_sn)
    sn_row = (sn.get("rows") or [None])[0] or {}
    cmb = _read_json(p_cmb)
    cmb_row = (cmb.get("rows") or [None])[0] or {}
    apm = _read_json(p_ap)
    ddr = _read_json(p_ddr)
    tol = _read_json(p_tol)
    reach = _read_json(p_reach)
    err_budget_sens = _read_json(p_err_budget_sens) if p_err_budget_sens.exists() else {}
    bao = _read_json(p_bao) if p_bao.exists() else {}

    ap_src = _source_short((((apm.get("rows") or [None])[0] or {}) if isinstance(apm, dict) else {}).get("source"))

    ddr_reps = _select_ddr_representatives(ddr.get("rows") or [])
    reach_reps = (reach.get("reach") or {}) if isinstance(reach.get("reach"), dict) else {}

    err_env_opt_total = ((err_budget_sens.get("envelope") or {}).get("opt_total") or {}) if isinstance(err_budget_sens, dict) else {}
    err_scan_best = ((err_budget_sens.get("scan") or {}).get("best") or {}) if isinstance(err_budget_sens, dict) else {}

    # 関数: `ddr_note` の入出力契約と処理意図を定義する。
    def ddr_note(kind: str) -> str:
        k = kind.lower()
        rep = reach_reps.get(k)
        # 条件分岐: `not isinstance(rep, dict)` を満たす経路を評価する。
        if not isinstance(rep, dict):
            return "距離指標の前提に依存（補正量は解釈により非一意）"

        z1 = rep.get("z1_band_abs_delta_mu_mag") or {}
        dm1 = _safe_float(z1.get("central"))
        zlim02 = None
        for item in rep.get("reach_z_limit_by_budget_mag") or []:
            # 条件分岐: `abs(float(item.get("budget_mag", 0.0)) - 0.2) < 1e-9` を満たす経路を評価する。
            if abs(float(item.get("budget_mag", 0.0)) - 0.2) < 1e-9:
                zlim02 = _safe_float(item.get("z_limit"))
                break

        parts = []
        # 条件分岐: `dm1 is not None` を満たす経路を評価する。
        if dm1 is not None:
            parts.append(f"Δμ(z=1)≈{_fmt_float(abs(dm1), digits=3)} mag")

        # 条件分岐: `zlim02 is not None` を満たす経路を評価する。

        if zlim02 is not None:
            parts.append(f"0.2 mag予算→z≈{_fmt_float(zlim02, digits=3)}")

        try:
            zlim_opt_total_3s = _safe_float(((err_env_opt_total.get(k) or {}).get("3.0sigma")) if isinstance(err_env_opt_total, dict) else None)
        except Exception:
            zlim_opt_total_3s = None

        # 条件分岐: `zlim_opt_total_3s is not None` を満たす経路を評価する。

        if zlim_opt_total_3s is not None:
            extra = f"誤差予算(opt_total 3σ)→z≈{_fmt_float(zlim_opt_total_3s, digits=3)}"
            # 条件分岐: `k == "no_bao" and isinstance(err_scan_best, dict) and err_scan_best` を満たす経路を評価する。
            if k == "no_bao" and isinstance(err_scan_best, dict) and err_scan_best:
                bw = err_scan_best.get("bin_width")
                nmin = err_scan_best.get("sn_min_points")
                try:
                    # 条件分岐: `bw is not None and nmin is not None` を満たす経路を評価する。
                    if bw is not None and nmin is not None:
                        extra += f"（最楽観: bin幅={_fmt_float(float(bw), digits=3)}; n≥{int(nmin)}）"
                except Exception:
                    pass

            parts.append(extra)

        # 条件分岐: `parts` を満たす経路を評価する。

        if parts:
            return " / ".join(parts)

        return "距離指標の前提に依存（補正量は解釈により非一意）"

    pack_rows: List[PackRow] = []

    # DDR representatives (distance-indicator dependent).
    for key, label in [("bao", "DDR（SNIa+BAO 代表）"), ("no_bao", "DDR（クラスター+SNe 代表）")]:
        r = ddr_reps.get(key)
        z_pbg = _safe_float((r or {}).get("z_pbg_static"))
        # 条件分岐: `z_pbg is None` を満たす経路を評価する。
        if z_pbg is None:
            continue

        pack_rows.append(
            PackRow(
                label=label,
                kind="DDR",
                depends_on_distance_indicators="高い（距離指標）",
                z_static=float(z_pbg),
                note=ddr_note(key),
                sources=_source_short((r or {}).get("source")),
            )
        )

    # AP: use max |z| across BOSS DR12 bins for the P_bg_exponential model.

    ap_z: List[float] = []
    for r in apm.get("rows") or []:
        mz = ((r.get("models") or {}).get("P_bg_exponential") or {}).get("z_score")
        z = _safe_float(mz)
        # 条件分岐: `z is not None` を満たす経路を評価する。
        if z is not None:
            ap_z.append(float(z))

    # 条件分岐: `ap_z` を満たす経路を評価する。

    if ap_z:
        ap_abs_max = max(abs(z) for z in ap_z)
        pack_rows.append(
            PackRow(
                label="AP（BOSS DR12; F_AP）",
                kind="AP",
                depends_on_distance_indicators="中（BAO形状; r_dは相殺）",
                z_static=float(ap_abs_max) * (1.0 if (ap_z[0] >= 0) else -1.0),
                note=f"3点の最大|z|≈{_fmt_float(ap_abs_max, digits=3)}",
                sources=ap_src,
            )
        )

    # BAO: best-fit effective ruler evolution vs minimal (s_R=0).

    best_bao = ((bao.get("fit") or {}).get("best_fit") or {}) if isinstance(bao, dict) else {}
    s_r_bao = _safe_float(best_bao.get("s_R"))
    s_r_sigma = _safe_float(best_bao.get("s_R_sigma_1d"))
    # 条件分岐: `s_r_bao is not None and s_r_sigma is not None and s_r_sigma > 0` を満たす経路を評価する。
    if s_r_bao is not None and s_r_sigma is not None and s_r_sigma > 0:
        z_sr0 = float(s_r_bao) / float(s_r_sigma)
        pack_rows.append(
            PackRow(
                label="BAO（BOSS DR12; s_R）",
                kind="BAO",
                depends_on_distance_indicators="中（標準定規; Qは自由）",
                z_static=float(z_sr0),
                note=f"s_R={_fmt_float(float(s_r_bao), digits=3)}±{_fmt_float(float(s_r_sigma), digits=3)} / 最小 s_R=0",
                sources=ap_src,
            )
        )

    # SN time dilation (distance-indicator independent): background-P predicts p_t=1 (same ratio as redshift).

    z_frw = _safe_float(sn_row.get("z_frw"))
    # 条件分岐: `z_frw is not None` を満たす経路を評価する。
    if z_frw is not None:
        pack_rows.append(
            PackRow(
                label="SN time dilation（SNIa, spectra）",
                kind="SN time dilation",
                depends_on_distance_indicators="低い（距離指標と独立）",
                z_static=float(z_frw),
                note=f"観測 p_t={_fmt_float(float(sn_row.get('p_t_obs', float('nan'))), digits=3)}±{_fmt_float(float(sn_row.get('p_t_sigma', float('nan'))), digits=3)} / P_bg予測 p_t=1",
                sources=_source_short(sn_row.get("source")),
            )
        )

    # CMB T(z) scaling (distance-indicator independent): background-P predicts standard scaling p_T=1 (β_T=0).

    z_std = _safe_float(cmb_row.get("z_std"))
    # 条件分岐: `z_std is not None` を満たす経路を評価する。
    if z_std is not None:
        pack_rows.append(
            PackRow(
                label="CMB 温度 T(z)（SZ+distance）",
                kind="T(z)",
                depends_on_distance_indicators="低い（距離指標と独立）",
                z_static=float(z_std),
                note=f"観測 p_T={_fmt_float(float(cmb_row.get('p_T_obs', float('nan'))), digits=3)}±{_fmt_float(float(cmb_row.get('p_T_sigma', float('nan'))), digits=3)} / P_bg予測 p_T=1",
                sources=_source_short(cmb_row.get("source")),
            )
        )

    # Tolman surface brightness (evolution systematics).

    for r in tol.get("rows") or []:
        z_pbg = _safe_float(r.get("z_pbg_static"))
        # 条件分岐: `z_pbg is None` を満たす経路を評価する。
        if z_pbg is None:
            continue

        band = str(r.get("short_label") or r.get("id") or "Tolman")
        pack_rows.append(
            PackRow(
                label=f"Tolman 表面輝度（{band}）",
                kind="Tolman",
                depends_on_distance_indicators="低い（進化系統が支配的）",
                z_static=float(z_pbg),
                note="銀河進化補正の符号/大きさに強く依存",
                sources=_source_short(r.get("source")),
            )
        )

    out_png = out_dir / "cosmology_static_infinite_hypothesis_pack.png"
    out_json = out_dir / "cosmology_static_infinite_hypothesis_pack_metrics.json"

    _plot_pack(pack_rows, out_png=out_png, cap_sigma=cap_sigma)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "sn_time_dilation_metrics": str(p_sn).replace("\\", "/"),
            "cmb_temperature_scaling_metrics": str(p_cmb).replace("\\", "/"),
            "alcock_paczynski_metrics": str(p_ap).replace("\\", "/"),
            "distance_duality_metrics": str(p_ddr).replace("\\", "/"),
            "tolman_metrics": str(p_tol).replace("\\", "/"),
            "reach_limit_metrics": str(p_reach).replace("\\", "/"),
            **(
                {"error_budget_sensitivity_metrics": str(p_err_budget_sens).replace("\\", "/")}
                if p_err_budget_sens.exists()
                else {}
            ),
            **({"bao_scaled_distance_fit_metrics": str(p_bao).replace("\\", "/")} if p_bao.exists() else {}),
        },
        "definition": {
            "z_score": "z=(obs-pred)/sigma（符号つき）。図では |z| を棄却度の目安として表示。",
            "static_minimum_assumptions": {
                "DDR": "ε0=-1（D_L=(1+z)D_A）",
                "Tolman": "n=2（SB ∝ (1+z)^-2）",
                "BAO": "s_R=0（標準定規のredshift依存なし; Qは自由）",
                "SN_time_dilation": "p_t=1（Δt_obs=(1+z)Δt_em; P_bgでν比がそのまま時間比に出る）",
                "CMB_Tz": "p_T=1（T(z)∝(1+z); β_T=0）",
            },
            "thresholds_sigma": {"ok": "<3", "mixed": "3〜5", "ng": ">5"},
        },
        "selection_policy": {
            "ddr_representative_bao": "uses_bao=true の中で最小σ（tightest）",
            "ddr_representative_no_bao": "uses_bao=false の中で |z_pbg_static| 最小（least rejecting）",
        },
        "params": {"cap_sigma": cap_sigma},
        "rows": [r.__dict__ for r in pack_rows],
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
        "notes": [
            "これは“静的無限空間（最小仮定）”が、現行の主要プローブでどの程度棄却されるかを俯瞰するための整理。",
            "DDRは距離指標（SNIa/BAO/クラスター距離）の前提に強く依存し、Tolmanは銀河進化補正（系統）が支配的になり得る。",
            "SN time dilation と T(z) は距離指標と独立な一次ソース制約であり、P_bg最小（p_t=1, p_T=1）は観測と整合する一方、tired-light（p_t=0）や no-scaling（p_T=0）は強く棄却される。",
            "BAO行は、静的幾何モデルの下で BOSS DR12 の BAO距離（D_M,H）を整合させるための有効定規進化 s_R を見たもの（最小仮定: s_R=0）。",
        ],
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_static_infinite_hypothesis_pack",
                "argv": list(sys.argv),
                "inputs": payload["inputs"],
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {"cap_sigma": cap_sigma, "n_rows": len(pack_rows)},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
