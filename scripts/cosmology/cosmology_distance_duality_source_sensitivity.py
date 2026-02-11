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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


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


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _fmt(x: Optional[float], *, digits: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return ""
    x = float(x)
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _category(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Coarse grouping to make "distance-indicator dependence" visible.
    Color uses a stable palette across runs.
    """
    uses_bao = bool(row.get("uses_bao", False))
    label = str(row.get("short_label") or row.get("id") or "")
    rid = str(row.get("id") or "")

    if uses_bao:
        return ("SNIa+BAO", "#1f77b4")
    if "H(z)" in label or "snIa_hz" in rid:
        return ("SNIa+H(z)", "#9467bd")
    if label.startswith("Clusters") or "Clusters" in label or "clusters" in rid:
        return ("Clusters+SNIa", "#2ca02c")
    if "SGL" in label or "sgl" in rid:
        return ("SGL+SNIa+GRB", "#ff7f0e")
    if "radio" in label or "radio" in rid:
        return ("SNe+radio", "#8c564b")
    return ("other", "#7f7f7f")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cosmology (Step 16.5.1): visualize how DDR rejection depends on distance-indicator assumptions."
    )
    ap.add_argument(
        "--out-dir",
        default=str(_ROOT / "output" / "cosmology"),
        help="Output directory (default: output/cosmology)",
    )
    ap.add_argument(
        "--cap",
        type=float,
        default=20.0,
        help="Cap for |z| axis to keep the chart readable (default: 20).",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_japanese_font()

    ddr_metrics_path = _ROOT / "output" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json"
    if not ddr_metrics_path.exists():
        raise FileNotFoundError(
            f"missing required metrics: {ddr_metrics_path} (run scripts/summary/run_all.py --offline first)"
        )
    ddr = _read_json(ddr_metrics_path)
    rows_in = ddr.get("rows") or []
    if not isinstance(rows_in, list) or not rows_in:
        raise ValueError("invalid DDR metrics (rows missing): cosmology_distance_duality_constraints_metrics.json")

    rows: List[Dict[str, Any]] = []
    for r in rows_in:
        if not isinstance(r, dict):
            continue
        z = _safe_float(r.get("z_pbg_static"))
        if z is None:
            continue
        az = abs(float(z))
        cat, color = _category(r)
        rows.append(
            {
                "id": str(r.get("id") or ""),
                "short_label": str(r.get("short_label") or r.get("id") or ""),
                "uses_bao": bool(r.get("uses_bao", False)),
                "category": cat,
                "category_color": color,
                "epsilon0_obs": _safe_float(r.get("epsilon0_obs")),
                "epsilon0_sigma": _safe_float(r.get("epsilon0_sigma")),
                "z_pbg_static": float(z),
                "abs_z_pbg_static": az,
                "delta_eps_needed": _safe_float(r.get("epsilon0_extra_needed_to_match_obs")),
                "delta_mu_mag_z1": _safe_float(r.get("delta_distance_modulus_mag_z1")),
                "tau_equiv_z1": _safe_float(r.get("tau_equivalent_dimming_z1")),
                "sigma_multiplier_nonreject_3sigma": _safe_float(r.get("sigma_multiplier_to_not_reject_pbg_static_3sigma")),
                "source": dict(r.get("source") or {}),
            }
        )

    rows.sort(key=lambda x: float(x["abs_z_pbg_static"]), reverse=True)

    # Highlights
    tightest_bao: Optional[Dict[str, Any]] = None
    least_reject_no_bao: Optional[Dict[str, Any]] = None
    strongest_reject_no_bao: Optional[Dict[str, Any]] = None
    best_bao_sig = float("inf")
    best_no_bao_absz = float("inf")
    worst_no_bao_absz = -1.0
    for r in rows:
        sig = _safe_float(r.get("epsilon0_sigma"))
        if bool(r.get("uses_bao", False)) and sig is not None and sig > 0 and sig < best_bao_sig:
            best_bao_sig = float(sig)
            tightest_bao = r
        if not bool(r.get("uses_bao", False)):
            az = float(r["abs_z_pbg_static"])
            if az < best_no_bao_absz:
                best_no_bao_absz = az
                least_reject_no_bao = r
            if az > worst_no_bao_absz:
                worst_no_bao_absz = az
                strongest_reject_no_bao = r

    # Figure
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    labels = [str(r["short_label"]) for r in rows]
    abs_z = np.array([float(r["abs_z_pbg_static"]) for r in rows], dtype=float)
    cap = float(args.cap)
    x = np.minimum(abs_z, cap)
    colors = [str(r["category_color"]) for r in rows]

    y = np.arange(len(labels))
    ax.barh(y, x, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.0, cap)
    ax.set_xlabel("|z|（静的背景P最小：ε0=-1 の棄却度）", fontsize=11)
    ax.set_title("宇宙論（DDR）：一次ソース依存（距離指標で結論がどれだけ変わるか）", fontsize=13)
    ax.axvline(3.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.axvline(5.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.text(3.0, -0.8, "3σ", ha="center", va="bottom", fontsize=9, color="#666666")
    ax.text(5.0, -0.8, "5σ", ha="center", va="bottom", fontsize=9, color="#666666")
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    for yi, xi, azi in zip(y, x, abs_z):
        t = f"{azi:.2f}" if azi < cap else f">{cap:.0f} ({azi:.2f})"
        ax.text(min(cap - 0.2, float(xi) + 0.2), yi, t, va="center", ha="left", fontsize=9, color="#333333")

    # Legend by category (dedup)
    seen: Dict[str, str] = {}
    for r in rows:
        seen.setdefault(str(r["category"]), str(r["category_color"]))
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=k) for k, c in seen.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax_info.axis("off")

    def _box(y_top: float, title: str, lines: List[str], *, fc: str = "#ffffff") -> float:
        txt = title + "\n" + "\n".join(lines)
        ax_info.text(
            0.02,
            y_top,
            txt,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": fc, "edgecolor": "#dddddd", "boxstyle": "round,pad=0.4"},
        )
        return y_top - (0.055 * (1 + len(lines))) - 0.03

    y0 = 0.98
    if isinstance(tightest_bao, dict):
        y0 = _box(
            y0,
            "代表（BAO含む：最も強い制約）",
            [
                f"{tightest_bao['short_label']}: ε0={_fmt(tightest_bao.get('epsilon0_obs'))}±{_fmt(tightest_bao.get('epsilon0_sigma'))} (1σ)",
                f"静的最小 ε0=-1 は |z|≈{_fmt(tightest_bao.get('abs_z_pbg_static'), digits=2)}",
                f"必要補正（z=1の目安）: Δμ≈{_fmt(tightest_bao.get('delta_mu_mag_z1'), digits=2)} mag, τ≈{_fmt(tightest_bao.get('tau_equiv_z1'), digits=2)}",
                f"非棄却へ必要: σ×{_fmt(tightest_bao.get('sigma_multiplier_nonreject_3sigma'), digits=2)}（3σ目安）",
            ],
            fc="#f4f8ff",
        )

    if isinstance(least_reject_no_bao, dict):
        y0 = _box(
            y0,
            "代表（BAOなし：最も緩い制約）",
            [
                f"{least_reject_no_bao['short_label']}: ε0={_fmt(least_reject_no_bao.get('epsilon0_obs'))}±{_fmt(least_reject_no_bao.get('epsilon0_sigma'))} (1σ)",
                f"静的最小 ε0=-1 は |z|≈{_fmt(least_reject_no_bao.get('abs_z_pbg_static'), digits=2)}",
                f"必要補正（z=1の目安）: Δμ≈{_fmt(least_reject_no_bao.get('delta_mu_mag_z1'), digits=2)} mag, τ≈{_fmt(least_reject_no_bao.get('tau_equiv_z1'), digits=2)}",
            ],
            fc="#f7fff4",
        )

    if isinstance(strongest_reject_no_bao, dict):
        y0 = _box(
            y0,
            "補足（BAOなしでも強く棄却する例）",
            [
                f"{strongest_reject_no_bao['short_label']}: |z|≈{_fmt(strongest_reject_no_bao.get('abs_z_pbg_static'), digits=2)}",
                "→ 『BAOを使うから棄却』に限らず、距離指標（SNe/H(z)/クラスター等）の前提が支配する点を示す。",
            ],
            fc="#fff8f0",
        )

    _box(
        y0,
        "読み方（Step 16.5.1）",
        [
            "同じ静的最小（ε0=-1）でも、一次ソース（距離指標）の採り方で棄却度が大きく動く。",
            "次は『どの仮定（校正/共分散/進化補正/幾何モデル）が |z| を支配するか』を個別に詰める。",
        ],
        fc="#ffffff",
    )

    fig.tight_layout()
    out_png = out_dir / "cosmology_distance_duality_source_sensitivity.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_json = out_dir / "cosmology_distance_duality_source_sensitivity_metrics.json"
    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"ddr_metrics": str(ddr_metrics_path.relative_to(_ROOT)).replace("\\", "/")},
        "definition": {
            "static_min_model": "背景P（膨張なし・静的幾何）の最小モデル: ε0=-1（η=1/(1+z)）",
            "sigma_reference": "ここでの |z| は、一次ソース側の 1σ（ε0_sigma）に対する (ε0=-1) の外れ度。距離指標の前提で大きく動き得る。",
        },
        "highlights": {
            "tightest_bao": None
            if tightest_bao is None
            else {
                k: tightest_bao.get(k)
                for k in [
                    "id",
                    "short_label",
                    "epsilon0_obs",
                    "epsilon0_sigma",
                    "abs_z_pbg_static",
                    "delta_mu_mag_z1",
                    "tau_equiv_z1",
                    "sigma_multiplier_nonreject_3sigma",
                ]
            },
            "least_rejecting_no_bao": None
            if least_reject_no_bao is None
            else {
                k: least_reject_no_bao.get(k)
                for k in [
                    "id",
                    "short_label",
                    "epsilon0_obs",
                    "epsilon0_sigma",
                    "abs_z_pbg_static",
                    "delta_mu_mag_z1",
                    "tau_equiv_z1",
                ]
            },
            "strongest_rejecting_no_bao": None
            if strongest_reject_no_bao is None
            else {
                k: strongest_reject_no_bao.get(k)
                for k in ["id", "short_label", "epsilon0_obs", "epsilon0_sigma", "abs_z_pbg_static"]
            },
        },
        "rows": rows,
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "json": str(out_json.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_json, metrics)

    try:
        worklog.append_event({"kind": "cosmology_ddr_source_sensitivity", "outputs": [out_png, out_json]})
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
