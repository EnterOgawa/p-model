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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(x: Any) -> Optional[float]:
    try:
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        v = float(x)
    except Exception:
        return None

    # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。

    if not math.isfinite(v):
        return None

    return v


def _category(row: Dict[str, Any]) -> Tuple[str, str]:
    uses_bao = bool(row.get("uses_bao", False))
    label = str(row.get("short_label") or row.get("id") or "")
    rid = str(row.get("id") or "")

    # 条件分岐: `uses_bao` を満たす経路を評価する。
    if uses_bao:
        return ("SNIa+BAO", "#1f77b4")

    # 条件分岐: `"H(z)" in label or "snIa_hz" in rid` を満たす経路を評価する。

    if "H(z)" in label or "snIa_hz" in rid:
        return ("SNIa+H(z)", "#9467bd")

    # 条件分岐: `label.startswith("Clusters") or "Clusters" in label or "clusters" in rid` を満たす経路を評価する。

    if label.startswith("Clusters") or "Clusters" in label or "clusters" in rid:
        return ("Clusters+SNIa", "#2ca02c")

    # 条件分岐: `"SGL" in label or "sgl" in rid` を満たす経路を評価する。

    if "SGL" in label or "sgl" in rid:
        return ("SGL+SNIa+GRB", "#ff7f0e")

    # 条件分岐: `"radio" in label or "radio" in rid` を満たす経路を評価する。

    if "radio" in label or "radio" in rid:
        return ("SNe+radio", "#8c564b")

    return ("other", "#7f7f7f")


def _pairwise_abs_z_matrix(eps: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    n = int(eps.shape[0])
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            # 条件分岐: `i == j` を満たす経路を評価する。
            if i == j:
                m[i, j] = 0.0
                continue

            denom = math.sqrt(float(sigma[i]) ** 2 + float(sigma[j]) ** 2)
            # 条件分岐: `denom <= 0` を満たす経路を評価する。
            if denom <= 0:
                m[i, j] = float("nan")
                continue

            m[i, j] = abs(float(eps[i]) - float(eps[j])) / denom

    return m


def _epsilon0_from_eta0(eta0: float, z_ref: float, *, nonlinear: bool) -> Optional[float]:
    # 条件分岐: `z_ref <= 0` を満たす経路を評価する。
    if z_ref <= 0:
        return None

    # 条件分岐: `nonlinear` を満たす経路を評価する。

    if nonlinear:
        eta = 1.0 + eta0 * z_ref / (1.0 + z_ref)
    else:
        eta = 1.0 + eta0 * z_ref

    # 条件分岐: `eta <= 0` を満たす経路を評価する。

    if eta <= 0:
        return None

    return math.log(eta) / math.log(1.0 + z_ref)


def _epsilon0_sigma_from_eta0(
    eta0: float, eta0_sigma: float, z_ref: float, *, nonlinear: bool
) -> Optional[float]:
    lo = _epsilon0_from_eta0(eta0 - eta0_sigma, z_ref, nonlinear=nonlinear)
    hi = _epsilon0_from_eta0(eta0 + eta0_sigma, z_ref, nonlinear=nonlinear)
    # 条件分岐: `lo is None or hi is None` を満たす経路を評価する。
    if lo is None or hi is None:
        return None

    return abs(hi - lo) / 2.0


def _anchor_sensitivity(
    constraints: List[Dict[str, Any]],
    *,
    z_refs: List[float],
    epsilon0_pred_pbg_static: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in constraints:
        rp = c.get("raw_parameterization")
        # 条件分岐: `not isinstance(rp, dict)` を満たす経路を評価する。
        if not isinstance(rp, dict):
            continue

        eta0 = _safe_float(rp.get("eta0"))
        eta0_sigma = _safe_float(rp.get("eta0_sigma"))
        form = str(rp.get("form") or "")
        # 条件分岐: `eta0 is None or eta0_sigma is None` を満たす経路を評価する。
        if eta0 is None or eta0_sigma is None:
            continue

        nonlinear = "z/(1+z)" in form.replace(" ", "")

        eps_list: List[Optional[float]] = []
        sig_list: List[Optional[float]] = []
        absz_list: List[Optional[float]] = []
        for z_ref in z_refs:
            eps = _epsilon0_from_eta0(float(eta0), float(z_ref), nonlinear=nonlinear)
            sig = _epsilon0_sigma_from_eta0(float(eta0), float(eta0_sigma), float(z_ref), nonlinear=nonlinear)
            # 条件分岐: `eps is None or sig is None or sig <= 0` を満たす経路を評価する。
            if eps is None or sig is None or sig <= 0:
                eps_list.append(None)
                sig_list.append(None)
                absz_list.append(None)
                continue

            absz = abs((float(eps) - float(epsilon0_pred_pbg_static)) / float(sig))
            eps_list.append(float(eps))
            sig_list.append(float(sig))
            absz_list.append(float(absz))

        out.append(
            {
                "id": str(c.get("id") or ""),
                "short_label": str(c.get("short_label") or c.get("id") or ""),
                "form": form,
                "eta0": float(eta0),
                "eta0_sigma": float(eta0_sigma),
                "nonlinear": bool(nonlinear),
                "z_refs": list(z_refs),
                "epsilon0_at_z_ref": eps_list,
                "epsilon0_sigma_at_z_ref": sig_list,
                "abs_z_pbg_static_at_z_ref": absz_list,
            }
        )

    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cosmology (Step 16.5.2): quantify DDR assumption dependence via (i) pairwise internal consistency and (ii) z_ref anchor sensitivity."
    )
    ap.add_argument(
        "--out-dir",
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Output directory (default: output/private/cosmology)",
    )
    ap.add_argument(
        "--cap",
        type=float,
        default=5.0,
        help="Cap for |z| in the internal-consistency heatmap (default: 5.0).",
    )
    ap.add_argument(
        "--cap-anchor",
        type=float,
        default=20.0,
        help="Cap for |z| in the anchor-sensitivity plot (default: 20.0).",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_japanese_font()

    metrics_path = _ROOT / "output" / "private" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json"
    # 条件分岐: `not metrics_path.exists()` を満たす経路を評価する。
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing required metrics: {metrics_path} (run scripts/summary/run_all.py --offline first)"
        )

    ddr = _read_json(metrics_path)
    rows_in = ddr.get("rows") or []
    # 条件分岐: `not isinstance(rows_in, list) or not rows_in` を満たす経路を評価する。
    if not isinstance(rows_in, list) or not rows_in:
        raise ValueError("invalid DDR metrics (rows missing): cosmology_distance_duality_constraints_metrics.json")

    rows: List[Dict[str, Any]] = []
    for r in rows_in:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        eps = _safe_float(r.get("epsilon0_obs"))
        sig = _safe_float(r.get("epsilon0_sigma"))
        # 条件分岐: `eps is None or sig is None or sig <= 0` を満たす経路を評価する。
        if eps is None or sig is None or sig <= 0:
            continue

        cat, cat_color = _category(r)
        rows.append(
            {
                "id": str(r.get("id") or ""),
                "short_label": str(r.get("short_label") or r.get("id") or ""),
                "uses_bao": bool(r.get("uses_bao", False)),
                "category": cat,
                "category_color": cat_color,
                "epsilon0_obs": float(eps),
                "epsilon0_sigma": float(sig),
                "source": dict(r.get("source") or {}),
            }
        )

    # 条件分岐: `not rows` を満たす経路を評価する。

    if not rows:
        raise ValueError("no usable DDR rows found (epsilon0_obs/epsilon0_sigma missing)")

    eps = np.array([float(r["epsilon0_obs"]) for r in rows], dtype=float)
    sig = np.array([float(r["epsilon0_sigma"]) for r in rows], dtype=float)
    labels = [str(r["short_label"]) for r in rows]
    ids = [str(r["id"]) for r in rows]

    m_absz = _pairwise_abs_z_matrix(eps, sig)
    cap = float(args.cap)
    m_plot = np.minimum(m_absz, cap)

    # Summary (top pairs)
    pairs: List[Dict[str, Any]] = []
    n = int(len(rows))
    for i in range(n):
        for j in range(i + 1, n):
            zij = float(m_absz[i, j])
            # 条件分岐: `not math.isfinite(zij)` を満たす経路を評価する。
            if not math.isfinite(zij):
                continue

            pairs.append(
                {
                    "abs_z": zij,
                    "i": i,
                    "j": j,
                    "id_i": ids[i],
                    "id_j": ids[j],
                    "label_i": labels[i],
                    "label_j": labels[j],
                }
            )

    pairs.sort(key=lambda x: float(x["abs_z"]), reverse=True)
    top_pairs = pairs[: min(10, len(pairs))]

    per_source = []
    for i in range(n):
        row_vals = [float(m_absz[i, j]) for j in range(n) if j != i and math.isfinite(float(m_absz[i, j]))]
        per_source.append(
            {
                "id": ids[i],
                "short_label": labels[i],
                "max_abs_z_vs_others": (max(row_vals) if row_vals else None),
                "median_abs_z_vs_others": (float(np.median(row_vals)) if row_vals else None),
            }
        )

    per_source.sort(key=lambda x: float(x["max_abs_z_vs_others"] or -1.0), reverse=True)

    # Anchor sensitivity (for entries that originate from η0 parameterizations)
    data_path = _ROOT / "data" / "cosmology" / "distance_duality_constraints.json"
    data = _read_json(data_path) if data_path.exists() else {}
    constraints_in = data.get("constraints") if isinstance(data, dict) else None
    constraints: List[Dict[str, Any]] = [c for c in constraints_in if isinstance(c, dict)] if constraints_in else []
    z_refs = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    epsilon0_pred_pbg_static = -1.0
    anchor = _anchor_sensitivity(
        constraints,
        z_refs=z_refs,
        epsilon0_pred_pbg_static=epsilon0_pred_pbg_static,
    )

    # Figures
    import matplotlib.pyplot as plt

    # (A) Internal consistency heatmap
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    im = ax.imshow(m_plot, cmap="viridis", vmin=0.0, vmax=cap)
    ax.set_title("宇宙論（DDR）：ε0 制約の内部整合性（ペア差の |z|）", fontsize=14)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="both", which="both", length=0)

    # grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("|z|  (independence approx.)")

    ax_info.axis("off")
    lines: List[str] = []
    lines.append("定義（独立近似）:")
    lines.append("  z_ij = (ε_i−ε_j) / sqrt(σ_i^2+σ_j^2)")
    lines.append("  |z| が大きいほど相互に矛盾")
    lines.append("")
    # 条件分岐: `top_pairs` を満たす経路を評価する。
    if top_pairs:
        mx = top_pairs[0]
        lines.append("最大ペア:")
        lines.append(f"  |z|={mx['abs_z']:.3f}")
        lines.append(f"  {mx['label_i']}  vs  {mx['label_j']}")
        lines.append("")
        lines.append("上位ペア（|z|）:")
        for k, p in enumerate(top_pairs[:8], start=1):
            lines.append(f"  {k:>2}. {p['abs_z']:.3f} : {p['label_i']}  vs  {p['label_j']}")
    else:
        lines.append("(ペア比較が作れませんでした)")

    ax_info.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11)

    fig.tight_layout()
    png_path = out_dir / "cosmology_distance_duality_internal_consistency.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # (B) z_ref anchor sensitivity (η0 -> ε0 conversion)
    fig2 = plt.figure(figsize=(16, 9))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
    ax_lin = fig2.add_subplot(gs2[0, 0])
    ax_non = fig2.add_subplot(gs2[0, 1])
    ax_lin.set_title("DDR：z_ref 依存（η0→ε0 変換）: η(z)=1+η0 z", fontsize=12)
    ax_non.set_title("DDR：z_ref 依存（η0→ε0 変換）: η(z)=1+η0 z/(1+z)", fontsize=12)
    ax_lin.set_xlabel("z_ref（換算アンカー）")
    ax_non.set_xlabel("z_ref（換算アンカー）")
    ax_lin.set_ylabel("|z|（静的最小 ε0=-1 の棄却度）")
    ax_non.set_ylabel("|z|（静的最小 ε0=-1 の棄却度）")

    cap_anchor = float(args.cap_anchor)
    for item in anchor:
        zref = item.get("z_refs") or []
        absz = item.get("abs_z_pbg_static_at_z_ref") or []
        # 条件分岐: `not isinstance(zref, list) or not isinstance(absz, list) or len(zref) != len(...` を満たす経路を評価する。
        if not isinstance(zref, list) or not isinstance(absz, list) or len(zref) != len(absz):
            continue

        ys = [float(v) if v is not None and math.isfinite(float(v)) else float("nan") for v in absz]
        xs = [float(v) for v in zref]
        target = ax_non if bool(item.get("nonlinear", False)) else ax_lin
        label = str(item.get("short_label") or item.get("id") or "")
        target.plot(xs, np.minimum(ys, cap_anchor), marker="o", linewidth=2.0, label=label)

    for axx in (ax_lin, ax_non):
        axx.axvline(1.0, color="#888888", linestyle="--", linewidth=1.0)
        axx.set_ylim(0.0, cap_anchor)
        axx.grid(True, alpha=0.3)
        axx.legend(loc="upper right", fontsize=9)

    fig2.suptitle("宇宙論（DDR）：η0パラメータ化の z_ref 感度（Step 16.5.2）", fontsize=14)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    anchor_png = out_dir / "cosmology_distance_duality_anchor_sensitivity.png"
    fig2.savefig(anchor_png, dpi=150)
    plt.close(fig2)

    out_json = out_dir / "cosmology_distance_duality_internal_consistency_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "ddr_metrics": str(metrics_path).replace("\\", "/"),
            "ddr_constraints_data": (str(data_path).replace("\\", "/") if data_path.exists() else None),
        },
        "definition": {
            "pairwise_abs_z": "abs_z_ij = |(ε_i−ε_j)/sqrt(σ_i^2+σ_j^2)| (independence approx.)",
            "anchor_conversion": "ε0(z_ref)=ln(η(z_ref))/ln(1+z_ref) for η(z)=1+η0 f(z). σ is approximated by (ε_high−ε_low)/2 using η0±σ.",
        },
        "assumptions": {
            "pairwise": "各制約の誤差は独立・正規近似（相関は無視）。",
            "anchor": "η0パラメータ化の換算は便宜上の整理であり、一次ソースの全z範囲フィットを置き換えるものではない。",
        },
        "rows": rows,
        "matrix_abs_z": m_absz.tolist(),
        "summary": {
            "cap_abs_z_for_plot": cap,
            "max_pair": (top_pairs[0] if top_pairs else None),
            "top_pairs": top_pairs,
            "per_source": per_source,
        },
        "anchor_sensitivity": {
            "z_refs": z_refs,
            "epsilon0_pred_pbg_static": epsilon0_pred_pbg_static,
            "rows": anchor,
            "cap_abs_z_for_plot": cap_anchor,
        },
        "outputs": {
            "png": str(png_path).replace("\\", "/"),
            "anchor_png": str(anchor_png).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
        },
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {png_path}")
    print(f"[ok] anchor_png : {anchor_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_distance_duality_internal_consistency",
                "argv": list(sys.argv),
                "inputs": {"ddr_metrics": metrics_path, "ddr_constraints_data": data_path},
                "outputs": {"png": png_path, "anchor_png": anchor_png, "metrics_json": out_json},
                "metrics": {"n_rows": len(rows), "cap": cap, "cap_anchor": cap_anchor},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
