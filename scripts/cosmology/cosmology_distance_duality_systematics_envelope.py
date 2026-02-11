from __future__ import annotations

import argparse
import json
import math
import statistics
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
    if not math.isfinite(v):
        return None
    return v


def _category(row: Dict[str, Any]) -> Tuple[str, str]:
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


def _robust_sigma(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    # For a normal distribution, sigma ≈ 1.4826 * MAD.
    return 1.4826 * float(mad)


def _sigma_needed_for_nonreject(
    delta_eps: float,
    sigma_obs: float,
    *,
    threshold: float,
) -> float:
    """
    Extra sigma_sys (added in quadrature) required to make |delta_eps|/sigma_total <= threshold.
    """
    if sigma_obs <= 0 or threshold <= 0:
        return float("nan")
    needed_total = abs(float(delta_eps)) / float(threshold)
    v = float(needed_total) ** 2 - float(sigma_obs) ** 2
    return math.sqrt(v) if v > 0 else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cosmology (Step 16.5.3): incorporate DDR model differences as systematic widths (sigma_sys) and re-evaluate rejection significance."
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
    ap.add_argument(
        "--threshold-nonreject",
        type=float,
        default=3.0,
        help="Threshold for 'non-reject' (default: 3.0; i.e., 3σ).",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_japanese_font()

    metrics_path = _ROOT / "output" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing required metrics: {metrics_path} (run scripts/summary/run_all.py --offline first)"
        )
    ddr = _read_json(metrics_path)
    rows_in = ddr.get("rows") or []
    if not isinstance(rows_in, list) or not rows_in:
        raise ValueError("invalid DDR metrics (rows missing): cosmology_distance_duality_constraints_metrics.json")

    epsilon0_pred_pbg_static = -1.0

    rows: List[Dict[str, Any]] = []
    eps_by_cat: Dict[str, List[float]] = {}
    for r in rows_in:
        if not isinstance(r, dict):
            continue
        eps = _safe_float(r.get("epsilon0_obs"))
        sig = _safe_float(r.get("epsilon0_sigma"))
        if eps is None or sig is None or sig <= 0:
            continue
        cat, cat_color = _category(r)
        eps_by_cat.setdefault(cat, []).append(float(eps))
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

    if not rows:
        raise ValueError("no usable DDR rows found (epsilon0_obs/epsilon0_sigma missing)")

    # Estimate category-level systematic widths from within-category spread.
    sigma_sys_by_cat = {cat: _robust_sigma(vals) for cat, vals in eps_by_cat.items()}
    sigma_sys_by_cat = {cat: float(v) for cat, v in sigma_sys_by_cat.items()}

    # Compute raw vs sys-inflated z-scores for the static-min model (epsilon0=-1).
    thr = float(args.threshold_nonreject)
    for r in rows:
        eps = float(r["epsilon0_obs"])
        sig = float(r["epsilon0_sigma"])
        delta = eps - float(epsilon0_pred_pbg_static)
        z_raw = abs(delta) / sig
        sigma_sys_cat = float(sigma_sys_by_cat.get(str(r["category"]), 0.0))
        sigma_total = math.sqrt(sig**2 + sigma_sys_cat**2)
        z_sys = abs(delta) / sigma_total if sigma_total > 0 else float("nan")

        r["epsilon0_pred_pbg_static"] = float(epsilon0_pred_pbg_static)
        r["delta_epsilon0"] = float(delta)
        r["abs_z_raw"] = float(z_raw)
        r["sigma_sys_category"] = float(sigma_sys_cat)
        r["sigma_total"] = float(sigma_total)
        r["abs_z_with_category_sys"] = float(z_sys)
        r["sigma_sys_needed_for_nonreject_threshold"] = _sigma_needed_for_nonreject(delta, sig, threshold=thr)

    rows.sort(key=lambda x: float(x["abs_z_raw"]), reverse=True)

    # Summary
    moved_to_ok = [
        r
        for r in rows
        if float(r["abs_z_raw"]) > thr
        and math.isfinite(float(r["abs_z_with_category_sys"]))
        and float(r["abs_z_with_category_sys"]) <= thr
    ]
    most_reduced = sorted(
        rows,
        key=lambda r: float(r["abs_z_with_category_sys"]) - float(r["abs_z_raw"]),
    )[:8]

    # Figure
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    labels = [str(r["short_label"]) for r in rows]
    z_raw = np.array([float(r["abs_z_raw"]) for r in rows], dtype=float)
    z_sys = np.array([float(r["abs_z_with_category_sys"]) for r in rows], dtype=float)
    colors = [str(r["category_color"]) for r in rows]
    cap = float(args.cap)

    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y + h / 2, np.minimum(z_raw, cap), height=h, color="#c7c7c7", alpha=0.85, label="観測σのみ")
    ax.barh(y - h / 2, np.minimum(z_sys, cap), height=h, color=colors, alpha=0.95, label="カテゴリー系統σ込み")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("|z|（静的最小 ε0=-1 の棄却度）")
    ax.set_title("宇宙論（DDR）：モデル差を系統幅として取り込んだ棄却度（Step 16.5.3）", fontsize=14)
    ax.set_xlim(0.0, cap)
    ax.grid(True, axis="x", alpha=0.25)
    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.axvline(thr, color="#666666", linestyle="--", linewidth=1.2)
    ax.axvline(5.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.legend(loc="lower right", fontsize=10)

    ax_info.axis("off")
    lines: List[str] = []
    lines.append("定義（系統幅の加算）:")
    lines.append("  σ_total^2 = σ_obs^2 + σ_cat^2")
    lines.append("  σ_cat は同一カテゴリ内の ε0_obs の広がり（MAD由来）")
    lines.append("")
    lines.append("カテゴリ別 σ_cat（ε0）:")
    for cat in sorted(sigma_sys_by_cat.keys()):
        n = len(eps_by_cat.get(cat) or [])
        lines.append(f"  - {cat}: n={n}, σ_cat={sigma_sys_by_cat[cat]:.3f}")
    lines.append("")
    lines.append(f"3σ基準で >{thr:g}→≤{thr:g} に移動: {len(moved_to_ok)} 件")
    if moved_to_ok:
        for r in moved_to_ok[:6]:
            lines.append(
                f"  - {r['short_label']}: {r['abs_z_raw']:.2f} → {r['abs_z_with_category_sys']:.2f}"
            )
    lines.append("")
    lines.append("減少が大きい例:")
    for r in most_reduced[:6]:
        d = float(r["abs_z_with_category_sys"]) - float(r["abs_z_raw"])
        lines.append(f"  - {r['short_label']}: Δ|z|={d:.2f} ({r['abs_z_raw']:.2f}→{r['abs_z_with_category_sys']:.2f})")

    ax_info.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11)

    fig.tight_layout()
    png_path = out_dir / "cosmology_distance_duality_systematics_envelope.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    out_json = out_dir / "cosmology_distance_duality_systematics_envelope_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"ddr_metrics": str(metrics_path).replace("\\", "/")},
        "definition": {
            "epsilon0_pred_pbg_static": epsilon0_pred_pbg_static,
            "z_score": "|z| = |ε_obs − ε_model| / σ",
            "sigma_total": "σ_total^2 = σ_obs^2 + σ_cat^2",
            "sigma_cat": "同一カテゴリ内の ε0_obs の広がり（robust: 1.4826*MAD）を系統幅の目安として加算",
        },
        "assumptions": [
            "同一カテゴリ内の差は『モデル差/系統』の代理指標として扱う（クラスター幾何・レンズ質量モデル・η0パラメータ化等）。",
            "カテゴリ間の相関（共通校正など）はここでは扱わない。",
            "SNIa+BAO / SNIa+H(z) も一次ソース追加により複数行が入り、σ_cat を推定できる。ただし n が小さいカテゴリの σ_cat は暫定であり、追加の一次ソースで更新され得る。",
        ],
        "category_systematics": {
            cat: {"sigma_cat": float(sigma_sys_by_cat[cat]), "n": len(eps_by_cat.get(cat) or [])}
            for cat in sorted(sigma_sys_by_cat.keys())
        },
        "rows": rows,
        "summary": {
            "threshold_nonreject": thr,
            "moved_to_ok_count": len(moved_to_ok),
            "moved_to_ok": [
                {
                    "id": r["id"],
                    "short_label": r["short_label"],
                    "abs_z_raw": r["abs_z_raw"],
                    "abs_z_with_category_sys": r["abs_z_with_category_sys"],
                    "category": r["category"],
                }
                for r in moved_to_ok
            ],
            "most_reduced": [
                {
                    "id": r["id"],
                    "short_label": r["short_label"],
                    "abs_z_raw": r["abs_z_raw"],
                    "abs_z_with_category_sys": r["abs_z_with_category_sys"],
                    "delta_abs_z": float(r["abs_z_with_category_sys"]) - float(r["abs_z_raw"]),
                    "category": r["category"],
                }
                for r in most_reduced
            ],
        },
        "outputs": {"png": str(png_path).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {png_path}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_distance_duality_systematics_envelope",
                "argv": list(sys.argv),
                "inputs": {"ddr_metrics": metrics_path},
                "outputs": {"png": png_path, "metrics_json": out_json},
                "metrics": {"n_rows": len(rows), "threshold_nonreject": thr, "cap": cap},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
