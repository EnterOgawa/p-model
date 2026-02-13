from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


_DAY_S = 86_400.0
_T_SUN_S = 4.925490947e-6  # GM_sun / c^3 [s] (pulsar timing standard constant)


def _pbdot_quadrupole_peters_mathews(*, Pb_days: float, e: float, m1_msun: float, m2_msun: float) -> float:
    """
    Peters–Mathews (1963) leading-order GR quadrupole orbital period derivative.

    Uses the pulsar timing constant T_sun = GM_sun/c^3 to avoid dependence on G and M_sun separately.
    Returns Pdot_b in [s/s].
    """

    if not (math.isfinite(Pb_days) and Pb_days > 0):
        return float("nan")
    if not (math.isfinite(e) and 0.0 <= e < 1.0):
        return float("nan")
    if not (math.isfinite(m1_msun) and m1_msun > 0 and math.isfinite(m2_msun) and m2_msun > 0):
        return float("nan")

    Pb_s = Pb_days * _DAY_S
    m_tot = m1_msun + m2_msun
    if not (math.isfinite(m_tot) and m_tot > 0):
        return float("nan")

    f_e = (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4) / (1.0 - e**2) ** (7.0 / 2.0)
    base = (_T_SUN_S ** (5.0 / 3.0)) * ((2.0 * math.pi / Pb_s) ** (5.0 / 3.0)) * (m1_msun * m2_msun) / (m_tot ** (1.0 / 3.0))
    return -(192.0 * math.pi / 5.0) * base * f_e


def _sigma_equiv_from_ci95(half_width: float) -> float:
    # 95% two-sided CI half width ~ 1.96 sigma for a Normal distribution
    return float(half_width) / 1.959963984540054


def _extract_uncertainties(sysrec: Dict[str, Any]) -> Tuple[float, float, str]:
    unc = sysrec.get("metric", {}).get("uncertainty", {})
    if not isinstance(unc, dict):
        return float("nan"), float("nan"), ""
    kind = str(unc.get("kind") or "").strip()
    if kind == "sigma":
        sigma_1 = float(unc.get("sigma", float("nan")))
        sigma_95 = 1.959963984540054 * sigma_1 if math.isfinite(sigma_1) and sigma_1 > 0 else float("nan")
        return sigma_1, sigma_95, str(unc.get("confidence") or "1σ")
    if kind == "ci95_half_width":
        hw = float(unc.get("half_width", float("nan")))
        if not math.isfinite(hw) or hw <= 0:
            return float("nan"), float("nan"), str(unc.get("confidence") or "95%")
        return _sigma_equiv_from_ci95(hw), hw, str(unc.get("confidence") or "95%")
    return float("nan"), float("nan"), str(unc.get("confidence") or "")


def _compute_metrics(systems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in systems:
        metric = s.get("metric", {})
        r_paper = float(metric.get("value", float("nan")))

        # Prefer "derived from inputs" when available (Phase 9 Step 9.2).
        r = float("nan")
        pbdot_intr = float("nan")
        pbdot_intr_sigma_1 = float("nan")
        pbdot_pred = float("nan")
        pb_days = float("nan")
        ecc = float("nan")
        m1_msun = float("nan")
        m2_msun = float("nan")
        pred_paper = float("nan")

        inputs = s.get("inputs") or {}
        if isinstance(inputs, dict):
            oe = inputs.get("orbital_elements") or {}
            masses = inputs.get("masses_msun") or {}
            terms = inputs.get("pbdot_terms") or {}
            if isinstance(oe, dict) and isinstance(masses, dict) and isinstance(terms, dict):
                pb_days = float(oe.get("Pb_days", float("nan")))
                ecc = float(oe.get("e", float("nan")))
                m1_msun = float(masses.get("m1", float("nan")))
                m2_msun = float(masses.get("m2", float("nan")))

                intr = terms.get("intrinsic") or {}
                if isinstance(intr, dict):
                    pbdot_intr = float(intr.get("value", float("nan")))
                    pbdot_intr_sigma_1 = float(intr.get("sigma_1", float("nan")))

                pred_ref = terms.get("gr_quadrupole_paper") or terms.get("gw_quadrupole_paper") or {}
                if isinstance(pred_ref, dict):
                    pred_paper = float(pred_ref.get("value", float("nan")))

                pbdot_pred = _pbdot_quadrupole_peters_mathews(
                    Pb_days=pb_days,
                    e=ecc,
                    m1_msun=m1_msun,
                    m2_msun=m2_msun,
                )
                if math.isfinite(pbdot_intr) and math.isfinite(pbdot_pred) and pbdot_pred != 0.0:
                    r = pbdot_intr / pbdot_pred

        # Uncertainty (prefer paper-stated R uncertainty; fallback to intrinsic Pdot uncertainty).
        sigma_1, sigma_95, conf = _extract_uncertainties(s)
        if (not math.isfinite(sigma_1) or sigma_1 <= 0) and math.isfinite(pbdot_intr_sigma_1) and math.isfinite(pbdot_pred) and pbdot_pred != 0.0:
            sigma_1 = abs(pbdot_intr_sigma_1 / pbdot_pred)
            sigma_95 = 1.959963984540054 * sigma_1
            conf = "1σ（Pdot_b,int の誤差から伝播）"

        if not math.isfinite(r):
            r = r_paper
        if not math.isfinite(r):
            continue

        delta = r - 1.0
        frac_abs = abs(delta)
        frac_3sigma = float("nan")
        frac_95 = float("nan")
        if math.isfinite(sigma_1) and sigma_1 > 0:
            frac_3sigma = frac_abs + 3.0 * sigma_1
            frac_95 = frac_abs + 1.959963984540054 * sigma_1

        pred_rel_err = float("nan")
        if math.isfinite(pred_paper) and pred_paper != 0.0 and math.isfinite(pbdot_pred):
            pred_rel_err = (pbdot_pred - pred_paper) / pred_paper

        out.append(
            {
                "id": str(s.get("id") or ""),
                "name": str(s.get("name") or ""),
                "R": r,
                "R_paper": r_paper if math.isfinite(r_paper) else None,
                "sigma_1": sigma_1,
                "sigma_95": sigma_95,
                "sigma_note": conf,
                "delta": delta,
                "abs_delta": frac_abs,
                "non_gr_fraction_upper_3sigma": frac_3sigma,
                "non_gr_fraction_upper_95": frac_95,
                "derived": {
                    "Pb_days": pb_days if math.isfinite(pb_days) else None,
                    "e": ecc if math.isfinite(ecc) else None,
                    "m1_msun": m1_msun if math.isfinite(m1_msun) else None,
                    "m2_msun": m2_msun if math.isfinite(m2_msun) else None,
                    "pbdot_intrinsic": pbdot_intr if math.isfinite(pbdot_intr) else None,
                    "pbdot_intrinsic_sigma_1": pbdot_intr_sigma_1 if math.isfinite(pbdot_intr_sigma_1) else None,
                    "pbdot_pred_quadrupole": pbdot_pred if math.isfinite(pbdot_pred) else None,
                    "pbdot_pred_paper": pred_paper if math.isfinite(pred_paper) else None,
                    "pbdot_pred_rel_error": pred_rel_err if math.isfinite(pred_rel_err) else None,
                },
            }
        )
    return out


def _render_plot(metrics: List[Dict[str, Any]], out_png: Path) -> None:
    _set_japanese_font()

    labels = [str(m.get("name") or "") for m in metrics]
    x = list(range(len(labels)))
    y = [float(m.get("R", float("nan"))) for m in metrics]
    yerr = [float(m.get("sigma_1", float("nan"))) for m in metrics]

    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=180)
    ax.axhline(1.0, color="#666", lw=1.2, ls="--", zorder=0)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        ms=6,
        capsize=5,
        elinewidth=1.4,
        color="#1f77b4",
        label="観測/P-model（四重極, R=1が一致）",
    )

    for xi, m in zip(x, metrics):
        r = float(m.get("R", float("nan")))
        if not math.isfinite(r):
            continue
        sig = float(m.get("sigma_1", float("nan")))
        sig_txt = ""
        if math.isfinite(sig) and sig > 0:
            sig_txt = f"±{sig:.2g}"
        ax.text(
            xi,
            r + 0.0025,
            f"{r:.6f}{sig_txt}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222",
        )

    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.set_ylabel("一致度 R = Pdot_b(obs) / Pdot_b(P-model quad)")
    ax.set_title("二重パルサー：軌道減衰（放射）とP-model（弱場四重極）予測の一致度")
    ax.grid(True, alpha=0.25)

    # Tight y-range for readability (auto but keep around 1)
    finite = [v for v in y if math.isfinite(v)]
    if finite:
        lo = min(finite)
        hi = max(finite)
        pad = max(0.002, 0.25 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_public_plot(metrics: List[Dict[str, Any]], out_png: Path) -> None:
    _set_japanese_font()

    labels = [str(m.get("name") or "") for m in metrics]
    x = list(range(len(labels)))

    delta_pct = []
    sigma_pct = []
    for m in metrics:
        r = float(m.get("R", float("nan")))
        sig = float(m.get("sigma_1", float("nan")))
        if not math.isfinite(r):
            delta_pct.append(float("nan"))
        else:
            delta_pct.append((r - 1.0) * 100.0)
        sigma_pct.append(sig * 100.0 if math.isfinite(sig) and sig > 0 else float("nan"))

    fig, ax = plt.subplots(figsize=(12.8, 6.0), dpi=180)
    ax.axhline(0.0, color="#666", lw=1.2, ls="--", zorder=0)

    # Bar with 1σ error (when available)
    yerr = [s if math.isfinite(s) else 0.0 for s in sigma_pct]
    ax.bar(
        x,
        [d if math.isfinite(d) else 0.0 for d in delta_pct],
        yerr=yerr,
        capsize=6,
        color="#1f77b4",
        alpha=0.9,
        label="0%が完全一致（エラーバーは1σ相当）",
    )

    for xi, d, s in zip(x, delta_pct, sigma_pct):
        if not math.isfinite(d):
            continue
        if math.isfinite(s) and s > 0:
            txt = f"{d:+.3f}% ±{s:.2g}%"
        else:
            txt = f"{d:+.3f}%"
        ax.text(xi, d + (0.02 if d >= 0 else -0.02), txt, ha="center", va="bottom" if d >= 0 else "top", fontsize=10)

    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.set_ylabel("ずれ（観測/予測 - 1）[%]")
    ax.set_title("二重パルサー：軌道減衰の観測はP-model（弱場四重極）とどれくらい一致？")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    finite = [v for v in delta_pct if math.isfinite(v)]
    if finite:
        lo = min(finite)
        hi = max(finite)
        pad = max(0.02, 0.35 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = _repo_root()

    default_in = root / "data" / "pulsar" / "binary_pulsar_orbital_decay.json"
    default_png = root / "output" / "private" / "pulsar" / "binary_pulsar_orbital_decay.png"
    default_png_public = root / "output" / "private" / "pulsar" / "binary_pulsar_orbital_decay_public.png"
    default_metrics = root / "output" / "private" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json"

    ap = argparse.ArgumentParser(description="Binary pulsar orbital decay check (observed vs P-model quadrupole limit).")
    ap.add_argument("--in-json", type=str, default=str(default_in))
    ap.add_argument("--out-png", type=str, default=str(default_png))
    ap.add_argument("--out-png-public", type=str, default=str(default_png_public))
    ap.add_argument("--out-json", type=str, default=str(default_metrics))
    args = ap.parse_args()

    in_json = Path(args.in_json)
    out_png = Path(args.out_png)
    out_png_public = Path(args.out_png_public)
    out_json = Path(args.out_json)

    payload = _read_json(in_json)
    systems = payload.get("systems", [])
    if not isinstance(systems, list):
        print(f"[err] invalid systems in {in_json}")
        return 2

    metrics = _compute_metrics([s for s in systems if isinstance(s, dict)])
    if not metrics:
        print("[err] no valid system records")
        return 2

    _render_plot(metrics, out_png=out_png)
    _render_public_plot(metrics, out_png=out_png_public)

    out_payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "inputs": {"binary_pulsar_orbital_decay_json": str(in_json).replace("\\", "/")},
        "outputs": {
            "binary_pulsar_orbital_decay_png": str(out_png).replace("\\", "/"),
            "binary_pulsar_orbital_decay_public_png": str(out_png_public).replace("\\", "/"),
            "binary_pulsar_orbital_decay_metrics_json": str(out_json).replace("\\", "/"),
        },
        "metrics": metrics,
        "notes": [
            "R=1が（弱場・遠方の）四重極放射の予測と一致。",
            "Phase 9 Step 9.2：一次ソースのPb,e,m,Pdot_b（補正後）から Pdot_b(Peters-Mathews) を再計算し、R を導出した。",
            "『非GRの追加減衰（双極放射など）』は、|R−1| と誤差から上限（概算）として見積もる。",
        ],
    }
    _write_json(out_json, out_payload)

    try:
        worklog.append_event(
            {
                "event_type": "pulsar_orbital_decay",
                "argv": list(sys.argv),
                "inputs": {"binary_pulsar_orbital_decay_json": in_json},
                "outputs": {"png": out_png, "public_png": out_png_public, "metrics_json": out_json},
                "summary": {"systems": [m.get("id") for m in metrics]},
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] pub : {out_png_public}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
