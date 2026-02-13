from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


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


def _fmt_float(x: float, *, digits: int = 6) -> str:
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Experiment:
    id: str
    short_label: str
    title: str
    mu: Optional[float]
    mu_sigma: Optional[float]
    omega_pred_mas_per_yr: Optional[float]
    omega_obs_mas_per_yr: Optional[float]
    omega_obs_sigma_mas_per_yr: Optional[float]
    sigma_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "Experiment":
        def _opt_float(v: Any) -> Optional[float]:
            try:
                vv = float(v)
            except Exception:
                return None
            if math.isnan(vv) or math.isinf(vv):
                return None
            return vv

        return Experiment(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            mu=_opt_float(j.get("mu")),
            mu_sigma=_opt_float(j.get("mu_sigma")),
            omega_pred_mas_per_yr=_opt_float(j.get("omega_pred_mas_per_yr")),
            omega_obs_mas_per_yr=_opt_float(j.get("omega_obs_mas_per_yr")),
            omega_obs_sigma_mas_per_yr=_opt_float(j.get("omega_obs_sigma_mas_per_yr")),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


def compute(experiments: Sequence[Experiment]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for e in experiments:
        mu = e.mu
        sig = e.mu_sigma

        # Prefer explicit μ. If absent, compute μ from drift rates (absolute values).
        if mu is None:
            if e.omega_pred_mas_per_yr is not None and e.omega_obs_mas_per_yr is not None:
                denom = abs(float(e.omega_pred_mas_per_yr))
                if denom > 0:
                    mu = abs(float(e.omega_obs_mas_per_yr)) / denom
                    if e.omega_obs_sigma_mas_per_yr is not None:
                        sig = abs(float(e.omega_obs_sigma_mas_per_yr)) / denom

        z = None
        if mu is not None and sig is not None and sig > 0:
            z = (mu - 1.0) / sig

        rows.append(
            {
                "id": e.id,
                "short_label": e.short_label,
                "title": e.title,
                "mu": None if mu is None else float(mu),
                "mu_sigma": None if sig is None else float(sig),
                "epsilon": None if mu is None else float(mu - 1.0),
                "z_score": None if z is None else float(z),
                "omega_pred_mas_per_yr": (
                    None if e.omega_pred_mas_per_yr is None else float(e.omega_pred_mas_per_yr)
                ),
                "omega_obs_mas_per_yr": None if e.omega_obs_mas_per_yr is None else float(e.omega_obs_mas_per_yr),
                "omega_obs_sigma_mas_per_yr": (
                    None if e.omega_obs_sigma_mas_per_yr is None else float(e.omega_obs_sigma_mas_per_yr)
                ),
                "sigma_note": e.sigma_note,
                "source": e.source,
            }
        )
    return rows


def _plot(rows: Sequence[Dict[str, Any]], *, out_png: Path) -> None:
    _set_japanese_font()

    labels = [str(r.get("short_label") or "") for r in rows]
    y = np.array([(float(r["mu"]) if r.get("mu") is not None else float("nan")) for r in rows], dtype=float)
    yerr = np.array([(float(r["mu_sigma"]) if r.get("mu_sigma") is not None else float("nan")) for r in rows], dtype=float)
    x = np.arange(len(rows), dtype=float)

    fig = plt.figure(figsize=(10.8, 5.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(1.0, color="#333333", linewidth=1.2, alpha=0.85, label="予測（GR / P-model, μ=1）")
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.6,
        capsize=4,
        label="観測（一次ソースの公表値）",
    )

    for xi, mu, sig in zip(x, y, yerr):
        if not math.isfinite(float(mu)):
            continue
        txt = f"{mu:.3f}"
        if math.isfinite(float(sig)) and float(sig) > 0:
            txt += f"±{sig:.3f}"
        ax.text(float(xi), float(mu) + 0.03, txt, ha="center", va="bottom", fontsize=9.5, color="#111111")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("比 μ = |Ω_obs| / |Ω_pred|（GR=1）")
    ax.set_title("回転（フレームドラッグ）：観測 vs P-model（一次ソース）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    finite_y = [float(v) for v in y.tolist() if math.isfinite(float(v))]
    finite_err = [float(v) for v in yerr.tolist() if math.isfinite(float(v)) and float(v) > 0]
    if finite_y:
        lo = min(finite_y)
        hi = max(finite_y)
        pad = max(0.15, 0.8 * (hi - lo))
        if finite_err:
            pad = max(pad, 2.0 * max(finite_err))
        ax.set_ylim(min(0.0, lo - pad), max(2.0, hi + pad))

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "short_label",
                "mu",
                "mu_sigma",
                "z_score",
                "omega_pred_mas_per_yr",
                "omega_obs_mas_per_yr",
                "omega_obs_sigma_mas_per_yr",
                "source_url",
                "source_doi",
            ]
        )
        for r in rows:
            src = r.get("source") or {}
            if not isinstance(src, dict):
                src = {}
            mu = r.get("mu")
            sig = r.get("mu_sigma")
            z = r.get("z_score")
            w.writerow(
                [
                    r.get("id", ""),
                    r.get("short_label", ""),
                    "" if mu is None else _fmt_float(float(mu), digits=6),
                    "" if sig is None else _fmt_float(float(sig), digits=6),
                    "" if z is None else _fmt_float(float(z), digits=4),
                    "" if r.get("omega_pred_mas_per_yr") is None else _fmt_float(float(r["omega_pred_mas_per_yr"]), digits=6),
                    "" if r.get("omega_obs_mas_per_yr") is None else _fmt_float(float(r["omega_obs_mas_per_yr"]), digits=6),
                    ""
                    if r.get("omega_obs_sigma_mas_per_yr") is None
                    else _fmt_float(float(r["omega_obs_sigma_mas_per_yr"]), digits=6),
                    str(src.get("url") or ""),
                    str(src.get("doi") or ""),
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _ROOT
    default_data = root / "data" / "theory" / "frame_dragging_experiments.json"
    default_outdir = root / "output" / "private" / "theory"

    ap = argparse.ArgumentParser(description="Frame-dragging experiments (observed ratio μ vs prediction μ=1).")
    ap.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Input JSON (default: data/theory/frame_dragging_experiments.json)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(default_outdir),
        help="Output directory (default: output/private/theory)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    src = _read_json(data_path)
    experiments = [Experiment.from_json(e) for e in (src.get("experiments") or [])]
    if not experiments:
        raise SystemExit(f"no experiments found in: {data_path}")

    rows = compute(experiments)

    png_path = outdir / "frame_dragging_experiments.png"
    _plot(rows, out_png=png_path)

    out_json = outdir / "frame_dragging_experiments.json"
    out_csv = outdir / "frame_dragging_experiments.csv"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "model": "P-model",
        "definition": src.get("definition") or {},
        "rows": rows,
        "outputs": {"plot_png": str(png_path).replace("\\", "/"), "rows_json": str(out_json).replace("\\", "/"), "rows_csv": str(out_csv).replace("\\", "/")},
    }
    _write_json(out_json, payload)
    _write_csv(out_csv, rows)

    try:
        worklog.append_event(
            {
                "event_type": "theory_frame_dragging_experiments",
                "argv": sys.argv,
                "inputs": {"data": data_path},
                "outputs": {"plot_png": png_path, "rows_json": out_json, "rows_csv": out_csv},
                "metrics": {"n_experiments": len(rows)},
            }
        )
    except Exception:
        pass

    print(f"[ok] plot : {png_path}")
    print(f"[ok] json : {out_json}")
    print(f"[ok] csv  : {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
