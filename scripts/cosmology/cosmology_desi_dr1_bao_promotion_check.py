from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ZRange:
    lam_min: float
    lam_max: float
    n: int
    z_min: float
    z_max: float

    def sign_flips(self) -> bool:
        return (self.z_min <= 0.0 <= self.z_max) and not (self.z_min == 0.0 == self.z_max)

    def stable_over(self, *, threshold_abs: float) -> bool:
        thr = float(threshold_abs)
        if not (math.isfinite(thr) and thr >= 0.0):
            raise ValueError("threshold_abs must be finite and >= 0")
        if self.sign_flips():
            return False
        return (self.z_max <= -thr) or (self.z_min >= thr)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lam_min": float(self.lam_min),
            "lam_max": float(self.lam_max),
            "n": int(self.n),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
            "sign_flips": bool(self.sign_flips()),
        }


def _iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _load_points(
    path: Path,
    *,
    lam_min: float,
    lam_max: float,
    z_field: str,
) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    pts: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for row in _iter_rows(path):
        tracer = str(row.get("tracer", "")).strip()
        dist = str(row.get("dist", "")).strip()
        lam = _safe_float(row.get("lambda"))
        z = _safe_float(row.get(z_field))
        if not (math.isfinite(lam) and math.isfinite(z)):
            continue
        if lam < float(lam_min) - 1e-12 or lam > float(lam_max) + 1e-12:
            continue
        key = (tracer, dist)
        pts.setdefault(key, []).append((lam, z))
    for items in pts.values():
        items.sort(key=lambda t: t[0])
    return pts


def _summarize_points(
    pts: Dict[Tuple[str, str], List[Tuple[float, float]]],
) -> Dict[Tuple[str, str], ZRange]:
    out: Dict[Tuple[str, str], ZRange] = {}
    for key, items in pts.items():
        lams = [t[0] for t in items]
        zs = [t[1] for t in items]
        out[key] = ZRange(
            lam_min=float(min(lams)),
            lam_max=float(max(lams)),
            n=int(len(lams)),
            z_min=float(min(zs)),
            z_max=float(max(zs)),
        )
    return out


def _ensure_paths(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    missing: List[str] = []
    for p in paths:
        path = Path(p)
        path = path if path.is_absolute() else (_ROOT / path).resolve()
        if not path.exists():
            missing.append(str(path))
            continue
        out.append(path)
    if missing:
        raise SystemExit("missing sweep csv(s):\n" + "\n".join(missing))
    return out


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if chosen:
            mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _short_method_label(method: str) -> str:
    m = str(method or "").lower()
    if "vac_lya" in m or "lya_corr" in m:
        return "VAC (Lyα)"
    if "jk_cov" in m:
        return "jackknife"
    if "rascalc" in m:
        return "RascalC"
    if len(method) > 44:
        return method[:44] + "…"
    return method


def main() -> None:
    ap = argparse.ArgumentParser(description="DESI DR1: promotion gate check from shrinkage sweep CSV(s).")
    ap.add_argument(
        "--csv",
        action="append",
        help="Path to a shrinkage sweep CSV (repeatable). Relative paths are resolved from repo root.",
    )
    ap.add_argument("--lam-min", type=float, default=0.0, help="Minimum lambda to include (default: 0.0)")
    ap.add_argument("--lam-max", type=float, default=1.0, help="Maximum lambda to include (default: 1.0)")
    ap.add_argument(
        "--z-field",
        type=str,
        default="z_score_combined",
        choices=["z_score_combined", "z_score_vs_y1data"],
        help="Which z-score field to summarize (default: z_score_combined)",
    )
    ap.add_argument("--target-dist", type=str, default="pbg", help="Distance label to gate on (default: pbg)")
    ap.add_argument("--threshold-abs", type=float, default=3.0, help="|z| threshold for passing (default: 3)")
    ap.add_argument("--min-tracers", type=int, default=2, help="Minimum passing tracers to promote (default: 2)")
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_desi_dr1_bao_promotion_check.json"),
        help="Output JSON path (default: output/private/cosmology/cosmology_desi_dr1_bao_promotion_check.json)",
    )
    args = ap.parse_args()

    lam_min = float(args.lam_min)
    lam_max = float(args.lam_max)
    if not (math.isfinite(lam_min) and math.isfinite(lam_max) and lam_min <= lam_max):
        raise SystemExit("--lam-min/--lam-max must be finite and lam_min<=lam_max")

    out_path = Path(str(args.out_json))
    out_path = out_path if out_path.is_absolute() else (_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_path.with_suffix("").with_name(out_path.stem + ".png")
    out_public_png = out_path.with_suffix("").with_name(out_path.stem + "_public.png")

    csv_paths: List[Path] = []
    if args.csv:
        csv_paths = _ensure_paths(list(args.csv or []))
    else:
        # Convenience: if --csv is omitted, try to reuse the previous inputs (out_json)
        # or auto-discover sweep CSVs under output/private/cosmology.
        reuse: List[str] = []
        try:
            if out_path.exists():
                prev = json.loads(out_path.read_text(encoding="utf-8"))
                prev_inputs = prev.get("inputs") if isinstance(prev, dict) else None
                prev_csv = prev_inputs.get("csv") if isinstance(prev_inputs, dict) else None
                if isinstance(prev_csv, list):
                    reuse = [str(x) for x in prev_csv if str(x).strip()]
        except Exception:
            reuse = []

        if reuse:
            try:
                csv_paths = _ensure_paths(reuse)
            except SystemExit:
                csv_paths = []

        if not csv_paths:
            sweep_dir = _ROOT / "output" / "private" / "cosmology"
            try:
                csv_paths = sorted(
                    [p for p in sweep_dir.glob("cosmology_desi_dr1_bao_cov_shrinkage_sweep__*.csv") if p.is_file()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except Exception:
                csv_paths = []

    if not csv_paths:
        payload = {
            "root": str(_ROOT),
            "inputs": {"csv": []},
            "outputs": {"json": str(out_path), "png": str(out_png), "public_png": str(out_public_png)},
            "params": {
                "z_field": str(args.z_field),
                "lam_min": float(lam_min),
                "lam_max": float(lam_max),
                "target_dist": str(args.target_dist),
                "threshold_abs": float(args.threshold_abs),
                "min_tracers": int(args.min_tracers),
            },
            "result": {"promoted": None, "passing_tracers": [], "passing_tracers_n": 0},
            "by_method": {},
            "gate_by_tracer": {},
            "notes": [
                "No shrinkage sweep CSV inputs were provided (and no cached inputs were found), so this script emitted a placeholder figure.",
                "To generate the real promotion check, run the upstream sweep generators (e.g. cosmology_desi_dr1_bao_cov_shrinkage_sweep.py) and pass their CSV(s) via --csv (or rerun after the CSVs exist).",
            ],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            import matplotlib.pyplot as plt

            _set_japanese_font()
            fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.8), dpi=160)
            ax.axis("off")
            ax.text(
                0.5,
                0.62,
                "DESI DR1 BAO promotion check (placeholder)",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.40,
                "Missing input: shrinkage sweep CSV(s).\n"
                "Run upstream generators and pass their outputs via --csv.",
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes,
            )
            fig.savefig(out_png, dpi=160)
            fig.savefig(out_public_png, dpi=160)
            plt.close(fig)
        except Exception:
            pass

        print(f"[ok] json: {out_path}")
        print(f"[ok] png : {out_png}")
        print(f"[ok] png : {out_public_png}")
        return

    summary_by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    points_by_method: Dict[str, Dict[Tuple[str, str], List[Tuple[float, float]]]] = {}
    by_tracer_dist_across_methods: Dict[Tuple[str, str], List[ZRange]] = {}

    for p in csv_paths:
        method = p.stem
        pts = _load_points(p, lam_min=lam_min, lam_max=lam_max, z_field=str(args.z_field))
        points_by_method[method] = pts
        zmap = _summarize_points(pts)
        method_out: Dict[str, Dict[str, Any]] = {}
        for (tracer, dist), zr in sorted(zmap.items()):
            method_out.setdefault(tracer, {})[dist] = zr.to_dict()
            by_tracer_dist_across_methods.setdefault((tracer, dist), []).append(zr)
        summary_by_method[method] = method_out

    target_dist = str(args.target_dist).strip()
    thr = float(args.threshold_abs)
    min_tracers = int(args.min_tracers)
    if min_tracers < 1:
        raise SystemExit("--min-tracers must be >= 1")

    passing_tracers: List[str] = []
    tracer_gate: Dict[str, Any] = {}
    for (tracer, dist), zrs in sorted(by_tracer_dist_across_methods.items()):
        if dist != target_dist:
            continue
        stable_all = all(zr.stable_over(threshold_abs=thr) for zr in zrs)
        tracer_gate[tracer] = {
            "dist": dist,
            "threshold_abs": float(thr),
            "stable_all_methods": bool(stable_all),
            "methods_n": int(len(zrs)),
            "ranges": [zr.to_dict() for zr in zrs],
        }
        if stable_all:
            passing_tracers.append(tracer)

    promoted = len(passing_tracers) >= min_tracers

    payload = {
        "root": str(_ROOT),
        "inputs": {"csv": [str(p) for p in csv_paths]},
        "outputs": {
            "json": str(out_path),
            "png": str(out_png),
            "public_png": str(out_public_png),
        },
        "params": {
            "z_field": str(args.z_field),
            "lam_min": float(lam_min),
            "lam_max": float(lam_max),
            "target_dist": target_dist,
            "threshold_abs": float(thr),
            "min_tracers": int(min_tracers),
        },
        "result": {
            "promoted": bool(promoted),
            "passing_tracers": list(passing_tracers),
            "passing_tracers_n": int(len(passing_tracers)),
        },
        "by_method": summary_by_method,
        "gate_by_tracer": tracer_gate,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot (public-friendly): z-score vs shrinkage λ for passing tracers (or, if none, all target_dist tracers).
    try:
        import matplotlib.pyplot as plt

        _set_japanese_font()

        target_dist = str(args.target_dist).strip()
        thr = float(args.threshold_abs)

        if passing_tracers:
            tracers_to_plot = list(passing_tracers)
        else:
            tracers_to_plot = sorted({tr for (tr, dist) in by_tracer_dist_across_methods.keys() if dist == target_dist})

        if tracers_to_plot:
            fig, axes = plt.subplots(
                nrows=len(tracers_to_plot),
                ncols=1,
                figsize=(10.0, 3.2 * len(tracers_to_plot)),
                sharex=True,
            )
            if len(tracers_to_plot) == 1:
                axes = [axes]  # type: ignore[list-item]

            for ax, tracer in zip(axes, tracers_to_plot):
                ax.axhspan(-thr, thr, color="0.92", zorder=0)
                ax.axhline(0.0, color="0.5", linewidth=1.0, linestyle="--", zorder=1)
                ax.axhline(+thr, color="0.3", linewidth=1.0, linestyle=":", zorder=1)
                ax.axhline(-thr, color="0.3", linewidth=1.0, linestyle=":", zorder=1)

                ys: List[float] = []
                for method, pts in sorted(points_by_method.items()):
                    series = pts.get((tracer, target_dist))
                    if not series:
                        continue
                    xs = [t[0] for t in series]
                    zs = [t[1] for t in series]
                    ys.extend(zs)
                    ax.plot(xs, zs, marker="o", markersize=3.0, linewidth=1.6, label=_short_method_label(method))

                gate = tracer_gate.get(tracer) if isinstance(tracer_gate, dict) else None
                stable = bool(gate.get("stable_all_methods")) if isinstance(gate, dict) else False
                ranges = gate.get("ranges") if isinstance(gate, dict) else None
                zmin_all: Optional[float] = None
                zmax_all: Optional[float] = None
                if isinstance(ranges, list) and ranges:
                    try:
                        zmin_all = min(float(r.get("z_min")) for r in ranges if r.get("z_min") is not None)
                        zmax_all = max(float(r.get("z_max")) for r in ranges if r.get("z_max") is not None)
                    except Exception:
                        zmin_all = None
                        zmax_all = None

                title = f"{tracer}（dist={target_dist}）"
                if zmin_all is not None and zmax_all is not None:
                    title += f"  z∈[{zmin_all:.2f},{zmax_all:.2f}]"
                title += f"  stable(|z|≥{thr:g})={'yes' if stable else 'no'}"
                ax.set_title(title)

                ax.set_ylabel(str(args.z_field))
                ax.grid(True, alpha=0.25)
                if ys:
                    ymin = min(ys + [-thr])
                    ymax = max(ys + [+thr])
                    pad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
                    ax.set_ylim(ymin - pad, ymax + pad)

            axes[-1].set_xlabel("covariance shrinkage λ")
            axes[0].legend(loc="best", framealpha=0.85, fontsize=9)

            fig.suptitle(
                f"DESI DR1 BAO promotion check: target_dist={target_dist}, threshold |z|≥{thr:g}, promoted={'yes' if promoted else 'no'}",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(out_png, dpi=160)
            fig.savefig(out_public_png, dpi=160)
            plt.close(fig)
    except Exception:
        pass

    print("[desi promotion check]")
    print(f"- z_field       : {args.z_field}")
    print(f"- lam range     : {lam_min:g}..{lam_max:g}")
    print(f"- target_dist   : {target_dist}")
    print(f"- threshold_abs : {thr:g}")
    print(f"- min_tracers   : {min_tracers}")
    print(f"- promoted      : {promoted} (passing={len(passing_tracers)}: {', '.join(passing_tracers) if passing_tracers else 'none'})")
    print(f"[ok] json: {out_path}")
    if out_png.exists():
        print(f"[ok] png: {out_png}")


if __name__ == "__main__":
    main()
