from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


def _set_japanese_font() -> None:
    try:
        import japanize_matplotlib  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    candidates = [
        "IPAexGothic",
        "IPAGothic",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
    ]
    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    chosen = [name for name in candidates if name in installed]
    if chosen:
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False


def _out_tag_suffix(out_tag: str) -> str:
    t = str(out_tag).strip()
    if (not t) or (t == "none"):
        return ""
    if t == "any":
        return "__any"
    return f"__{t}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_metrics_file(*, root: Path, sample: str, caps: str, out_tag: str) -> Path | None:
    out_dir = root / "output" / "private" / "cosmology"
    if out_tag == "any":
        pattern = f"cosmology_bao_catalog_peakfit_{sample}_{caps}*_metrics.json"
        files = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None
    suffix = _out_tag_suffix(out_tag)
    name = f"cosmology_bao_catalog_peakfit_{sample}_{caps}{suffix}_metrics.json"
    path = out_dir / name
    return path if path.exists() else None


@dataclass(frozen=True)
class PeakfitPoint:
    sample: str
    caps: str
    dist: str
    z_eff: float
    eps: float
    sigma_eps_1sigma: float
    status: str
    source_metrics: str


def _iter_points_from_metrics(*, metrics_path: Path) -> Iterable[PeakfitPoint]:
    d = _load_json(metrics_path)
    for case in d.get("results", []):
        try:
            yield PeakfitPoint(
                sample=str(case["sample"]),
                caps=str(case["caps"]),
                dist=str(case["dist"]),
                z_eff=float(case["z_eff"]),
                eps=float(case["fit"]["free"]["eps"]),
                sigma_eps_1sigma=float(case["screening"]["sigma_eps_1sigma"]),
                status=str(case["screening"]["status"]),
                source_metrics=str(metrics_path.as_posix()),
            )
        except Exception:
            continue


def _caps_label(caps: str) -> str:
    caps = str(caps)
    if caps == "combined":
        return "combined"
    if caps == "north":
        return "NGC"
    if caps == "south":
        return "SGC"
    return caps


def _status_color(status: str) -> str:
    status = str(status)
    if status == "ok":
        return "#2ca02c"
    if status == "mixed":
        return "#ffbf00"
    if status == "ng":
        return "#d62728"
    return "#7f7f7f"


def _dist_style(dist: str) -> tuple[str, str]:
    dist = str(dist)
    if dist == "lcdm":
        return "#1f77b4", "o"
    if dist == "pbg":
        return "#ff7f0e", "s"
    return "#7f7f7f", "D"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: BAO catalog-based peakfit eps summary across caps (NGC/SGC).")
    ap.add_argument("--out-tag", type=str, default="none", help="Filter out_tag: none (default), any, or exact string")
    ap.add_argument("--samples", type=str, default="cmass,lowz", help="comma-separated: cmass,lowz (default)")
    ap.add_argument("--caps", type=str, default="combined,north,south", help="comma-separated: combined,north,south (default)")
    ap.add_argument("--dists", type=str, default="lcdm,pbg", help="comma-separated: lcdm,pbg (default)")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [s.strip() for s in str(args.samples).split(",") if s.strip()]
    caps_list = [c.strip() for c in str(args.caps).split(",") if c.strip()]
    dists = [d.strip() for d in str(args.dists).split(",") if d.strip()]
    out_tag = str(args.out_tag).strip() or "none"

    points: list[PeakfitPoint] = []
    input_files: list[str] = []
    missing: list[str] = []
    for sample in samples:
        for caps in caps_list:
            mp = _select_metrics_file(root=root, sample=sample, caps=caps, out_tag=out_tag)
            if mp is None:
                missing.append(f"{sample}/{caps}")
                continue
            input_files.append(mp.as_posix())
            for p in _iter_points_from_metrics(metrics_path=mp):
                if p.dist in dists:
                    points.append(p)

    if not points:
        raise SystemExit(f"no peakfit inputs found (out_tag={out_tag!r}; missing={missing})")

    _set_japanese_font()

    fig_w = max(12.8, 6.8 * max(1, len(samples)))
    fig, axes = plt.subplots(1, len(samples), figsize=(fig_w, 6.4), sharey=True, dpi=180)
    # matplotlib returns either a single Axes or a numpy array of Axes.
    try:
        import numpy as np

        if isinstance(axes, np.ndarray):
            axes_list = list(axes.reshape(-1))
        else:
            axes_list = [axes]
    except Exception:
        axes_list = list(axes) if isinstance(axes, (list, tuple)) else [axes]

    dist_order = dists
    caps_order = caps_list

    for ax, sample in zip(axes_list, samples):
        sub = [p for p in points if p.sample == sample]
        if not sub:
            ax.set_axis_off()
            ax.set_title(f"{sample} (no data)")
            continue

        x_map = {c: i for i, c in enumerate(caps_order)}
        for dist in dist_order:
            dist_points = [p for p in sub if p.dist == dist and p.caps in x_map]
            if not dist_points:
                continue
            dist_points = sorted(dist_points, key=lambda p: x_map[p.caps])
            xs = [x_map[p.caps] for p in dist_points]
            ys = [p.eps for p in dist_points]
            es = [p.sigma_eps_1sigma for p in dist_points]
            color, marker = _dist_style(dist)

            # light guide line (caps dependence)
            ax.plot(xs, ys, color=color, alpha=0.35, linewidth=2, zorder=1)

            # error bars with status-coded edge
            for x, y, e, p in zip(xs, ys, es, dist_points):
                ax.errorbar(
                    [x],
                    [y],
                    yerr=[e],
                    fmt=marker,
                    markersize=7,
                    color=color,
                    ecolor=color,
                    elinewidth=1.5,
                    capsize=3,
                    markerfacecolor=color,
                    markeredgecolor=_status_color(p.status),
                    markeredgewidth=2.0,
                    zorder=3,
                )

        ax.axhline(0.0, color="#777777", linewidth=1, alpha=0.6, zorder=0)
        ax.set_xticks([x_map[c] for c in caps_order], [_caps_label(c) for c in caps_order])
        ax.set_title(f"{sample}")
        ax.grid(True, alpha=0.25)

    axes_list[0].set_ylabel("ε (AP warping; peakfit, screening)")
    fig.suptitle(f"BAO catalog-based peakfit: ε across caps (out_tag={out_tag})", y=1.03)

    # Legend: distance model markers
    handles = []
    labels = []
    for dist in dist_order:
        color, marker = _dist_style(dist)
        h = mpl.lines.Line2D(
            [0],
            [0],
            marker=marker,
            color=color,
            label=dist,
            linestyle="-",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="#444444",
            markeredgewidth=1.5,
        )
        handles.append(h)
        labels.append(dist)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)

    fig.text(
        0.01,
        0.01,
        "marker edge: ok=green / mixed=yellow / ng=red   (thresholds in each peakfit *_metrics.json)",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    suffix = _out_tag_suffix(out_tag)
    out_png = out_dir / f"cosmology_bao_catalog_peakfit_caps_summary{suffix}.png"
    out_json = out_dir / f"cosmology_bao_catalog_peakfit_caps_summary{suffix}_metrics.json"
    fig.tight_layout(rect=[0.0, 0.06, 0.84, 0.92])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO catalog-based peakfit caps summary)",
        "inputs": {
            "out_tag": out_tag,
            "samples": samples,
            "caps": caps_list,
            "dists": dists,
            "input_files": input_files,
            "missing": missing,
        },
        "points": [
            {
                "sample": p.sample,
                "caps": p.caps,
                "dist": p.dist,
                "z_eff": p.z_eff,
                "eps": p.eps,
                "sigma_eps_1sigma": p.sigma_eps_1sigma,
                "status": p.status,
                "source_metrics": p.source_metrics,
            }
            for p in points
        ],
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
