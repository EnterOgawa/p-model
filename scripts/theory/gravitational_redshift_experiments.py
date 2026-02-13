from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


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


@dataclass(frozen=True)
class Experiment:
    id: str
    short_label: str
    title: str
    epsilon: float
    sigma: float
    sigma_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "Experiment":
        return Experiment(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            epsilon=float(j["epsilon"]),
            sigma=float(j["sigma"]),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_sci(x: float, *, digits: int = 3) -> str:
    return f"{x:.{digits}e}"


def compute(experiments: Sequence[Experiment]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for e in experiments:
        z = None
        if e.sigma > 0:
            z = e.epsilon / e.sigma
        rows.append(
            {
                "id": e.id,
                "short_label": e.short_label,
                "title": e.title,
                "epsilon": float(e.epsilon),
                "sigma": float(e.sigma),
                "z_score": None if z is None else float(z),
                "sigma_note": e.sigma_note,
                "source": e.source,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "short_label", "epsilon", "sigma", "z_score", "source_url", "source_doi"])
        for r in rows:
            src = r.get("source") or {}
            w.writerow(
                [
                    r.get("id", ""),
                    r.get("short_label", ""),
                    _fmt_sci(float(r.get("epsilon") or 0.0), digits=6),
                    _fmt_sci(float(r.get("sigma") or 0.0), digits=6),
                    "" if r.get("z_score") is None else f"{float(r['z_score']):.3f}",
                    str(src.get("url") or ""),
                    str(src.get("doi") or ""),
                ]
            )


def _plot(rows: Sequence[Dict[str, Any]], *, out_png: Path, scale: float) -> None:
    _set_japanese_font()

    labels = [str(r.get("short_label") or "") for r in rows]
    y = np.array([float(r["epsilon"]) / scale for r in rows], dtype=float)
    yerr = np.array([float(r["sigma"]) / scale for r in rows], dtype=float)
    x = np.arange(len(rows), dtype=float)

    fig = plt.figure(figsize=(10.5, 5.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8, label="P-model / GR 予測（ε=0）")
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color="#d62728",
        ecolor="#d62728",
        elinewidth=1.5,
        capsize=4,
        label="観測（一次ソースの公表値）",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")

    power = int(round(np.log10(scale)))
    ax.set_ylabel(f"偏差 ε（観測のずれ） [×10^{power}]")
    ax.set_title("重力赤方偏移：観測 vs P-model（一次ソース）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_data = root / "data" / "theory" / "gravitational_redshift_experiments.json"
    default_outdir = root / "output" / "private" / "theory"

    ap = argparse.ArgumentParser(description="Gravitational redshift experiments (observed deviation vs P-model).")
    ap.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Input JSON (default: data/theory/gravitational_redshift_experiments.json)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(default_outdir),
        help="Output directory (default: output/private/theory)",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1e-5,
        help="Plot scale for epsilon axis (default: 1e-5 means y-axis is epsilon / 1e-5).",
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

    png_path = outdir / "gravitational_redshift_experiments.png"
    _plot(rows, out_png=png_path, scale=float(args.scale))

    out_json = outdir / "gravitational_redshift_experiments.json"
    out_csv = outdir / "gravitational_redshift_experiments.csv"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path),
        "model": "P-model",
        "definition": src.get("definition") or {},
        "scale": float(args.scale),
        "rows": rows,
        "outputs": {"plot_png": str(png_path), "rows_json": str(out_json), "rows_csv": str(out_csv)},
    }
    _write_json(out_json, payload)
    _write_csv(out_csv, rows)

    print(f"[ok] plot : {png_path}")
    print(f"[ok] json : {out_json}")
    print(f"[ok] csv  : {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
