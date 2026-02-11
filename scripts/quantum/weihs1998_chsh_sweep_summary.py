from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class RunSpec:
    subdir: str
    run: str
    out_tag: str


def _default_runs() -> List[RunSpec]:
    base = "weihs1998_longdist_"
    runs = ["longdist0", "longdist1", "longdist2", "longdist10"]
    return [RunSpec(subdir="longdist", run=r, out_tag=base + r) for r in runs]


def _load_sweep(csv_path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            x = float(row["window_ns"])
            y_s = row.get("S_fixed_abs")
            y = float(y_s) if y_s not in (None, "", "nan") else float("nan")
            xs.append(x)
            ys.append(y)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize Weihs 1998 CHSH window sweeps across multiple runs (requires per-run CSV outputs)."
    )
    ap.add_argument(
        "--summary-tag",
        default="longdist",
        help="Output tag for the summary figure/json (default: longdist).",
    )
    ap.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help=(
            "Run specs as 'subdir:run:out_tag'. "
            "Default: longdist{0,1,2,10} (out_tag=weihs1998_longdist_<run>)."
        ),
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.runs:
        specs: list[RunSpec] = []
        for s in args.runs:
            parts = str(s).split(":")
            if len(parts) != 3:
                raise SystemExit(f"[fail] invalid --runs item: {s} (expected subdir:run:out_tag)")
            specs.append(RunSpec(subdir=parts[0], run=parts[1], out_tag=parts[2]))
    else:
        specs = _default_runs()

    series: list[dict[str, object]] = []
    missing: list[str] = []

    for sp in specs:
        csv_path = out_dir / f"weihs1998_chsh_sweep__{sp.out_tag}.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        xs, ys = _load_sweep(csv_path)
        best = max((y for y in ys if y == y), default=float("nan"))  # ignore NaN
        series.append(
            {
                "subdir": sp.subdir,
                "run": sp.run,
                "out_tag": sp.out_tag,
                "csv": str(csv_path),
                "max_abs_S_fixed": float(best),
            }
        )

    if missing:
        print("[fail] missing per-run sweeps (run weihs1998_time_tag_reanalysis.py first):")
        for p in missing:
            print(f"- {p}")
        raise SystemExit(1)

    # Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=150)
    for sp in specs:
        csv_path = out_dir / f"weihs1998_chsh_sweep__{sp.out_tag}.csv"
        xs, ys = _load_sweep(csv_path)
        ax.plot(xs, ys, marker="o", lw=1.8, label=f"{sp.subdir}/{sp.run}")
    ax.axhline(2.0, color="0.25", ls="--", lw=1.0, label="local bound |S|=2")
    ax.set_xlabel("coincidence window half-width (ns)")
    ax.set_ylabel("|S| (fixed CHSH variant)")
    ax.set_title("Weihs 1998 (Zenodo 7185335): coincidence-window sensitivity across runs/subdirs")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()

    summary_tag = str(args.summary_tag)
    out_png = out_dir / f"weihs1998_chsh_sweep_summary__{summary_tag}.png"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "dataset": "Weihs et al. 1998 time-tag (Zenodo 7185335)",
        "series": series,
        "outputs": {"png": str(out_png)},
        "repro": {
            "fetch": "python -B scripts/quantum/fetch_weihs1998_zenodo_7185335.py  (offline cached in this repo)",
            "per_run_sweep": (
                "python -B scripts/quantum/weihs1998_time_tag_reanalysis.py "
                "--subdir longdist --run <run> --encoding bit0-setting --out-tag weihs1998_longdist_<run>"
            ),
            "summary": f"python -B scripts/quantum/weihs1998_chsh_sweep_summary.py --summary-tag {summary_tag}",
        },
    }
    out_json = out_dir / f"weihs1998_chsh_sweep_summary__{summary_tag}_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
