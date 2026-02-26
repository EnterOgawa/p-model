from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。

    if not math.isfinite(v):
        return None

    return float(v)


@dataclass(frozen=True)
class NuclearPoint:
    step: str
    label: str
    eq_label: Optional[int]
    r1_fm: Optional[float]
    r2_fm: Optional[float]
    rc_fm: Optional[float]
    v2s_obs_fm3: Optional[float]
    v2s_pred_fm3: Optional[float]


def _load_pion_lambdas(*, root: Path) -> Dict[str, float]:
    """
    Returns:
      {'lambda_pi_pm_fm': ..., 'lambda_pi0_fm': ..., 'lambda_pi_avg_fm': ...}
    """
    qcd = root / "output" / "public" / "quantum" / "qcd_hadron_masses_baseline_metrics.json"
    # 条件分岐: `not qcd.exists()` を満たす経路を評価する。
    if not qcd.exists():
        raise SystemExit(
            "[fail] missing hadron baseline metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/qcd_hadron_masses_baseline.py\n"
            f"Expected: {qcd}"
        )

    rows = _read_json(qcd).get("rows", [])
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        raise SystemExit(f"[fail] invalid hadron baseline metrics: rows is not list: {qcd}")

    lam_pm = None
    lam_0 = None
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        label = str(r.get("label", ""))
        lam = _safe_float(r.get("compton_lambda_fm"))
        # 条件分岐: `lam is None` を満たす経路を評価する。
        if lam is None:
            continue

        # 条件分岐: `label == "π±"` を満たす経路を評価する。

        if label == "π±":
            lam_pm = lam

        # 条件分岐: `label == "π0"` を満たす経路を評価する。

        if label == "π0":
            lam_0 = lam

    # 条件分岐: `lam_pm is None or lam_0 is None` を満たす経路を評価する。

    if lam_pm is None or lam_0 is None:
        raise SystemExit("[fail] pion lambdas missing from hadron baseline metrics (expected π± and π0)")

    return {
        "lambda_pi_pm_fm": float(lam_pm),
        "lambda_pi0_fm": float(lam_0),
        "lambda_pi_avg_fm": float(0.5 * (lam_pm + lam_0)),
    }


def _extract_points(metrics: Dict[str, Any]) -> List[NuclearPoint]:
    step = str(metrics.get("step") or "")
    rows = metrics.get("results_by_dataset")
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        return []

    out: List[NuclearPoint] = []
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        label = str(row.get("label") or "")
        eq_label = None
        try:
            # 条件分岐: `row.get("eq_label") is not None` を満たす経路を評価する。
            if row.get("eq_label") is not None:
                eq_label = int(row.get("eq_label"))
        except Exception:
            eq_label = None

        # Geometry (triplet fit)

        geom: Dict[str, Any] = {}
        ft = row.get("fit_triplet")
        # 条件分岐: `isinstance(ft, dict)` を満たす経路を評価する。
        if isinstance(ft, dict):
            geom = ft.get("geometry") if isinstance(ft.get("geometry"), dict) else {}

        r1 = _safe_float(geom.get("R1_fm"))
        r2 = _safe_float(geom.get("R2_fm"))
        rc = _safe_float(geom.get("Rc_fm"))

        # Singlet v2 (obs/pred)
        v2s_obs = None
        inputs = row.get("inputs")
        # 条件分岐: `isinstance(inputs, dict)` を満たす経路を評価する。
        if isinstance(inputs, dict):
            sing = inputs.get("singlet")
            # 条件分岐: `isinstance(sing, dict)` を満たす経路を評価する。
            if isinstance(sing, dict):
                v2s_obs = _safe_float(sing.get("v2s_fm3"))

        v2s_pred = None
        fs = row.get("fit_singlet")
        # 条件分岐: `isinstance(fs, dict)` を満たす経路を評価する。
        if isinstance(fs, dict):
            ere = fs.get("ere")
            # 条件分岐: `isinstance(ere, dict)` を満たす経路を評価する。
            if isinstance(ere, dict):
                v2s_pred = _safe_float(ere.get("v2_fm3"))

        out.append(
            NuclearPoint(
                step=step,
                label=label,
                eq_label=eq_label,
                r1_fm=r1,
                r2_fm=r2,
                rc_fm=rc,
                v2s_obs_fm3=v2s_obs,
                v2s_pred_fm3=v2s_pred,
            )
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 7 / Step 7.13.2: bridge hadron (π mass → range) scale and nuclear effective-potential geometry "
            "from Step 7.9 outputs, to make future ansatz constraints explicit and reproducible."
        )
    )
    ap.add_argument(
        "--out-tag",
        default="qcd_nuclear_range_bridge",
        help="Output tag for png/json/csv (default: qcd_nuclear_range_bridge).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    pion = _load_pion_lambdas(root=root)
    lam_pi = float(pion["lambda_pi_pm_fm"])
    lam_pi_avg = float(pion["lambda_pi_avg_fm"])

    # Inputs from Step 7.9 (nuclear effective potential fits)
    in_files = [
        root / "output" / "public" / "quantum" / "nuclear_effective_potential_two_range_metrics.json",
        root / "output" / "public" / "quantum" / "nuclear_effective_potential_two_range_fit_as_rs_metrics.json",
        root / "output" / "public" / "quantum" / "nuclear_effective_potential_repulsive_core_two_range_metrics.json",
    ]
    missing = [p for p in in_files if not p.exists()]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise SystemExit(
            "[fail] missing nuclear metrics. Run Step 7.9 scripts first:\n"
            "  python -B scripts/quantum/nuclear_effective_potential_two_range.py\n"
            "  python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.9.7\n"
            "  python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.9.8\n"
            "missing:\n" + "\n".join(f"- {p}" for p in missing)
        )

    points: List[NuclearPoint] = []
    for p in in_files:
        points.extend(_extract_points(_read_json(p)))

    points = [pt for pt in points if pt.r2_fm is not None and pt.v2s_obs_fm3 is not None and pt.v2s_pred_fm3 is not None]
    # 条件分岐: `not points` を満たす経路を評価する。
    if not points:
        raise SystemExit("[fail] no valid points extracted from nuclear metrics")

    # Observed envelope (analysis-dependent proxy) across datasets.

    v2s_obs = [pt.v2s_obs_fm3 for pt in points if pt.v2s_obs_fm3 is not None]
    v2s_obs_min = float(min(v2s_obs)) if v2s_obs else float("nan")
    v2s_obs_max = float(max(v2s_obs)) if v2s_obs else float("nan")

    # --- Plot
    import matplotlib.pyplot as plt

    # Keep a stable order.
    step_order = {"7.9.6": 0, "7.9.7": 1, "7.9.8": 2}
    points_sorted = sorted(points, key=lambda p: (step_order.get(p.step, 99), p.eq_label or 0, p.label))

    labels = [f"{p.step}\n{p.label}" for p in points_sorted]
    r2 = [float(p.r2_fm or float("nan")) for p in points_sorted]
    r2_ratio = [x / lam_pi for x in r2]
    r1 = [float(p.r1_fm or float("nan")) for p in points_sorted]
    r1_ratio = [x / lam_pi for x in r1]
    rc = [float(p.rc_fm or 0.0) for p in points_sorted]
    rc_ratio = [x / lam_pi for x in rc]
    v2s_pred = [float(p.v2s_pred_fm3 or float("nan")) for p in points_sorted]
    v2s_obs_each = [float(p.v2s_obs_fm3 or float("nan")) for p in points_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=150)

    ax = axes[0]
    xs = list(range(len(labels)))
    ax.plot(xs, r2_ratio, marker="o", lw=1.8, label="R2 / λπ±")
    ax.plot(xs, r1_ratio, marker="s", lw=1.6, label="R1 / λπ±")
    # 条件分岐: `any(v > 0 for v in rc_ratio)` を満たす経路を評価する。
    if any(v > 0 for v in rc_ratio):
        ax.plot(xs, rc_ratio, marker="^", lw=1.2, label="Rc / λπ±")

    ax.axhline(1.0, color="0.35", ls="--", lw=1.0, label="λπ± scale")
    ax.axhline(2.0, color="0.55", ls=":", lw=1.0, label="2×λπ±")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("range / λπ±  (dimensionless)")
    ax.set_title("Nuclear ansatz geometry vs pion Compton scale")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True, fontsize=8, loc="upper right")

    ax = axes[1]
    ax.axhspan(v2s_obs_min, v2s_obs_max, color="tab:blue", alpha=0.10, label="obs envelope (eq18–eq19)")
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.plot(xs, v2s_pred, marker="o", lw=1.8, color="tab:orange", label="pred v2s")
    ax.plot(xs, v2s_obs_each, marker="s", lw=1.0, color="tab:blue", alpha=0.55, label="obs v2s (per dataset)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("v2s (fm³)")
    ax.set_title("Singlet shape parameter: predicted vs observed")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True, fontsize=8, loc="best")

    fig.tight_layout()

    out_tag = str(args.out_tag)
    out_png = out_dir / f"{out_tag}.png"
    out_json = out_dir / f"{out_tag}_metrics.json"
    out_csv = out_dir / f"{out_tag}.csv"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[ok] png : {out_png}")

    # --- Metrics + CSV
    metrics = {
        "generated_utc": _utc_now(),
        "step": "Phase 7 / Step 7.13.2 (QCD↔nuclear range bridge)",
        "pion_range_scale": {
            "lambda_pi_pm_fm": float(pion["lambda_pi_pm_fm"]),
            "lambda_pi0_fm": float(pion["lambda_pi0_fm"]),
            "lambda_pi_avg_fm": float(lam_pi_avg),
            "notes": [
                "λπ = ħc/(mπ c^2) from PDG RPP 2024 masses (via cached PDG mcdata file).",
                "This is used as an operational range scale to constrain/interpret nuclear u(r) ansatz geometry.",
            ],
        },
        "inputs": {
            "hadron_baseline_metrics": str(root / "output" / "public" / "quantum" / "qcd_hadron_masses_baseline_metrics.json"),
            "nuclear_metrics": [str(p) for p in in_files],
        },
        "observed_envelope_proxy": {"v2s_min_fm3": float(v2s_obs_min), "v2s_max_fm3": float(v2s_obs_max)},
        "points": [
            {
                "step": p.step,
                "label": p.label,
                "eq_label": p.eq_label,
                "R1_fm": p.r1_fm,
                "R2_fm": p.r2_fm,
                "Rc_fm": p.rc_fm,
                "R1_over_lambda_pi_pm": (None if p.r1_fm is None else float(p.r1_fm / lam_pi)),
                "R2_over_lambda_pi_pm": (None if p.r2_fm is None else float(p.r2_fm / lam_pi)),
                "Rc_over_lambda_pi_pm": (None if p.rc_fm is None else float(p.rc_fm / lam_pi)),
                "v2s_obs_fm3": p.v2s_obs_fm3,
                "v2s_pred_fm3": p.v2s_pred_fm3,
            }
            for p in points_sorted
        ],
        "outputs": {"png": str(out_png), "csv": str(out_csv)},
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] json: {out_json}")

    header = [
        "step",
        "label",
        "eq_label",
        "R1_fm",
        "R2_fm",
        "Rc_fm",
        "R1_over_lambda_pi_pm",
        "R2_over_lambda_pi_pm",
        "Rc_over_lambda_pi_pm",
        "v2s_obs_fm3",
        "v2s_pred_fm3",
    ]
    lines = [",".join(header)]
    for p in points_sorted:
        lines.append(
            ",".join(
                [
                    str(p.step),
                    str(p.label).replace(",", " "),
                    "" if p.eq_label is None else str(int(p.eq_label)),
                    "" if p.r1_fm is None else f"{p.r1_fm:.9g}",
                    "" if p.r2_fm is None else f"{p.r2_fm:.9g}",
                    "" if p.rc_fm is None else f"{p.rc_fm:.9g}",
                    "" if p.r1_fm is None else f"{(p.r1_fm/lam_pi):.9g}",
                    "" if p.r2_fm is None else f"{(p.r2_fm/lam_pi):.9g}",
                    "" if p.rc_fm is None else f"{(p.rc_fm/lam_pi):.9g}",
                    "" if p.v2s_obs_fm3 is None else f"{p.v2s_obs_fm3:.9g}",
                    "" if p.v2s_pred_fm3 is None else f"{p.v2s_pred_fm3:.9g}",
                ]
            )
        )

    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] csv : {out_csv}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

