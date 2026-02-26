from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


MAGIC_Z = [2, 8, 20, 28, 50, 82]
MAGIC_N = [2, 8, 20, 28, 50, 82, 126]
EDGE_WIDTH = 2

OBS_ORDER = ["S_n", "S_p", "S_2n", "S_2p"]
GAP_ORDER = ["gap_n", "gap_2n", "gap_p", "gap_2p"]


def _parse_bool(value: str) -> bool:
    token = str(value).strip().lower()
    return token in {"1", "true", "t", "yes", "y"}


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


def _rms(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return math.sqrt(sum(v * v for v in finite) / float(len(finite)))


def _safe_median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return float(median(finite))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        # 条件分岐: `not rows` を満たす経路を評価する。
        if not rows:
            f.write("")
            return

        headers = list(rows[0].keys())
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h) for h in headers])


def _edge_class(*, dist_min: int, dist_max: int, edge_width: int = EDGE_WIDTH) -> str:
    min_edge = dist_min <= edge_width
    max_edge = dist_max <= edge_width
    # 条件分岐: `min_edge and max_edge` を満たす経路を評価する。
    if min_edge and max_edge:
        return "both_edges"

    # 条件分岐: `min_edge` を満たす経路を評価する。

    if min_edge:
        return "min_edge"

    # 条件分岐: `max_edge` を満たす経路を評価する。

    if max_edge:
        return "max_edge"

    return "interior"


def _read_per_nucleus_input(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(row["Z"])
            n = int(row["N"])
            a = int(row["A"])
            b_obs = float(row["B_obs_MeV"])
            b_before = float(row["B_pred_before_MeV"])
            b_after = float(row["B_pred_after_MeV"])
            # 条件分岐: `not (math.isfinite(b_obs) and math.isfinite(b_before))` を満たす経路を評価する。
            if not (math.isfinite(b_obs) and math.isfinite(b_before)):
                continue

            out[(z, n)] = {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": str(row.get("symbol", "")),
                "parity": str(row.get("parity", "")),
                "is_magic_any": _parse_bool(str(row.get("is_magic_any", "False"))),
                "B_obs_MeV": b_obs,
                "B_pred_before_MeV": b_before,
                "B_pred_after_MeV": b_after,
            }

    return out


def _compute_separation_rows(
    *,
    observable: str,
    by_zn: dict[tuple[int, int], dict[str, Any]],
    n_by_z: dict[int, list[int]],
    z_by_n: dict[int, list[int]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int], dict[str, float]] = {}

    for z, n in sorted(by_zn.keys()):
        parent = by_zn[(z, n)]
        # 条件分岐: `observable == "S_n"` を満たす経路を評価する。
        if observable == "S_n":
            ref = by_zn.get((z, n - 1))
            axis_values = n_by_z.get(z, [])
            axis_name = "N"
            axis_parent = n
            is_magic_axis = n in MAGIC_N
        # 条件分岐: 前段条件が不成立で、`observable == "S_p"` を追加評価する。
        elif observable == "S_p":
            ref = by_zn.get((z - 1, n))
            axis_values = z_by_n.get(n, [])
            axis_name = "Z"
            axis_parent = z
            is_magic_axis = z in MAGIC_Z
        # 条件分岐: 前段条件が不成立で、`observable == "S_2n"` を追加評価する。
        elif observable == "S_2n":
            ref = by_zn.get((z, n - 2))
            axis_values = n_by_z.get(z, [])
            axis_name = "N"
            axis_parent = n
            is_magic_axis = n in MAGIC_N
        # 条件分岐: 前段条件が不成立で、`observable == "S_2p"` を追加評価する。
        elif observable == "S_2p":
            ref = by_zn.get((z - 2, n))
            axis_values = z_by_n.get(n, [])
            axis_name = "Z"
            axis_parent = z
            is_magic_axis = z in MAGIC_Z
        else:
            raise ValueError(f"unknown observable: {observable}")

        # 条件分岐: `ref is None` を満たす経路を評価する。

        if ref is None:
            continue

        # 条件分岐: `not axis_values` を満たす経路を評価する。

        if not axis_values:
            continue

        sep_obs = float(parent["B_obs_MeV"] - ref["B_obs_MeV"])
        sep_before = float(parent["B_pred_before_MeV"] - ref["B_pred_before_MeV"])
        b_after_parent = float(parent["B_pred_after_MeV"])
        b_after_ref = float(ref["B_pred_after_MeV"])
        # 条件分岐: `math.isfinite(b_after_parent) and math.isfinite(b_after_ref)` を満たす経路を評価する。
        if math.isfinite(b_after_parent) and math.isfinite(b_after_ref):
            sep_after = float(b_after_parent - b_after_ref)
            resid_after = float(sep_after - sep_obs)
        else:
            sep_after = float("nan")
            resid_after = float("nan")

        resid_before = float(sep_before - sep_obs)
        axis_min = int(min(axis_values))
        axis_max = int(max(axis_values))
        dist_min = int(axis_parent - axis_min)
        dist_max = int(axis_max - axis_parent)

        row = {
            "observable": observable,
            "Z_parent": z,
            "N_parent": n,
            "A_parent": int(parent["A"]),
            "symbol": str(parent["symbol"]),
            "parity": str(parent["parity"]),
            "is_magic_any": bool(parent["is_magic_any"]),
            "is_magic_axis": bool(is_magic_axis),
            "axis_name": axis_name,
            "axis_parent": axis_parent,
            "axis_min": axis_min,
            "axis_max": axis_max,
            "axis_span_n": len(axis_values),
            "distance_min_edge": dist_min,
            "distance_max_edge": dist_max,
            "edge_class": _edge_class(dist_min=dist_min, dist_max=dist_max),
            "sep_obs_MeV": sep_obs,
            "sep_pred_before_MeV": sep_before,
            "sep_pred_after_MeV": sep_after,
            "resid_before_MeV": resid_before,
            "resid_after_MeV": resid_after,
            "abs_resid_before_MeV": abs(resid_before),
            "abs_resid_after_MeV": abs(resid_after) if math.isfinite(resid_after) else float("nan"),
        }
        rows.append(row)
        lookup[(z, n)] = {
            "obs": sep_obs,
            "pred_before": sep_before,
            "pred_after": sep_after,
        }

    return rows, lookup


def _build_magic_gap_rows(
    *,
    sep_lookup: dict[str, dict[tuple[int, int], dict[str, float]]],
    n_by_z: dict[int, list[int]],
    z_by_n: dict[int, list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _append_gap(
        *,
        gap_name: str,
        fixed_axis: str,
        fixed_value: int,
        magic_anchor: int,
        left: dict[str, float],
        right: dict[str, float],
    ) -> None:
        obs = float(left["obs"] - right["obs"])
        pred_before = float(left["pred_before"] - right["pred_before"])
        a_after = float(left["pred_after"])
        b_after = float(right["pred_after"])
        # 条件分岐: `math.isfinite(a_after) and math.isfinite(b_after)` を満たす経路を評価する。
        if math.isfinite(a_after) and math.isfinite(b_after):
            pred_after = float(a_after - b_after)
            resid_after = float(pred_after - obs)
        else:
            pred_after = float("nan")
            resid_after = float("nan")

        resid_before = float(pred_before - obs)
        rows.append(
            {
                "gap_observable": gap_name,
                "fixed_axis": fixed_axis,
                "fixed_value": fixed_value,
                "magic_anchor": magic_anchor,
                "gap_obs_MeV": obs,
                "gap_pred_before_MeV": pred_before,
                "gap_pred_after_MeV": pred_after,
                "resid_before_MeV": resid_before,
                "resid_after_MeV": resid_after,
                "abs_resid_before_MeV": abs(resid_before),
                "abs_resid_after_MeV": abs(resid_after) if math.isfinite(resid_after) else float("nan"),
            }
        )

    sn = sep_lookup["S_n"]
    s2n = sep_lookup["S_2n"]
    for z in sorted(n_by_z.keys()):
        for n0 in MAGIC_N:
            left = sn.get((z, n0))
            right = sn.get((z, n0 + 1))
            # 条件分岐: `left is not None and right is not None` を満たす経路を評価する。
            if left is not None and right is not None:
                _append_gap(
                    gap_name="gap_n",
                    fixed_axis="Z",
                    fixed_value=z,
                    magic_anchor=n0,
                    left=left,
                    right=right,
                )

            left2 = s2n.get((z, n0))
            right2 = s2n.get((z, n0 + 2))
            # 条件分岐: `left2 is not None and right2 is not None` を満たす経路を評価する。
            if left2 is not None and right2 is not None:
                _append_gap(
                    gap_name="gap_2n",
                    fixed_axis="Z",
                    fixed_value=z,
                    magic_anchor=n0,
                    left=left2,
                    right=right2,
                )

    sp = sep_lookup["S_p"]
    s2p = sep_lookup["S_2p"]
    for n in sorted(z_by_n.keys()):
        for z0 in MAGIC_Z:
            left = sp.get((z0, n))
            right = sp.get((z0 + 1, n))
            # 条件分岐: `left is not None and right is not None` を満たす経路を評価する。
            if left is not None and right is not None:
                _append_gap(
                    gap_name="gap_p",
                    fixed_axis="N",
                    fixed_value=n,
                    magic_anchor=z0,
                    left=left,
                    right=right,
                )

            left2 = s2p.get((z0, n))
            right2 = s2p.get((z0 + 2, n))
            # 条件分岐: `left2 is not None and right2 is not None` を満たす経路を評価する。
            if left2 is not None and right2 is not None:
                _append_gap(
                    gap_name="gap_2p",
                    fixed_axis="N",
                    fixed_value=n,
                    magic_anchor=z0,
                    left=left2,
                    right=right2,
                )

    return rows


def _summarize_observables(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for obs in OBS_ORDER:
        sub = [r for r in rows if str(r["observable"]) == obs]
        before = [float(r["resid_before_MeV"]) for r in sub]
        after = [float(r["resid_after_MeV"]) for r in sub if math.isfinite(float(r["resid_after_MeV"]))]
        abs_before = [abs(v) for v in before]
        abs_after = [abs(v) for v in after]
        edge_after = [
            abs(float(r["resid_after_MeV"]))
            for r in sub
            if str(r["edge_class"]) != "interior" and math.isfinite(float(r["resid_after_MeV"]))
        ]
        interior_after = [
            abs(float(r["resid_after_MeV"]))
            for r in sub
            if str(r["edge_class"]) == "interior" and math.isfinite(float(r["resid_after_MeV"]))
        ]
        edge_med = _safe_median(edge_after)
        interior_med = _safe_median(interior_after)
        # 条件分岐: `math.isfinite(edge_med) and math.isfinite(interior_med) and interior_med > 0.0` を満たす経路を評価する。
        if math.isfinite(edge_med) and math.isfinite(interior_med) and interior_med > 0.0:
            edge_ratio = float(edge_med / interior_med)
        else:
            edge_ratio = float("nan")

        out.append(
            {
                "group_type": "observable",
                "group": obs,
                "n": len(sub),
                "rms_resid_before_MeV": _rms(before),
                "rms_resid_after_MeV": _rms(after),
                "median_abs_resid_before_MeV": _safe_median(abs_before),
                "median_abs_resid_after_MeV": _safe_median(abs_after),
                "median_abs_resid_after_edge_MeV": edge_med,
                "median_abs_resid_after_interior_MeV": interior_med,
                "edge_to_interior_ratio_after": edge_ratio,
            }
        )

    return out


def _summarize_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    edge_labels = ["interior", "min_edge", "max_edge", "both_edges"]
    for obs in OBS_ORDER:
        for edge_name in edge_labels:
            sub = [r for r in rows if str(r["observable"]) == obs and str(r["edge_class"]) == edge_name]
            before = [float(r["resid_before_MeV"]) for r in sub]
            after = [float(r["resid_after_MeV"]) for r in sub if math.isfinite(float(r["resid_after_MeV"]))]
            out.append(
                {
                    "group_type": "edge",
                    "group": f"{obs}:{edge_name}",
                    "n": len(sub),
                    "rms_resid_before_MeV": _rms(before),
                    "rms_resid_after_MeV": _rms(after),
                    "median_abs_resid_before_MeV": _safe_median([abs(v) for v in before]),
                    "median_abs_resid_after_MeV": _safe_median([abs(v) for v in after]),
                }
            )

    return out


def _summarize_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for gap_name in GAP_ORDER:
        sub = [r for r in rows if str(r["gap_observable"]) == gap_name]
        before = [float(r["resid_before_MeV"]) for r in sub]
        after = [float(r["resid_after_MeV"]) for r in sub if math.isfinite(float(r["resid_after_MeV"]))]
        out.append(
            {
                "group_type": "magic_gap",
                "group": gap_name,
                "n": len(sub),
                "rms_resid_before_MeV": _rms(before),
                "rms_resid_after_MeV": _rms(after),
                "median_abs_resid_before_MeV": _safe_median([abs(v) for v in before]),
                "median_abs_resid_after_MeV": _safe_median([abs(v) for v in after]),
            }
        )

    return out


def _build_figure(
    *,
    sep_rows: list[dict[str, Any]],
    observable_summary: list[dict[str, Any]],
    gap_summary: list[dict[str, Any]],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    x = list(range(len(OBS_ORDER)))
    med_before = [float(next(r for r in observable_summary if str(r["group"]) == obs)["median_abs_resid_before_MeV"]) for obs in OBS_ORDER]
    med_after = [float(next(r for r in observable_summary if str(r["group"]) == obs)["median_abs_resid_after_MeV"]) for obs in OBS_ORDER]
    ax00.bar([i - 0.18 for i in x], med_before, width=0.36, label="before pairing corr.", alpha=0.9)
    ax00.bar([i + 0.18 for i in x], med_after, width=0.36, label="after pairing corr.", alpha=0.9)
    ax00.set_xticks(x)
    ax00.set_xticklabels(OBS_ORDER)
    ax00.set_ylabel("median abs residual [MeV]")
    ax00.set_title("Residual scale by separation observable")
    ax00.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax00.legend(loc="best", fontsize=8)

    edge_ratio = [float(next(r for r in observable_summary if str(r["group"]) == obs)["edge_to_interior_ratio_after"]) for obs in OBS_ORDER]
    ax01.bar(OBS_ORDER, edge_ratio, color="#4c78a8", alpha=0.9)
    ax01.axhline(1.0, color="#444444", ls="--", lw=1.0)
    ax01.set_ylabel("median abs(edge) / median abs(interior)")
    ax01.set_title("Dripline-proxy edge inflation (after correction)")
    ax01.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    colors = {
        "S_n": "#4c78a8",
        "S_p": "#f58518",
        "S_2n": "#54a24b",
        "S_2p": "#e45756",
    }
    for obs in OBS_ORDER:
        sub = [r for r in sep_rows if str(r["observable"]) == obs and math.isfinite(float(r["resid_after_MeV"]))]
        xs = [int(r["A_parent"]) for r in sub]
        ys = [float(r["resid_after_MeV"]) for r in sub]
        ax10.scatter(xs, ys, s=8.0, alpha=0.18, color=colors[obs], label=obs)

    ax10.set_xlabel("A")
    ax10.set_ylabel("residual after correction [MeV]")
    ax10.set_yscale("symlog", linthresh=1.0)
    ax10.set_title("A-dependence of residuals (after correction)")
    ax10.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax10.legend(loc="best", fontsize=8)

    gap_vals = [float(next(r for r in gap_summary if str(r["group"]) == gap)["median_abs_resid_after_MeV"]) for gap in GAP_ORDER]
    gap_counts = [int(next(r for r in gap_summary if str(r["group"]) == gap)["n"]) for gap in GAP_ORDER]
    ax11.bar(GAP_ORDER, gap_vals, color="#72b7b2", alpha=0.9)
    for idx, n in enumerate(gap_counts):
        ax11.text(idx, gap_vals[idx], f"n={n}", ha="center", va="bottom", fontsize=8)

    ax11.set_ylabel("median abs residual [MeV]")
    ax11.set_title("Magic-gap diagnostics (after correction)")
    ax11.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.16.15: separation-energy systematics (S_n/S_p/S_2n/S_2p)", y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = out_dir / "nuclear_pairing_effect_systematics_per_nucleus.csv"
    # 条件分岐: `not in_csv.exists()` を満たす経路を評価する。
    if not in_csv.exists():
        raise SystemExit(
            f"[fail] missing required input: {in_csv}\n"
            "Run Step 7.16.3 first:\n"
            "  python -B scripts/quantum/nuclear_pairing_effect_systematics_analysis.py"
        )

    by_zn = _read_per_nucleus_input(in_csv)
    # 条件分岐: `not by_zn` を満たす経路を評価する。
    if not by_zn:
        raise SystemExit(f"[fail] no usable rows in {in_csv}")

    n_by_z: dict[int, list[int]] = defaultdict(list)
    z_by_n: dict[int, list[int]] = defaultdict(list)
    for z, n in by_zn.keys():
        n_by_z[z].append(n)
        z_by_n[n].append(z)

    for z in list(n_by_z.keys()):
        n_by_z[z] = sorted(set(n_by_z[z]))

    for n in list(z_by_n.keys()):
        z_by_n[n] = sorted(set(z_by_n[n]))

    sep_rows: list[dict[str, Any]] = []
    sep_lookup: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    for obs in OBS_ORDER:
        rows, lookup = _compute_separation_rows(
            observable=obs,
            by_zn=by_zn,
            n_by_z=n_by_z,
            z_by_n=z_by_n,
        )
        sep_rows.extend(rows)
        sep_lookup[obs] = lookup

    # 条件分岐: `not sep_rows` を満たす経路を評価する。

    if not sep_rows:
        raise SystemExit("[fail] no separation-energy rows produced")

    sep_rows = sorted(
        sep_rows,
        key=lambda r: (
            OBS_ORDER.index(str(r["observable"])),
            int(r["Z_parent"]),
            int(r["N_parent"]),
        ),
    )

    gap_rows = _build_magic_gap_rows(sep_lookup=sep_lookup, n_by_z=n_by_z, z_by_n=z_by_n)
    gap_rows = sorted(gap_rows, key=lambda r: (GAP_ORDER.index(str(r["gap_observable"])), str(r["fixed_axis"]), int(r["fixed_value"]), int(r["magic_anchor"])))

    observable_summary = _summarize_observables(sep_rows)
    edge_summary = _summarize_edges(sep_rows)
    gap_summary = _summarize_gaps(gap_rows)
    summary_rows = observable_summary + edge_summary + gap_summary

    out_full_csv = out_dir / "nuclear_separation_energy_systematics_full.csv"
    out_summary_csv = out_dir / "nuclear_separation_energy_systematics_summary.csv"
    out_gap_csv = out_dir / "nuclear_separation_energy_systematics_magic_kink.csv"
    out_png = out_dir / "nuclear_separation_energy_systematics_quantification.png"
    out_json = out_dir / "nuclear_separation_energy_systematics_metrics.json"

    _write_csv(out_full_csv, sep_rows)
    _write_csv(out_summary_csv, summary_rows)
    _write_csv(out_gap_csv, gap_rows)
    _build_figure(
        sep_rows=sep_rows,
        observable_summary=observable_summary,
        gap_summary=gap_summary,
        out_png=out_png,
    )

    counts_by_obs = {
        obs: len([r for r in sep_rows if str(r["observable"]) == obs])
        for obs in OBS_ORDER
    }
    counts_by_gap = {
        gap: len([r for r in gap_rows if str(r["gap_observable"]) == gap])
        for gap in GAP_ORDER
    }

    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.16.15",
                "inputs": {
                    "pairing_per_nucleus_csv": {"path": str(in_csv), "sha256": _sha256(in_csv)},
                },
                "definitions": {
                    "S_n": "S_n(Z,N)=B(Z,N)-B(Z,N-1)",
                    "S_p": "S_p(Z,N)=B(Z,N)-B(Z-1,N)",
                    "S_2n": "S_2n(Z,N)=B(Z,N)-B(Z,N-2)",
                    "S_2p": "S_2p(Z,N)=B(Z,N)-B(Z-2,N)",
                    "residual": "S_pred - S_obs",
                    "magic_gap_n": "gap_n(Z;N0)=S_n(Z,N0)-S_n(Z,N0+1)",
                    "magic_gap_2n": "gap_2n(Z;N0)=S_2n(Z,N0)-S_2n(Z,N0+2)",
                    "magic_gap_p": "gap_p(N;Z0)=S_p(Z0,N)-S_p(Z0+1,N)",
                    "magic_gap_2p": "gap_2p(N;Z0)=S_2p(Z0,N)-S_2p(Z0+2,N)",
                    "dripline_proxy_edge": "within each fixed-axis chain envelope, distance<=2 from min/max axis index",
                },
                "counts": {
                    "n_nuclei_input": len(by_zn),
                    "n_total_separation_rows": len(sep_rows),
                    "n_total_gap_rows": len(gap_rows),
                    "by_observable": counts_by_obs,
                    "by_gap": counts_by_gap,
                },
                "observable_summary": observable_summary,
                "edge_summary": edge_summary,
                "gap_summary": gap_summary,
                "outputs": {
                    "full_csv": str(out_full_csv),
                    "summary_csv": str(out_summary_csv),
                    "magic_kink_csv": str(out_gap_csv),
                    "figure_png": str(out_png),
                },
                "notes": [
                    "Step 7.16.15 freezes an independent cross-check I/F with S_n/S_p/S_2n/S_2p using fixed per-nucleus B predictions from Step 7.16.3.",
                    "Magic-gap diagnostics are evaluated without re-fitting and reported as independent residual channels.",
                    "Dripline-proxy stress is tracked by chain-envelope edge distance (<=2) versus interior medians.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_full_csv}")
    print(f"  {out_summary_csv}")
    print(f"  {out_gap_csv}")
    print(f"  {out_png}")
    print(f"  {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

