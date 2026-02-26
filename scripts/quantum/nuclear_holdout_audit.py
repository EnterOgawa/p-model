from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


MAGIC_NUMBERS = (2, 8, 20, 28, 50, 82, 126)


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_percentile` の入出力契約と処理意図を定義する。

def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """
    Inclusive percentile with linear interpolation.
    p in [0,100].
    """
    # 条件分岐: `not sorted_vals` を満たす経路を評価する。
    if not sorted_vals:
        raise ValueError("empty")

    # 条件分岐: `p <= 0` を満たす経路を評価する。

    if p <= 0:
        return float(sorted_vals[0])

    # 条件分岐: `p >= 100` を満たす経路を評価する。

    if p >= 100:
        return float(sorted_vals[-1])

    x = (len(sorted_vals) - 1) * (p / 100.0)
    i0 = int(math.floor(x))
    i1 = int(math.ceil(x))
    # 条件分岐: `i0 == i1` を満たす経路を評価する。
    if i0 == i1:
        return float(sorted_vals[i0])

    w = x - i0
    return float((1.0 - w) * float(sorted_vals[i0]) + w * float(sorted_vals[i1]))


# 関数: `_safe_median` の入出力契約と処理意図を定義する。

def _safe_median(vals: Sequence[float]) -> float:
    s = sorted(float(v) for v in vals if math.isfinite(float(v)))
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return float("nan")

    return _percentile(s, 50.0)


# 関数: `_mad` の入出力契約と処理意図を定義する。

def _mad(vals: Sequence[float], *, center: float) -> float:
    dev = [abs(float(v) - float(center)) for v in vals if math.isfinite(float(v))]
    return _safe_median(dev)


# 関数: `_is_truthy` の入出力契約と処理意図を定義する。

def _is_truthy(v: object) -> bool:
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")


# 関数: `_edge_class` の入出力契約と処理意図を定義する。

def _edge_class(*, d_proton: int, d_neutron: int, edge_width: int = 2) -> str:
    proton_edge = d_proton <= edge_width
    neutron_edge = d_neutron <= edge_width
    # 条件分岐: `proton_edge and neutron_edge` を満たす経路を評価する。
    if proton_edge and neutron_edge:
        return "both_edges"

    # 条件分岐: `proton_edge` を満たす経路を評価する。

    if proton_edge:
        return "proton_rich_edge"

    # 条件分岐: `neutron_edge` を満たす経路を評価する。

    if neutron_edge:
        return "neutron_rich_edge"

    return "interior"


# 関数: `_is_near_magic` の入出力契約と処理意図を定義する。

def _is_near_magic(*, z: int, n: int, width: int = 2) -> bool:
    return (min(abs(z - m) for m in MAGIC_NUMBERS) <= width) or (min(abs(n - m) for m in MAGIC_NUMBERS) <= width)


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        # 条件分岐: `not rows` を満たす経路を評価する。
        if not rows:
            f.write("")
            return

        headers = list(rows[0].keys())
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h) for h in headers])


# クラス: `NucleusRow` の責務と境界条件を定義する。

@dataclass(frozen=True)
class NucleusRow:
    z: int
    n: int
    a: int
    symbol: str
    b_obs_mev: float
    b_pred_mev: float
    log10_ratio: float
    z_robust: float
    abs_z: float
    is_outlier_abs_z_gt3: bool
    is_magic_any: bool
    is_near_magic: bool
    edge_class: str
    nz_ratio: float
    beta2: Optional[float]
    is_deformed_abs_beta2_ge_0p20: bool


# 関数: `_group_stats` の入出力契約と処理意図を定義する。

def _group_stats(rows: Sequence[NucleusRow], *, group_id: str) -> Dict[str, Any]:
    log_vals = [r.log10_ratio for r in rows if math.isfinite(r.log10_ratio)]
    abs_z_vals = [r.abs_z for r in rows if math.isfinite(r.abs_z)]
    out_n = sum(1 for r in rows if r.is_outlier_abs_z_gt3)
    out_frac = float(out_n) / float(len(rows)) if rows else float("nan")
    s_log = sorted(log_vals)
    s_absz = sorted(abs_z_vals)
    return {
        "group_id": group_id,
        "n": int(len(rows)),
        "outlier_abs_z_gt3_n": int(out_n),
        "outlier_abs_z_gt3_frac": out_frac,
        "median_log10_ratio": _percentile(s_log, 50.0) if s_log else float("nan"),
        "p16_log10_ratio": _percentile(s_log, 16.0) if s_log else float("nan"),
        "p84_log10_ratio": _percentile(s_log, 84.0) if s_log else float("nan"),
        "median_abs_z": _percentile(s_absz, 50.0) if s_absz else float("nan"),
        "p84_abs_z": _percentile(s_absz, 84.0) if s_absz else float("nan"),
        "max_abs_z": float(max(s_absz)) if s_absz else float("nan"),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    out_dir = _ROOT / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = out_dir / "nuclear_binding_energy_frequency_mapping_minimal_additional_physics.csv"
    beta2_json = _ROOT / "data" / "quantum" / "sources" / "nndc_be2_adopted_entries" / "extracted_beta2.json"
    # 条件分岐: `not in_csv.exists()` を満たす経路を評価する。
    if not in_csv.exists():
        raise SystemExit(
            "[fail] missing input CSV.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics.py\n"
            f"Expected: {in_csv}"
        )

    # 条件分岐: `not beta2_json.exists()` を満たす経路を評価する。

    if not beta2_json.exists():
        raise SystemExit(f"[fail] missing beta2 extracted JSON: {beta2_json}")

    beta2_rows = json.loads(beta2_json.read_text(encoding="utf-8")).get("rows")
    beta2_map: Dict[Tuple[int, int], float] = {}
    # 条件分岐: `isinstance(beta2_rows, list)` を満たす経路を評価する。
    if isinstance(beta2_rows, list):
        for r in beta2_rows:
            # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
            if not isinstance(r, dict):
                continue

            try:
                z = int(r["Z"])
                n = int(r["N"])
                b2 = float(r["beta2"])
            except Exception:
                continue

            # 条件分岐: `math.isfinite(b2)` を満たす経路を評価する。

            if math.isfinite(b2):
                beta2_map[(z, n)] = float(b2)

    # First pass: read rows and compute log10 ratios.

    raw: List[Dict[str, Any]] = []
    log_all: List[float] = []
    by_z_nlist: Dict[int, List[int]] = {}
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                z = int(row["Z"])
                n = int(row["N"])
                a = int(row["A"])
                b_obs = float(row["B_obs_MeV"])
                b_pred = float(row["B_pred_local_spacing_sat_MeV"])
            except Exception:
                continue

            # 条件分岐: `not (math.isfinite(b_obs) and b_obs > 0 and math.isfinite(b_pred) and b_pred...` を満たす経路を評価する。

            if not (math.isfinite(b_obs) and b_obs > 0 and math.isfinite(b_pred) and b_pred > 0):
                continue

            lr = math.log10(b_pred / b_obs)
            # 条件分岐: `not math.isfinite(lr)` を満たす経路を評価する。
            if not math.isfinite(lr):
                continue

            raw.append({**row, "_Z": z, "_N": n, "_A": a, "_B_obs": b_obs, "_B_pred": b_pred, "_log10_ratio": lr})
            log_all.append(lr)
            by_z_nlist.setdefault(z, []).append(n)

    # 条件分岐: `not raw` を満たす経路を評価する。

    if not raw:
        raise SystemExit(f"[fail] parsed 0 usable rows from: {in_csv}")

    for z in list(by_z_nlist.keys()):
        by_z_nlist[z] = sorted(set(by_z_nlist[z]))

    # Robust baseline (global).

    global_med = _safe_median(log_all)
    mad = _mad(log_all, center=global_med)
    robust_sigma = max(1.4826 * float(mad), 1.0e-12)

    # N/Z ratio thresholds for "extreme" categories.
    nz_ratios = [float(r["_N"]) / float(r["_Z"]) for r in raw if int(r["_Z"]) > 0]
    nz_sorted = sorted(nz_ratios)
    nz_p05 = _percentile(nz_sorted, 5.0)
    nz_p95 = _percentile(nz_sorted, 95.0)

    rows: List[NucleusRow] = []
    for r in raw:
        z = int(r["_Z"])
        n = int(r["_N"])
        a = int(r["_A"])
        b_obs = float(r["_B_obs"])
        b_pred = float(r["_B_pred"])
        lr = float(r["_log10_ratio"])

        z_robust = (lr - global_med) / robust_sigma
        abs_z = abs(z_robust)
        is_outlier = abs_z > 3.0

        is_magic_any = _is_truthy(r.get("is_magic_any", False))
        is_near_magic = _is_near_magic(z=z, n=n, width=2)

        n_list = by_z_nlist.get(z) or []
        # 条件分岐: `not n_list` を満たす経路を評価する。
        if not n_list:
            d_proton = 999999
            d_neutron = 999999
            edge_class = "unknown"
        else:
            n_min = int(min(n_list))
            n_max = int(max(n_list))
            d_proton = int(n - n_min)
            d_neutron = int(n_max - n)
            edge_class = _edge_class(d_proton=d_proton, d_neutron=d_neutron, edge_width=2)

        nz_ratio = float(n) / float(z) if z > 0 else float("nan")

        beta2 = beta2_map.get((z, n))
        is_deformed = bool((beta2 is not None) and math.isfinite(float(beta2)) and abs(float(beta2)) >= 0.20 and (not is_magic_any))

        rows.append(
            NucleusRow(
                z=z,
                n=n,
                a=a,
                symbol=str(r.get("symbol", "")),
                b_obs_mev=b_obs,
                b_pred_mev=b_pred,
                log10_ratio=lr,
                z_robust=float(z_robust),
                abs_z=float(abs_z),
                is_outlier_abs_z_gt3=bool(is_outlier),
                is_magic_any=bool(is_magic_any),
                is_near_magic=bool(is_near_magic),
                edge_class=str(edge_class),
                nz_ratio=nz_ratio,
                beta2=float(beta2) if beta2 is not None and math.isfinite(float(beta2)) else None,
                is_deformed_abs_beta2_ge_0p20=bool(is_deformed),
            )
        )

    # Define groups (fixed).

    group_defs: List[Tuple[str, Callable[[NucleusRow], bool]]] = [
        ("all", lambda r: True),
        ("magic_any", lambda r: r.is_magic_any),
        ("nonmagic", lambda r: not r.is_magic_any),
        ("near_magic_width2", lambda r: r.is_near_magic),
        ("deformed_abs_beta2_ge_0p20_nonmagic", lambda r: r.is_deformed_abs_beta2_ge_0p20),
        ("edge_interior", lambda r: r.edge_class == "interior"),
        ("edge_proton_rich", lambda r: r.edge_class == "proton_rich_edge"),
        ("edge_neutron_rich", lambda r: r.edge_class == "neutron_rich_edge"),
        ("edge_both", lambda r: r.edge_class == "both_edges"),
        ("nz_ratio_low_p05", lambda r: math.isfinite(r.nz_ratio) and r.nz_ratio <= nz_p05),
        ("nz_ratio_high_p95", lambda r: math.isfinite(r.nz_ratio) and r.nz_ratio >= nz_p95),
    ]

    groups = []
    groups_csv_rows: List[Dict[str, Any]] = []
    for gid, fn in group_defs:
        g_rows = [r for r in rows if fn(r)]
        s = _group_stats(g_rows, group_id=gid)
        groups.append(s)
        groups_csv_rows.append(s)

    # Outliers CSV (full; sorted by abs_z desc).

    outlier_rows_sorted = sorted(rows, key=lambda rr: (float(rr.abs_z) if math.isfinite(rr.abs_z) else -1.0), reverse=True)
    outlier_csv_rows: List[Dict[str, Any]] = []
    for r in outlier_rows_sorted:
        outlier_csv_rows.append(
            {
                "Z": r.z,
                "N": r.n,
                "A": r.a,
                "symbol": r.symbol,
                "B_obs_MeV": f"{r.b_obs_mev:.6f}",
                "B_pred_MeV": f"{r.b_pred_mev:.6f}",
                "log10_ratio": f"{r.log10_ratio:.6f}",
                "z_robust": f"{r.z_robust:.6f}",
                "abs_z": f"{r.abs_z:.6f}",
                "is_outlier_abs_z_gt3": bool(r.is_outlier_abs_z_gt3),
                "is_magic_any": bool(r.is_magic_any),
                "is_near_magic_width2": bool(r.is_near_magic),
                "edge_class": r.edge_class,
                "N_over_Z": f"{r.nz_ratio:.6f}" if math.isfinite(r.nz_ratio) else "",
                "beta2": f"{r.beta2:.6f}" if r.beta2 is not None else "",
                "is_deformed_abs_beta2_ge_0p20_nonmagic": bool(r.is_deformed_abs_beta2_ge_0p20),
            }
        )

    out_summary = out_dir / "nuclear_holdout_audit_summary.json"
    out_groups_csv = out_dir / "nuclear_holdout_audit_groups.csv"
    out_outliers_csv = out_dir / "nuclear_holdout_audit_outliers.csv"
    out_png = out_dir / "nuclear_holdout_audit.png"

    _write_csv(out_groups_csv, groups_csv_rows)
    _write_csv(out_outliers_csv, outlier_csv_rows)

    # Plot (simple bar charts for key groups).
    try:
        import matplotlib.pyplot as plt

        key_order = [gid for gid, _ in group_defs]
        by_id = {g["group_id"]: g for g in groups}
        outlier_fracs = [float(by_id[gid]["outlier_abs_z_gt3_frac"]) for gid in key_order]
        med_absz = [float(by_id[gid]["median_abs_z"]) for gid in key_order]

        fig = plt.figure(figsize=(13.6, 6.8), dpi=160)
        gs = fig.add_gridspec(2, 1, hspace=0.28)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(len(key_order)), outlier_fracs, color="tab:blue", alpha=0.85)
        ax1.set_xticks(range(len(key_order)))
        ax1.set_xticklabels(key_order, rotation=30, ha="right")
        ax1.set_ylabel("outlier frac (abs(z)>3)")
        ax1.set_ylim(0, max([v for v in outlier_fracs if math.isfinite(v)] + [0.0]) * 1.15 + 1e-6)
        ax1.grid(axis="y", linestyle=":", alpha=0.4)
        ax1.set_title("Nuclear holdout audit (AME2020 all nuclei): outlier fraction by group")

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.bar(range(len(key_order)), med_absz, color="tab:orange", alpha=0.85)
        ax2.set_xticks(range(len(key_order)))
        ax2.set_xticklabels(key_order, rotation=30, ha="right")
        ax2.set_ylabel("median abs(z)")
        ax2.set_ylim(0, max([v for v in med_absz if math.isfinite(v)] + [0.0]) * 1.15 + 1e-6)
        ax2.grid(axis="y", linestyle=":", alpha=0.4)
        ax2.set_title("Nuclear holdout audit: residual severity by group")

        fig.suptitle(
            f"log10(B_pred/B_obs) with robust z (global median={global_med:.4f}, sigma={robust_sigma:.4f}); nz_ratio p05={nz_p05:.3f}, p95={nz_p95:.3f}",
            y=0.98,
            fontsize=10,
        )
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # Plot is optional; keep audit usable without matplotlib.
        pass

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": 7,
        "step": "7.20.4",
        "inputs": {
            "minimal_additional_physics_csv": {"path": _rel(in_csv), "sha256": _sha256(in_csv)},
            "beta2_extracted_json": {"path": _rel(beta2_json), "sha256": _sha256(beta2_json)},
        },
        "model": {
            "b_pred_field": "B_pred_local_spacing_sat_MeV",
            "log_ratio_definition": "log10(B_pred/B_obs)",
            "robust_baseline": {
                "median_log10_ratio": global_med,
                "mad_log10_ratio": float(mad),
                "robust_sigma_log10_ratio": robust_sigma,
                "z_definition": "(log10_ratio - median) / (1.4826*MAD)",
            },
            "outlier_rule": {"abs_z_gt": 3.0},
        },
        "thresholds": {"near_magic_width": 2, "edge_width": 2, "nz_ratio_p05": nz_p05, "nz_ratio_p95": nz_p95},
        "groups": groups,
        "outputs": {
            "summary_json": _rel(out_summary),
            "groups_csv": _rel(out_groups_csv),
            "outliers_csv": _rel(out_outliers_csv),
            "png": _rel(out_png) if out_png.exists() else None,
        },
        "notes": [
            "This audit is an operational holdout/segmentation check: it catalogs where residuals concentrate (magic/edges/extreme N/Z/deformed).",
            "It does not fit parameters; it uses the frozen minimal-additional-physics mapping output as-is.",
        ],
    }
    out_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "quantum",
            "action": "nuclear_holdout_audit",
            "outputs": [out_summary, out_groups_csv, out_outliers_csv, out_png if out_png.exists() else None],
            "params": {"near_magic_width": 2, "edge_width": 2, "nz_ratio_quantiles": [5, 95], "deformed_beta2_abs_min": 0.20},
            "result": {"n_nuclei": int(len(rows)), "groups_n": int(len(groups))},
        }
    )

    print("[ok] wrote:")
    print(f"- {out_summary}")
    print(f"- {out_groups_csv}")
    print(f"- {out_outliers_csv}")
    # 条件分岐: `out_png.exists()` を満たす経路を評価する。
    if out_png.exists():
        print(f"- {out_png}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

