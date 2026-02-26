#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_sampling_convergence.py

Step 16.4（BAO一次情報：銀河+random） / Phase A（スクリーニング）:
random/galaxy の部分抽出（prefix_rows / reservoir）に対して、ξℓ由来の指標が
どの程度で収束するか（安定するか）を確認する。

狙い：
- prefix_rows は行順の偏り（セクタ欠落）で LS 推定が崩れ得るため、
  行数を振って「指標が収束するか」をまず確認する（Phase A の健全性チェック）。
- 収束が悪い場合は reservoir（均一サンプル）へ移行する判断材料にする。

出力（固定）:
- output/private/cosmology/cosmology_bao_sampling_convergence.png
- output/private/cosmology/cosmology_bao_sampling_convergence_metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


@dataclass(frozen=True)
class RunResult:
    random_max_rows: int
    out_tag: str
    metrics_json: Path
    metrics: Dict[str, Any]


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for tok in str(s).split(","):
        t = tok.strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            continue

        out.append(int(float(t)))

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise ValueError("empty list")

    return out


def _run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_path(*, sample: str, caps: str, dist: str, ztag: str, out_tag: str) -> Path:
    base = f"cosmology_bao_xi_from_catalogs_{sample}_{caps}_{dist}"
    # 条件分岐: `ztag` を満たす経路を評価する。
    if ztag:
        base = f"{base}_{ztag}"

    # 条件分岐: `out_tag` を満たす経路を評価する。

    if out_tag:
        base = f"{base}__{out_tag}"

    return _ROOT / "output" / "private" / "cosmology" / f"{base}_metrics.json"


def _extract_key_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    derived = d.get("derived", {}) if isinstance(d.get("derived", {}), dict) else {}
    bao_peak = derived.get("bao_peak", {}) if isinstance(derived.get("bao_peak", {}), dict) else {}
    bao_xi2 = derived.get("bao_feature_xi2", {}) if isinstance(derived.get("bao_feature_xi2", {}), dict) else {}
    wedges = derived.get("bao_wedges", {}) if isinstance(derived.get("bao_wedges", {}), dict) else {}
    sectors = d.get("sectors", {}) if isinstance(d.get("sectors", {}), dict) else {}
    by_cap = sectors.get("by_cap", []) if isinstance(sectors.get("by_cap", []), list) else []

    kept = {}
    for item in by_cap:
        # 条件分岐: `not isinstance(item, dict)` を満たす経路を評価する。
        if not isinstance(item, dict):
            continue

        cap = str(item.get("cap", ""))
        # 条件分岐: `not cap` を満たす経路を評価する。
        if not cap:
            continue

        kept[cap] = {
            "kept_frac_gal": item.get("kept_frac_gal"),
            "kept_frac_rnd": item.get("kept_frac_rnd"),
            "n_sectors_common": item.get("n_sectors_common"),
        }

    out: Dict[str, Any] = {
        "z_eff": derived.get("z_eff_gal_weighted"),
        "n_gal": d.get("sizes", {}).get("n_gal") if isinstance(d.get("sizes", {}), dict) else None,
        "n_rnd": d.get("sizes", {}).get("n_rnd") if isinstance(d.get("sizes", {}), dict) else None,
        "xi0_s_peak": bao_peak.get("s_peak"),
        "xi2_s_abs": bao_xi2.get("s_abs"),
        "xi2_residual_abs": bao_xi2.get("residual_abs"),
        "wedge_enabled": wedges.get("enabled", False),
        "wedge_delta_s": wedges.get("delta_s_peak"),
        "wedge_ratio": wedges.get("ratio_s_peak"),
        "kept_by_cap": kept,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convergence test for catalog-based BAO xi metrics under subsampling.")
    ap.add_argument("--sample", default="cmass", help="cmass/lowz/cmasslowz/cmasslowztot (default: cmass)")
    ap.add_argument("--caps", default="combined", help="combined/north/south (default: combined)")
    ap.add_argument("--dist", default="lcdm", help="lcdm or pbg (default: lcdm)")
    ap.add_argument("--random-kind", default="random1", help="random0 or random1 (default: random1)")
    ap.add_argument("--random-sampling", default="prefix_rows", choices=["prefix_rows", "reservoir"])
    ap.add_argument("--random-max-rows-list", default="200000,500000", help="comma list (default: 200000,500000)")
    ap.add_argument("--galaxy-sampling", default="prefix_rows", choices=["prefix_rows", "reservoir", "download_full"])
    ap.add_argument("--galaxy-max-rows", type=int, default=200000, help="for prefix/reservoir (default: 200000)")
    ap.add_argument("--sampling-seed", type=int, default=0, help="reservoir seed (default: 0)")
    ap.add_argument("--random-scan-max-rows", type=int, default=0, help="reservoir scan cap (0=all)")
    ap.add_argument("--chunk-rows", type=int, default=200000, help="reservoir chunk rows (default: 200000)")
    ap.add_argument("--z-bin", default="none", choices=["none", "b1", "b2", "b3"])
    ap.add_argument("--nmu", type=int, default=120, help="mu bins forwarded to xi script (default: 120)")
    ap.add_argument("--mu-max", type=float, default=1.0, help="mu max forwarded to xi script (default: 1.0)")
    ap.add_argument("--mu-split", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--run", action="store_true", help="actually run fetch + xi (requires Corrfunc under WSL/Linux)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    sample = str(args.sample).strip().lower()
    caps = str(args.caps).strip().lower()
    dist = str(args.dist).strip().lower()
    random_kind = str(args.random_kind).strip()
    random_sampling = str(args.random_sampling).strip().lower()
    galaxy_sampling = str(args.galaxy_sampling).strip().lower()
    galaxy_max_rows = int(args.galaxy_max_rows)
    z_bin = str(args.z_bin).strip().lower()

    rows_list = _parse_int_list(str(args.random_max_rows_list))

    # Tag used in output filenames of xi script.
    ztag = z_bin if z_bin != "none" else ""
    run_results: List[RunResult] = []

    manifest_path = _ROOT / "data" / "cosmology" / "boss_dr12v5_lss" / "manifest.json"
    manifest_backup = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""

    try:
        for random_max_rows in rows_list:
            out_tag = f"conv_rnd{random_sampling}_{random_max_rows}_gal{galaxy_sampling}_{galaxy_max_rows}"

            # 条件分岐: `args.run` を満たす経路を評価する。
            if args.run:
                # 1) fetch (updates manifest to point to the desired extracted npz variants)
                fetch_cmd = [
                    sys.executable,
                    str(_ROOT / "scripts" / "cosmology" / "fetch_boss_dr12v5_lss.py"),
                    "--samples",
                    sample,
                    "--caps",
                    "north,south" if caps == "combined" else caps,
                    "--random-kind",
                    random_kind,
                    "--random-sampling",
                    random_sampling,
                    "--random-max-rows",
                    str(int(random_max_rows)),
                    "--galaxy-sampling",
                    galaxy_sampling,
                    "--galaxy-max-rows",
                    str(int(galaxy_max_rows)),
                    "--sampling-seed",
                    str(int(args.sampling_seed)),
                    "--chunk-rows",
                    str(int(args.chunk_rows)),
                ]
                # 条件分岐: `int(args.random_scan_max_rows) > 0` を満たす経路を評価する。
                if int(args.random_scan_max_rows) > 0:
                    fetch_cmd += ["--random-scan-max-rows", str(int(args.random_scan_max_rows))]

                _run_cmd(fetch_cmd)

                # 2) xi (writes output/private/cosmology/cosmology_bao_xi_from_catalogs_...__out_tag_*.json)
                xi_cmd = [
                    sys.executable,
                    str(_ROOT / "scripts" / "cosmology" / "cosmology_bao_xi_from_catalogs.py"),
                    "--sample",
                    sample,
                    "--caps",
                    caps,
                    "--dist",
                    dist,
                    "--random-kind",
                    random_kind,
                    "--z-bin",
                    z_bin,
                    "--nmu",
                    str(int(args.nmu)),
                    "--mu-max",
                    str(float(args.mu_max)),
                    "--mu-split",
                    str(float(args.mu_split)),
                    "--threads",
                    str(int(args.threads)),
                    "--out-tag",
                    out_tag,
                ]
                _run_cmd(xi_cmd)

            mpath = _metrics_path(sample=sample, caps=caps, dist=dist, ztag=ztag, out_tag=out_tag)
            # 条件分岐: `not mpath.exists()` を満たす経路を評価する。
            if not mpath.exists():
                print(f"[warn] metrics not found (skip): {mpath}")
                continue

            d = _read_json(mpath)
            run_results.append(RunResult(random_max_rows=int(random_max_rows), out_tag=out_tag, metrics_json=mpath, metrics=d))
    finally:
        # Restore manifest to avoid surprising side effects after convergence runs.
        if manifest_backup and manifest_path.exists():
            manifest_path.write_text(manifest_backup, encoding="utf-8")

    # 条件分岐: `not run_results` を満たす経路を評価する。

    if not run_results:
        raise SystemExit("no metrics collected (did you run with --run or provide existing outputs?)")

    # Summarize

    rows = []
    for r in run_results:
        rows.append(
            {
                "random_max_rows": r.random_max_rows,
                "out_tag": r.out_tag,
                "metrics_json": str(r.metrics_json),
                **_extract_key_metrics(r.metrics),
            }
        )

    rows = sorted(rows, key=lambda x: int(x["random_max_rows"]))

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cosmology_bao_sampling_convergence.png"
    out_json = out_dir / "cosmology_bao_sampling_convergence_metrics.json"

    # Plot
    import matplotlib.pyplot as plt

    x = np.asarray([int(r["random_max_rows"]) for r in rows], dtype=float)
    y_s0 = np.asarray([float(r["xi0_s_peak"]) for r in rows], dtype=float)
    y_ds = np.asarray([float(r["wedge_delta_s"]) for r in rows], dtype=float)
    y_ratio = np.asarray([float(r["wedge_ratio"]) for r in rows], dtype=float)

    def _cap_series(key: str) -> np.ndarray:
        vals = []
        for rr in rows:
            kept = rr.get("kept_by_cap", {})
            # 条件分岐: `not isinstance(kept, dict)` を満たす経路を評価する。
            if not isinstance(kept, dict):
                vals.append(float("nan"))
                continue

            v = kept.get(key, {}).get("kept_frac_rnd") if isinstance(kept.get(key, {}), dict) else None
            vals.append(float(v) if v is not None else float("nan"))

        return np.asarray(vals, dtype=float)

    kept_n = _cap_series("north")
    kept_s = _cap_series("south")

    fig, axs = plt.subplots(2, 2, figsize=(15.0, 7.8))
    ax = axs[0, 0]
    ax.plot(x, y_s0, marker="o", linewidth=2.0)
    ax.set_title("ξ0 peak (s_peak)")
    ax.set_ylabel("s_peak [Mpc/h]")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axs[0, 1]
    ax.plot(x, y_ds, marker="o", linewidth=2.0, color="#9467bd")
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title("wedge anisotropy (Δs = s∥-s⊥)")
    ax.set_ylabel("Δs [Mpc/h]")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axs[1, 0]
    ax.plot(x, y_ratio, marker="o", linewidth=2.0, color="#2ca02c")
    ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title("wedge anisotropy (ratio = s∥/s⊥)")
    ax.set_xlabel("random max rows")
    ax.set_ylabel("ratio")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axs[1, 1]
    ax.plot(x, kept_n, marker="o", linewidth=2.0, label="north kept_frac_rnd")
    ax.plot(x, kept_s, marker="s", linewidth=2.0, label="south kept_frac_rnd")
    ax.set_title("sector matching impact (kept_frac_rnd)")
    ax.set_xlabel("random max rows")
    ax.set_ylabel("kept_frac_rnd")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="best")

    for ax in axs.reshape(-1):
        ax.set_xscale("log")

    fig.suptitle(f"BAO sampling convergence: sample={sample}, caps={caps}, dist={dist}, random={random_sampling}/{random_kind}", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO sampling convergence)",
        "params": {
            "sample": sample,
            "caps": caps,
            "dist": dist,
            "random_kind": random_kind,
            "random_sampling": random_sampling,
            "random_max_rows_list": rows_list,
            "galaxy_sampling": galaxy_sampling,
            "galaxy_max_rows": galaxy_max_rows,
            "z_bin": z_bin,
            "mu_split": float(args.mu_split),
        },
        "results": rows,
        "outputs": {"png": str(out_png)},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_sampling_convergence",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": payload.get("params", {}),
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
