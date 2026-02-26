#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_peak_summary.py

Step 16.4（BAO一次情報：銀河+random）:
`cosmology_bao_xi_from_catalogs.py` の出力（*_metrics.json）から、
BAOピーク位置（粗い推定：s^2 xi0 の broadband 除去→残差ピーク）を集計し、
距離写像ごとの z 依存（幾何の整合チェック）を 1 枚にまとめる。

目的（Phase A: スクリーニング）:
- 圧縮BAO出力（D_M/r_d 等）に依存せず、一次統計から得たピーク位置の
  (i) 絶対スケール（r_d free なら基準は不要）と
  (ii) redshift 依存（同一写像でピークが z に対して安定か）
  を確認する。

出力（固定）:
- output/private/cosmology/cosmology_bao_catalog_peak_summary.png
- output/private/cosmology/cosmology_bao_catalog_peak_summary_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.cosmology.cosmology_bao_xi_from_catalogs import _estimate_bao_feature_s2_xi  # noqa: E402

_WIN_ABS_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_WSL_ABS_RE = re.compile(r"^/mnt/([a-zA-Z])/(.+)$")


# 関数: `_resolve_path_like` の入出力契約と処理意図を定義する。
def _resolve_path_like(p: Any) -> Optional[Path]:
    # 条件分岐: `p is None` を満たす経路を評価する。
    if p is None:
        return None

    s = str(p).strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    # 条件分岐: `os.name == "nt"` を満たす経路を評価する。

    if os.name == "nt":
        m = _WSL_ABS_RE.match(s)
        # 条件分岐: `m` を満たす経路を評価する。
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
    else:
        # 条件分岐: `_WIN_ABS_RE.match(s)` を満たす経路を評価する。
        if _WIN_ABS_RE.match(s):
            drive = s[0].lower()
            rest = s[2:].lstrip("\\/").replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")

    path = Path(s)
    # 条件分岐: `path.is_absolute()` を満たす経路を評価する。
    if path.is_absolute():
        return path

    return _ROOT / path


# クラス: `PeakPoint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class PeakPoint:
    sample: str
    caps: str
    dist: str
    z_eff: float
    s_peak_xi0: float
    s_feature_xi2: Optional[float]
    residual_feature_xi2: Optional[float]
    out_tag: Optional[str]
    z_bin: str
    z_min: Optional[float]
    z_max: Optional[float]
    source_json: str


# 関数: `_load_points` の入出力契約と処理意図を定義する。

def _load_points(paths: Iterable[Path]) -> List[PeakPoint]:
    out: List[PeakPoint] = []
    for p in paths:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        params = d.get("params", {})
        derived = d.get("derived", {})
        bao_peak = derived.get("bao_peak", {}) if isinstance(derived, dict) else {}
        bao_feature_xi2 = derived.get("bao_feature_xi2", {}) if isinstance(derived, dict) else {}
        # 条件分岐: `not isinstance(params, dict) or not isinstance(derived, dict) or not isinstan...` を満たす経路を評価する。
        if not isinstance(params, dict) or not isinstance(derived, dict) or not isinstance(bao_peak, dict):
            continue

        try:
            sample = str(params.get("sample"))
            caps = str(params.get("caps"))
            dist = str(params.get("distance_model"))
            z_eff = float(derived.get("z_eff_gal_weighted"))
            s_peak_xi0 = float(bao_peak.get("s_peak"))
            tag_in = params.get("out_tag", None)
            out_tag: Optional[str]
            # 条件分岐: `tag_in is None` を満たす経路を評価する。
            if tag_in is None:
                out_tag = None
            else:
                tag_s = str(tag_in).strip()
                out_tag = tag_s if tag_s and tag_s.lower() != "null" else None

            z_cut = params.get("z_cut", {}) if isinstance(params.get("z_cut", {}), dict) else {}
            z_bin = str(z_cut.get("bin", "none"))
            z_min = z_cut.get("z_min", None)
            z_max = z_cut.get("z_max", None)
            z_min_f = float(z_min) if z_min is not None else None
            z_max_f = float(z_max) if z_max is not None else None
        except Exception:
            continue

        # Optional: xi2 feature (dominant |residual| in s^2 xi2 after broadband removal).

        s_feature_xi2: Optional[float] = None
        residual_feature_xi2: Optional[float] = None
        try:
            # 条件分岐: `isinstance(bao_feature_xi2, dict) and ("s_abs" in bao_feature_xi2)` を満たす経路を評価する。
            if isinstance(bao_feature_xi2, dict) and ("s_abs" in bao_feature_xi2):
                s_feature_xi2 = float(bao_feature_xi2.get("s_abs"))
                residual_feature_xi2 = float(bao_feature_xi2.get("residual_abs"))
            else:
                # Fallback: compute from the saved NPZ (cheap; avoids rerunning Corrfunc).
                npz_path = None
                outputs = d.get("outputs", {}) if isinstance(d.get("outputs", {}), dict) else {}
                # 条件分岐: `isinstance(outputs, dict)` を満たす経路を評価する。
                if isinstance(outputs, dict):
                    npz_path = outputs.get("npz", None)

                # 条件分岐: `not npz_path` を満たす経路を評価する。

                if not npz_path:
                    # Best-effort: replace suffix in metrics filename.
                    base = str(p)
                    # 条件分岐: `base.endswith("_metrics.json")` を満たす経路を評価する。
                    if base.endswith("_metrics.json"):
                        npz_path = base[: -len("_metrics.json")] + ".npz"

                npz_path_resolved = _resolve_path_like(npz_path)
                # 条件分岐: `npz_path_resolved and npz_path_resolved.exists()` を満たす経路を評価する。
                if npz_path_resolved and npz_path_resolved.exists():
                    arr = np.load(str(npz_path_resolved))
                    s_arr = np.asarray(arr["s"], dtype=float)
                    xi2_arr = np.asarray(arr["xi2"], dtype=float)
                    bf = _estimate_bao_feature_s2_xi(s=s_arr, xi=xi2_arr)
                    s_feature_xi2 = float(bf.get("s_abs"))
                    residual_feature_xi2 = float(bf.get("residual_abs"))
        except Exception:
            s_feature_xi2 = None
            residual_feature_xi2 = None

        out.append(
            PeakPoint(
                sample=sample,
                caps=caps,
                dist=dist,
                z_eff=z_eff,
                s_peak_xi0=s_peak_xi0,
                s_feature_xi2=s_feature_xi2,
                residual_feature_xi2=residual_feature_xi2,
                out_tag=out_tag,
                z_bin=z_bin,
                z_min=z_min_f,
                z_max=z_max_f,
                source_json=str(p),
            )
        )

    return out


# 関数: `_group_points` の入出力契約と処理意図を定義する。

def _group_points(points: List[PeakPoint]) -> Dict[Tuple[str, str, str], List[PeakPoint]]:
    g: Dict[Tuple[str, str, str], List[PeakPoint]] = {}
    for pt in points:
        key = (pt.sample, pt.caps, pt.dist)
        g.setdefault(key, []).append(pt)

    for k in list(g):
        g[k] = sorted(g[k], key=lambda x: x.z_eff)

    return g


# 関数: `_drift_stats` の入出力契約と処理意図を定義する。

def _drift_stats(values: List[float]) -> Dict[str, float]:
    s = np.asarray(list(values), dtype=np.float64)
    # 条件分岐: `s.size == 0` を満たす経路を評価する。
    if s.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "span": float("nan")}

    return {
        "n": int(s.size),
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=0)),
        "span": float(np.max(s) - np.min(s)),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize BAO peak positions from catalog-based xi outputs.")
    ap.add_argument("--glob", default="output/private/cosmology/cosmology_bao_xi_from_catalogs_*_metrics.json", help="metrics glob")
    ap.add_argument("--sample", default="", help="filter by sample (e.g., cmasslowztot); empty => all")
    ap.add_argument("--caps", default="", help="filter by caps (north/south/combined); empty => all")
    ap.add_argument(
        "--out-tag",
        default="none",
        help="Filter out_tag in inputs: none (default; only out_tag=null), any, or exact string",
    )
    ap.add_argument(
        "--require-zbin",
        action="store_true",
        help="only include metrics with z_bin != none (default: include all)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    paths = sorted(_ROOT.glob(str(args.glob)))
    pts = _load_points(paths)

    sample_f = str(args.sample).strip().lower()
    # 条件分岐: `sample_f` を満たす経路を評価する。
    if sample_f:
        pts = [p for p in pts if p.sample.lower() == sample_f]

    caps_f = str(args.caps).strip().lower()
    # 条件分岐: `caps_f` を満たす経路を評価する。
    if caps_f:
        pts = [p for p in pts if p.caps.lower() == caps_f]

    out_tag_f = str(args.out_tag).strip()
    # 条件分岐: `out_tag_f == "none"` を満たす経路を評価する。
    if out_tag_f == "none":
        pts = [p for p in pts if p.out_tag is None]
    # 条件分岐: 前段条件が不成立で、`out_tag_f == "any"` を追加評価する。
    elif out_tag_f == "any":
        pass
    else:
        pts = [p for p in pts if (p.out_tag or "") == out_tag_f]

    # 条件分岐: `bool(args.require_zbin)` を満たす経路を評価する。

    if bool(args.require_zbin):
        pts = [p for p in pts if p.z_bin != "none"]

    groups = _group_points(pts)

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the default output names stable for run_all (which may pass sample/caps filters).
    # Only add a suffix when out_tag is explicitly non-default to avoid clobbering baseline outputs.
    suffix = ""
    # 条件分岐: `out_tag_f == "any"` を満たす経路を評価する。
    if out_tag_f == "any":
        suffix = "__outtag_any"
    # 条件分岐: 前段条件が不成立で、`out_tag_f not in ("none", "any") and out_tag_f` を追加評価する。
    elif out_tag_f not in ("none", "any") and out_tag_f:
        suffix = f"__{out_tag_f}"

    out_png = out_dir / f"cosmology_bao_catalog_peak_summary{suffix}.png"
    out_json = out_dir / f"cosmology_bao_catalog_peak_summary{suffix}_metrics.json"

    # Plot
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.8))
    markers = {"lcdm": "o", "pbg": "s"}
    colors = {"lcdm": "#1f77b4", "pbg": "#ff7f0e"}

    for (sample, caps, dist), xs in sorted(groups.items()):
        z = [p.z_eff for p in xs]
        s_peak = [p.s_peak_xi0 for p in xs]
        label = f"{sample}/{caps}/{dist}"
        ax1.plot(
            z,
            s_peak,
            marker=markers.get(dist, "o"),
            color=colors.get(dist, "#333333"),
            linewidth=1.8,
            markersize=6,
            label=label,
        )

        # xi2 feature (optional)
        xs2 = [p for p in xs if (p.s_feature_xi2 is not None and p.residual_feature_xi2 is not None)]
        # 条件分岐: `xs2` を満たす経路を評価する。
        if xs2:
            z2 = [p.z_eff for p in xs2]
            s2 = [float(p.s_feature_xi2) for p in xs2]
            ax2.plot(z2, s2, color=colors.get(dist, "#333333"), linewidth=1.2, alpha=0.65)
            for p2 in xs2:
                mk = "^" if float(p2.residual_feature_xi2) >= 0.0 else "v"
                ax2.scatter(
                    [p2.z_eff],
                    [float(p2.s_feature_xi2)],
                    marker=mk,
                    s=55,
                    color=colors.get(dist, "#333333"),
                    edgecolor="#222222",
                    linewidths=0.5,
                )

    ax1.set_xlabel("z_eff (galaxy-weighted)")
    ax1.set_ylabel("s_peak [Mpc/h]  (from s² ξ0 residual peak)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=9, loc="best")

    ax2.set_xlabel("z_eff (galaxy-weighted)")
    ax2.set_ylabel("s_feature [Mpc/h]  (dominant |residual| in s² ξ2)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.set_title("ξ2 feature (marker: ^ positive / v negative)")

    title_suffix = ""
    # 条件分岐: `sample_f` を満たす経路を評価する。
    if sample_f:
        title_suffix += f" sample={sample_f}"

    # 条件分岐: `caps_f` を満たす経路を評価する。

    if caps_f:
        title_suffix += f" caps={caps_f}"

    # 条件分岐: `str(args.out_tag) != "none"` を満たす経路を評価する。

    if str(args.out_tag) != "none":
        title_suffix += f" out_tag={out_tag_f or 'none'}"

    # 条件分岐: `args.require_zbin` を満たす経路を評価する。

    if args.require_zbin:
        title_suffix += " zbin_only"

    fig.suptitle(f"BAO peak summary (catalog-based ξℓ){title_suffix}", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO catalog-based peak summary)",
        "inputs": {"glob": str(args.glob), "n_files_scanned": int(len(paths))},
        "filters": {
            "sample": sample_f or None,
            "caps": caps_f or None,
            "out_tag": out_tag_f,
            "require_zbin": bool(args.require_zbin),
        },
        "groups": {},
        "outputs": {"png": str(out_png)},
    }

    for (sample, caps, dist), xs in sorted(groups.items()):
        payload["groups"][f"{sample}/{caps}/{dist}"] = {
            "points": [
                {
                    "z_eff": p.z_eff,
                    "s_peak_xi0": p.s_peak_xi0,
                    "s_feature_xi2": p.s_feature_xi2,
                    "residual_feature_xi2": p.residual_feature_xi2,
                    "out_tag": p.out_tag,
                    "z_bin": p.z_bin,
                    "z_min": p.z_min,
                    "z_max": p.z_max,
                    "source_json": p.source_json,
                }
                for p in xs
            ],
            "drift": {
                "xi0_peak": _drift_stats([p.s_peak_xi0 for p in xs]),
                "xi2_feature": _drift_stats([float(p.s_feature_xi2) for p in xs if p.s_feature_xi2 is not None]),
            },
        }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_peak_summary",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": payload.get("filters", {}),
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
