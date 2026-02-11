#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_xi_ngc_sgc_split_summary.py

Phase 4 / Step 4.5（BAO一次統計から再構築）:
NGC/SGC（north/south）を分割したときに、BAOの特徴量がどの程度ずれるかを
catalog-based ξℓ（ℓ=0,2）の出力（*_metrics.json）から要約する。

目的：
- NGC/SGC split を「系統の切り分け」として数値化し、後続の議論で
  “理論差” と “座標化/観測系の差” が混ざるのを防ぐ。

入力：
- `scripts/cosmology/cosmology_bao_xi_from_catalogs.py` の *_metrics.json（north/south/combined）

出力（固定）:
- output/private/cosmology/cosmology_bao_xi_ngc_sgc_split_summary.png
- output/private/cosmology/cosmology_bao_xi_ngc_sgc_split_summary_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_WIN_ABS_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_WSL_ABS_RE = re.compile(r"^/mnt/([a-zA-Z])/(.+)$")


def _resolve_path_like(p: Any) -> Optional[Path]:
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return None
    if os.name == "nt":
        m = _WSL_ABS_RE.match(s)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
    else:
        if _WIN_ABS_RE.match(s):
            drive = s[0].lower()
            rest = s[2:].lstrip("\\/").replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
    path = Path(s)
    return path if path.is_absolute() else (_ROOT / path)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass(frozen=True)
class Point:
    zbin: str
    caps: str
    z_eff: float
    s_peak_xi0: float
    s_feature_xi2: float
    y_feature_xi2: float
    source_metrics: str
    source_npz: Optional[str]


def _metrics_path(*, sample: str, caps: str, dist: str, zbin: str, suffix: str) -> Path:
    suffix = str(suffix)
    if suffix and not suffix.startswith("__"):
        suffix = "__" + suffix
    name = f"cosmology_bao_xi_from_catalogs_{sample}_{caps}_{dist}_{zbin}{suffix}_metrics.json"
    return _ROOT / "output" / "private" / "cosmology" / name


def _load_point(path: Path, *, zbin: str, caps: str) -> Point:
    d = _load_json(path)
    z_eff = float(_pick(d, "derived", "z_eff_gal_weighted"))
    s_peak = float(_pick(d, "derived", "bao_peak", "s_peak"))
    s_abs = float(_pick(d, "derived", "bao_feature_xi2", "s_abs"))
    y_abs = float(_pick(d, "derived", "bao_feature_xi2", "y_abs"))
    npz_like = _pick(d, "outputs", "npz", default=None)
    return Point(
        zbin=str(zbin),
        caps=str(caps),
        z_eff=z_eff,
        s_peak_xi0=s_peak,
        s_feature_xi2=s_abs,
        y_feature_xi2=y_abs,
        source_metrics=str(path),
        source_npz=str(npz_like) if npz_like is not None else None,
    )


def _caps_label(caps: str) -> str:
    caps = str(caps)
    if caps == "north":
        return "NGC"
    if caps == "south":
        return "SGC"
    return caps


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize NGC/SGC split for catalog-based BAO xi multipoles.")
    ap.add_argument("--sample", type=str, default="cmasslowztot")
    ap.add_argument("--dist", type=str, default="lcdm")
    ap.add_argument("--zbins", type=str, default="b1,b2,b3", help="comma-separated zbin labels (e.g., b1,b2,b3)")
    ap.add_argument(
        "--suffix",
        type=str,
        default="__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757",
        help="metrics suffix (same string used after zbin in *_metrics.json; leading '__' optional)",
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_xi_ngc_sgc_split_summary.png"),
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_xi_ngc_sgc_split_summary_metrics.json"),
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    zbins = [s.strip() for s in str(args.zbins).split(",") if s.strip()]
    if not zbins:
        raise SystemExit("--zbins must be non-empty")

    out_png = Path(args.out_png)
    out_json = Path(args.out_json)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    points: List[Point] = []
    caps_list = ["combined", "north", "south"]
    for zbin in zbins:
        for caps in caps_list:
            p = _metrics_path(sample=str(args.sample), caps=caps, dist=str(args.dist), zbin=zbin, suffix=str(args.suffix))
            if not p.exists():
                raise SystemExit(f"missing metrics: {p}")
            points.append(_load_point(p, zbin=zbin, caps=caps))

    # Compute per-zbin NGC-SGC deltas (systematic indicator).
    deltas: List[Dict[str, Any]] = []
    curve_rmse: List[Dict[str, Any]] = []
    for zbin in zbins:
        ngc = next((pt for pt in points if pt.zbin == zbin and pt.caps == "north"), None)
        sgc = next((pt for pt in points if pt.zbin == zbin and pt.caps == "south"), None)
        if ngc is None or sgc is None:
            continue
        deltas.append(
            {
                "zbin": zbin,
                "z_eff_ngc": ngc.z_eff,
                "z_eff_sgc": sgc.z_eff,
                "delta_s_peak_xi0_mpc_over_h": float(ngc.s_peak_xi0 - sgc.s_peak_xi0),
                "delta_s_feature_xi2_mpc_over_h": float(ngc.s_feature_xi2 - sgc.s_feature_xi2),
                "delta_y_feature_xi2": float(ngc.y_feature_xi2 - sgc.y_feature_xi2),
            }
        )

        # RMSE between NGC and SGC curves (s^2 xi0 / s^2 xi2).
        try:
            p_ngc = _resolve_path_like(ngc.source_npz)
            p_sgc = _resolve_path_like(sgc.source_npz)
            if p_ngc and p_sgc and p_ngc.exists() and p_sgc.exists():
                a = np.load(str(p_ngc))
                b = np.load(str(p_sgc))
                s_a = np.asarray(a["s"], dtype=float)
                s_b = np.asarray(b["s"], dtype=float)
                if s_a.shape == s_b.shape and np.allclose(s_a, s_b, rtol=0.0, atol=1e-12):
                    s2 = s_a * s_a
                    s2xi0_a = s2 * np.asarray(a["xi0"], dtype=float)
                    s2xi0_b = s2 * np.asarray(b["xi0"], dtype=float)
                    s2xi2_a = s2 * np.asarray(a["xi2"], dtype=float)
                    s2xi2_b = s2 * np.asarray(b["xi2"], dtype=float)
                    rmse0 = float(np.sqrt(np.mean((s2xi0_a - s2xi0_b) ** 2)))
                    rmse2 = float(np.sqrt(np.mean((s2xi2_a - s2xi2_b) ** 2)))
                    curve_rmse.append(
                        {
                            "zbin": zbin,
                            "rmse_s2_xi0_ngc_vs_sgc": rmse0,
                            "rmse_s2_xi2_ngc_vs_sgc": rmse2,
                        }
                    )
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "sample": str(args.sample),
            "dist": str(args.dist),
            "zbins": zbins,
            "suffix": str(args.suffix),
        },
        "points": [asdict(p) for p in points],
        "ngc_minus_sgc_by_zbin": deltas,
        "ngc_vs_sgc_curve_rmse_by_zbin": curve_rmse,
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "json": str(out_json.relative_to(_ROOT)).replace("\\", "/"),
        },
        "notes": [
            "s_peak_xi0 は s^2 ξ0 の broad-band 除去後の残差ピーク（粗い推定）。",
            "s_feature_xi2 は s^2 ξ2 の broad-band 除去後、|残差|が最大となる特徴位置（粗い推定）。",
            "曲線RMSEは、同一sビン上で s^2 ξℓ のRMSE（NGC vs SGC）を計算したもの。",
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot (2 panels): xi0 peak and xi2 feature location.
    _set_japanese_font()
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax0, ax1 = axes

    color_map = {"combined": "#333333", "north": "#1f77b4", "south": "#ff7f0e"}
    marker_map = {"combined": "o", "north": "s", "south": "D"}

    for caps in caps_list:
        pts = [p for p in points if p.caps == caps]
        pts = sorted(pts, key=lambda p: p.z_eff)
        x = np.array([p.z_eff for p in pts], dtype=float)
        y0 = np.array([p.s_peak_xi0 for p in pts], dtype=float)
        y2 = np.array([p.s_feature_xi2 for p in pts], dtype=float)
        ax0.plot(x, y0, marker=marker_map[caps], color=color_map[caps], label=_caps_label(caps))
        ax1.plot(x, y2, marker=marker_map[caps], color=color_map[caps], label=_caps_label(caps))
        for p in pts:
            ax0.text(p.z_eff, p.s_peak_xi0, f" {p.zbin}", fontsize=8, ha="left", va="center")
            ax1.text(p.z_eff, p.s_feature_xi2, f" {p.zbin}", fontsize=8, ha="left", va="center")

    ax0.set_title("ξ0: BAOピーク位置（粗い推定）")
    ax0.set_xlabel("z_eff (galaxy-weighted)")
    ax0.set_ylabel("s_peak [Mpc/h] (from s²ξ0 residual)")
    ax0.grid(True, alpha=0.3)

    ax1.set_title("ξ2: 特徴位置（|残差|最大; 粗い推定）")
    ax1.set_xlabel("z_eff (galaxy-weighted)")
    ax1.set_ylabel("s_feature [Mpc/h] (from s²ξ2 residual)")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(f"BOSS DR12v5 LSS: NGC/SGC split (sample={args.sample}, dist={args.dist})", fontsize=11)
    ax0.legend(fontsize=9)
    ax1.legend(fontsize=9)

    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_xi_ngc_sgc_split_summary",
                "argv": sys.argv,
                "inputs": {
                    "sample": str(args.sample),
                    "dist": str(args.dist),
                    "zbins": zbins,
                    "suffix": str(args.suffix),
                },
                "outputs": {"png": out_png, "json": out_json},
            }
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
