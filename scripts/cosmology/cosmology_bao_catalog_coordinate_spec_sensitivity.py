#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_coordinate_spec_sensitivity.py

Phase 4 / Step 4.5B.21.4.4.5.2:
DESI（等）の catalog-based ξℓ→peakfit（ε）について、座標化仕様の差
（z_source / comoving distance の数値積分設定 など）が結果に与える影響を可視化する。

狙い：
- ε が動いたときに「理論差」か「座標化差」かが混ざらないよう、A0仕様を固定した上で
  “1項目ずつ” 感度を確認する（議論のブレ防止）。
- Corrfunc は使わない（WSL不要）。`cosmology_bao_catalog_peakfit.py` の出力（metrics）を読むだけ。

出力（固定名）:
- output/private/cosmology/cosmology_bao_catalog_coordinate_spec_sensitivity__{sample}_{caps}__{base_out_tag}.png
- output/private/cosmology/cosmology_bao_catalog_coordinate_spec_sensitivity__{sample}_{caps}__{base_out_tag}_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


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
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float_token(x: float) -> str:
    t = f"{float(x):g}".replace(".", "p").replace("-", "m")
    return t


def _scenario_label(meta: Dict[str, Any], *, default_lcdm: Dict[str, Any]) -> str:
    z_source = str(meta.get("z_source") or "obs")
    parts: List[str] = []
    # 条件分岐: `z_source != "obs"` を満たす経路を評価する。
    if z_source != "obs":
        parts.append(f"z={z_source}")

    lcdm = meta.get("lcdm") if isinstance(meta.get("lcdm"), dict) else {}
    n_grid = lcdm.get("n_grid", default_lcdm.get("n_grid"))
    z_grid_max = lcdm.get("z_grid_max", default_lcdm.get("z_grid_max"))
    # 条件分岐: `(n_grid is not None) and int(n_grid) != int(default_lcdm.get("n_grid", n_grid))` を満たす経路を評価する。
    if (n_grid is not None) and int(n_grid) != int(default_lcdm.get("n_grid", n_grid)):
        parts.append(f"ng={int(n_grid)}")

    # 条件分岐: `(z_grid_max is not None) and float(z_grid_max) != float(default_lcdm.get("z_g...` を満たす経路を評価する。

    if (z_grid_max is not None) and float(z_grid_max) != float(default_lcdm.get("z_grid_max", z_grid_max)):
        parts.append(f"zg={_fmt_float_token(float(z_grid_max))}")

    return "A0" if not parts else " / ".join(parts)


def _zrange_key(z_min: float, z_max: float) -> str:
    def fmt(x: float) -> str:
        t = f"{float(x):.3f}".rstrip("0").rstrip(".")
        return t.replace(".", "p")

    return f"zmin{fmt(z_min)}_zmax{fmt(z_max)}"


@dataclass(frozen=True)
class Point:
    out_tag: str
    label: str
    dist: str
    z_min: float
    z_max: float
    z_eff: float
    eps: float
    sigma_eps: float
    abs_sigma: float
    status: str
    xi_metrics_path: Path
    xi_coord_hash: Optional[str]

    @property
    def zrange_key(self) -> str:
        return _zrange_key(self.z_min, self.z_max)


def _extract_points_from_peakfit_metrics(metrics_path: Path) -> Tuple[Dict[str, Any], List[Point]]:
    d = _load_json(metrics_path)
    inputs = d.get("inputs", {}) if isinstance(d.get("inputs"), dict) else {}
    out_tag = str(inputs.get("out_tag") or "")
    coord_common = inputs.get("coordinate_spec_common") if isinstance(inputs.get("coordinate_spec_common"), dict) else {}
    default_lcdm = coord_common.get("lcdm") if isinstance(coord_common.get("lcdm"), dict) else {}
    label = _scenario_label(coord_common, default_lcdm=default_lcdm)

    points: List[Point] = []
    for r in d.get("results", []) if isinstance(d.get("results"), list) else []:
        try:
            dist = str(r.get("dist"))
            z_eff = float(r.get("z_eff"))
            eps = float(((r.get("fit") or {}).get("free") or {}).get("eps"))
            screening = r.get("screening", {}) if isinstance(r.get("screening"), dict) else {}
            sigma_eps = float(screening.get("sigma_eps_1sigma"))
            abs_sigma = float(screening.get("abs_sigma"))
            status = str(screening.get("status") or "")
            xi_metrics_path = Path(str(((r.get("inputs") or {}).get("metrics_json")))).resolve()
            xi = _load_json(xi_metrics_path)
            z_cut = (xi.get("params", {}) or {}).get("z_cut", {}) or {}
            z_min = float(z_cut.get("z_min"))
            z_max = float(z_cut.get("z_max"))
            xi_coord_hash = (xi.get("params", {}) or {}).get("coordinate_spec_hash")
            points.append(
                Point(
                    out_tag=out_tag,
                    label=label,
                    dist=dist,
                    z_min=z_min,
                    z_max=z_max,
                    z_eff=z_eff,
                    eps=eps,
                    sigma_eps=sigma_eps,
                    abs_sigma=abs_sigma,
                    status=status,
                    xi_metrics_path=xi_metrics_path,
                    xi_coord_hash=str(xi_coord_hash) if xi_coord_hash is not None else None,
                )
            )
        except Exception:
            continue

    return d, points


def _pick(points: Sequence[Point], *, out_tag: str, dist: str, zrange_key: str) -> Optional[Point]:
    for p in points:
        # 条件分岐: `p.out_tag == out_tag and p.dist == dist and p.zrange_key == zrange_key` を満たす経路を評価する。
        if p.out_tag == out_tag and p.dist == dist and p.zrange_key == zrange_key:
            return p

    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BAO coordinateization-spec sensitivity (from peakfit metrics).")
    ap.add_argument("--sample", default="lrg", help="sample (default: lrg)")
    ap.add_argument("--caps", default="combined", help="caps (default: combined)")
    ap.add_argument(
        "--base-out-tag",
        default="w_desi_default_ms_off_y1bins",
        help="baseline out_tag (default: w_desi_default_ms_off_y1bins)",
    )
    ap.add_argument(
        "--variant-out-tags",
        default="",
        help="comma-separated variant out_tag list (optional). If omitted, uses a small DESI-oriented default list.",
    )
    ap.add_argument(
        "--out-png",
        default="",
        help="output png path (default: output/private/cosmology/cosmology_bao_catalog_coordinate_spec_sensitivity__{sample}_{caps}__{base_out_tag}.png)",
    )
    ap.add_argument(
        "--out-json",
        default="",
        help="output json path (default: output/private/cosmology/cosmology_bao_catalog_coordinate_spec_sensitivity__{sample}_{caps}__{base_out_tag}_metrics.json)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    sample = str(args.sample)
    caps = str(args.caps)
    base_tag = str(args.base_out_tag)

    # 条件分岐: `str(args.variant_out_tags).strip()` を満たす経路を評価する。
    if str(args.variant_out_tags).strip():
        variant_tags = [s.strip() for s in str(args.variant_out_tags).split(",") if s.strip()]
    else:
        variant_tags = [
            f"{base_tag}_zs_cmb",
            f"{base_tag}_ng2000",
            f"{base_tag}_ng20000",
            f"{base_tag}_zg1",
            f"{base_tag}_zg5",
        ]

    out_tags = [base_tag] + [t for t in variant_tags if t != base_tag]

    def peakfit_metrics_path(tag: str) -> Path:
        return (_ROOT / "output" / "private" / "cosmology" / f"cosmology_bao_catalog_peakfit_{sample}_{caps}__{tag}_metrics.json").resolve()

    p_metrics = {tag: peakfit_metrics_path(tag) for tag in out_tags}
    missing = [tag for tag, p in p_metrics.items() if not p.exists()]
    # 条件分岐: `not p_metrics[base_tag].exists()` を満たす経路を評価する。
    if not p_metrics[base_tag].exists():
        print(f"[skip] missing baseline peakfit metrics: {p_metrics[base_tag]}")
        return 0

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        print("[warn] missing variant peakfit metrics (skip those tags):")
        for tag in missing:
            # 条件分岐: `tag != base_tag` を満たす経路を評価する。
            if tag != base_tag:
                print(f"  - {tag}: {p_metrics[tag]}")

    all_points: List[Point] = []
    by_tag_inputs: Dict[str, Any] = {}
    for tag in out_tags:
        p = p_metrics[tag]
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            continue

        m, pts = _extract_points_from_peakfit_metrics(p)
        by_tag_inputs[tag] = {
            "metrics_json": str(p),
            "coordinate_spec_common": (m.get("inputs", {}) or {}).get("coordinate_spec_common"),
            "cov_source": ((m.get("inputs", {}) or {}).get("covariance") or {}).get("source"),
        }
        all_points.extend([Point(out_tag=tag, label=p0.label, dist=p0.dist, z_min=p0.z_min, z_max=p0.z_max, z_eff=p0.z_eff, eps=p0.eps, sigma_eps=p0.sigma_eps, abs_sigma=p0.abs_sigma, status=p0.status, xi_metrics_path=p0.xi_metrics_path, xi_coord_hash=p0.xi_coord_hash) for p0 in pts])

    # Determine z ranges present (prefer stable order by z_min).

    z_ranges = sorted({p.zrange_key for p in all_points}, key=lambda k: float(k.split("_")[0].replace("zmin", "").replace("p", ".")))
    # 条件分岐: `not z_ranges` を満たす経路を評価する。
    if not z_ranges:
        print("[skip] no points extracted")
        return 0

    dist_list = ["lcdm", "pbg"]
    tag_list = [t for t in out_tags if t in by_tag_inputs]

    # Build labels per tag (from coordinate_spec_common).
    coord_common_base = by_tag_inputs.get(base_tag, {}).get("coordinate_spec_common") or {}
    default_lcdm = coord_common_base.get("lcdm") if isinstance(coord_common_base.get("lcdm"), dict) else {}
    tag_labels = []
    for t in tag_list:
        coord = by_tag_inputs.get(t, {}).get("coordinate_spec_common") or {}
        tag_labels.append(_scenario_label(coord, default_lcdm=default_lcdm))

    _set_japanese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = {"z0": "#1f77b4", "z1": "#ff7f0e", "z2": "#2ca02c", "z3": "#d62728"}
    markers = {"z0": "o", "z1": "s", "z2": "D", "z3": "^"}

    for ax, dist in zip(axes, dist_list):
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        for i_z, zkey in enumerate(z_ranges):
            c = colors.get(f"z{i_z}", None) or f"C{i_z}"
            m = markers.get(f"z{i_z}", None) or "o"
            y = []
            yerr = []
            for t in tag_list:
                p = _pick(all_points, out_tag=t, dist=dist, zrange_key=zkey)
                # 条件分岐: `p is None` を満たす経路を評価する。
                if p is None:
                    y.append(float("nan"))
                    yerr.append(float("nan"))
                else:
                    y.append(float(p.eps))
                    yerr.append(float(p.sigma_eps))

            x = np.arange(len(tag_list), dtype=float)
            y = np.asarray(y, dtype=float)
            yerr = np.asarray(yerr, dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=m,
                linestyle="-",
                color=c,
                capsize=3,
                markersize=5,
                label=zkey.replace("_", " "),
            )

        ax.set_title(f"{sample}/{caps} : ε（peakfit; dist={dist}）")
        ax.set_xlabel("座標化仕様（差分）")
        ax.set_ylabel("ε (AP warping)")
        ax.set_xticks(np.arange(len(tag_list), dtype=float))
        ax.set_xticklabels(tag_labels, rotation=20, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("BAO coordinate-spec sensitivity (catalog-based peakfit)", fontsize=12)

    # 条件分岐: `str(args.out_png).strip()` を満たす経路を評価する。
    if str(args.out_png).strip():
        out_png = (_ROOT / str(args.out_png)).resolve()
    else:
        out_png = (_ROOT / "output" / "private" / "cosmology" / f"cosmology_bao_catalog_coordinate_spec_sensitivity__{sample}_{caps}__{base_tag}.png").resolve()

    # 条件分岐: `str(args.out_json).strip()` を満たす経路を評価する。

    if str(args.out_json).strip():
        out_json = (_ROOT / str(args.out_json)).resolve()
    else:
        out_json = (_ROOT / "output" / "private" / "cosmology" / f"cosmology_bao_catalog_coordinate_spec_sensitivity__{sample}_{caps}__{base_tag}_metrics.json").resolve()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4.5.2 (coordinate_spec sensitivity)",
        "inputs": {"peakfit_metrics_by_out_tag": by_tag_inputs, "base_out_tag": base_tag, "variant_out_tags": variant_tags},
        "summary": {
            "sample": sample,
            "caps": caps,
            "dists": dist_list,
            "z_ranges": z_ranges,
            "tags": [{"out_tag": t, "label": lab} for t, lab in zip(tag_list, tag_labels)],
        },
        "points": [
            {
                "out_tag": p.out_tag,
                "label": p.label,
                "dist": p.dist,
                "z_min": p.z_min,
                "z_max": p.z_max,
                "z_eff": p.z_eff,
                "eps": p.eps,
                "sigma_eps_1sigma": p.sigma_eps,
                "abs_sigma": p.abs_sigma,
                "status": p.status,
                "xi_metrics_json": str(p.xi_metrics_path),
                "xi_coordinate_spec_hash": p.xi_coord_hash,
            }
            for p in all_points
        ],
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_coordinate_spec_sensitivity",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {"base_out_tag": base_tag, "variant_out_tags": variant_tags},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

