#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_read_json` の入出力契約と処理意図を定義する。
def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None

    # 条件分岐: `not math.isfinite(out)` を満たす経路を評価する。

    if not math.isfinite(out):
        return None

    return out


# 関数: `_find_row` の入出力契約と処理意図を定義する。

def _find_row(rows: Sequence[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        # 条件分岐: `str(row.get("key", "")).strip().lower() == key.lower()` を満たす経路を評価する。
        if str(row.get("key", "")).strip().lower() == key.lower():
            return row

    return None


# 関数: `_budget_scattering_proxy` の入出力契約と処理意図を定義する。

def _budget_scattering_proxy(payload_budget: Dict[str, Any]) -> Optional[float]:
    row = ((payload_budget.get("rows") or {}).get("sgra")) or {}
    keys = [
        "kappa_sigma_proxy_scattering_kernel_major_over_ring",
        "kappa_sigma_proxy_refractive_wander_mid_over_ring",
        "kappa_sigma_proxy_refractive_distortion_mid_over_ring",
    ]
    vals = [_safe_float(row.get(k)) for k in keys]
    vals_f = [v for v in vals if v is not None and v > 0]
    # 条件分岐: `not vals_f` を満たす経路を評価する。
    if not vals_f:
        return None

    return float(max(vals_f))


# 関数: `_predict_row` の入出力契約と処理意図を定義する。

def _predict_row(
    row: Dict[str, Any],
    *,
    sigma_scat_override: Optional[float],
) -> Optional[Dict[str, Any]]:
    key = str(row.get("key", "")).strip().lower()
    name = str(row.get("name", key))
    theta_sh = _safe_float(row.get("shadow_diameter_pmodel_uas"))
    ring_sigma = _safe_float(row.get("ring_diameter_obs_uas_sigma"))
    kappa_fit = _safe_float(row.get("kappa_ring_over_shadow_fit_pmodel"))
    sigma_abs = _safe_float(row.get("kappa_sigma_assumed_kerr"))
    # 条件分岐: `theta_sh is None or theta_sh <= 0 or ring_sigma is None or kappa_fit is None` を満たす経路を評価する。
    if theta_sh is None or theta_sh <= 0 or ring_sigma is None or kappa_fit is None:
        return None

    # 条件分岐: `sigma_abs is None or sigma_abs <= 0` を満たす経路を評価する。

    if sigma_abs is None or sigma_abs <= 0:
        sigma_abs = 0.0

    sigma_emit = float(ring_sigma / theta_sh)
    sigma_scat = float(sigma_scat_override or 0.0)
    sigma_fp = float(math.sqrt(max(0.0, sigma_emit * sigma_emit + sigma_abs * sigma_abs + sigma_scat * sigma_scat)))
    kappa_fp_center = 1.0
    kappa_fp_min_1sigma = float(kappa_fp_center - sigma_fp)
    kappa_fp_max_1sigma = float(kappa_fp_center + sigma_fp)
    kappa_fp_min_3sigma = float(kappa_fp_center - 3.0 * sigma_fp)
    kappa_fp_max_3sigma = float(kappa_fp_center + 3.0 * sigma_fp)
    z_fit_vs_fp = float((kappa_fit - kappa_fp_center) / sigma_fp) if sigma_fp > 0 else math.inf
    status = "pass" if abs(z_fit_vs_fp) <= 3.0 else "reject"

    return {
        "key": key,
        "name": name,
        "kappa_fit": kappa_fit,
        "kappa_fp_center": kappa_fp_center,
        "kappa_fp_sigma_1sigma": sigma_fp,
        "kappa_fp_interval_1sigma": [kappa_fp_min_1sigma, kappa_fp_max_1sigma],
        "kappa_fp_interval_3sigma": [kappa_fp_min_3sigma, kappa_fp_max_3sigma],
        "sigma_components": {
            "emit_gradient_sigma": sigma_emit,
            "absorption_sigma": sigma_abs,
            "scattering_sigma": sigma_scat,
        },
        "z_fit_vs_fp_center": z_fit_vs_fp,
        "status": status,
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(rows: Sequence[Dict[str, Any]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib not available: {exc}") from exc

    _set_japanese_font()
    labels = [str(r["name"]) for r in rows]
    x = np.arange(len(labels), dtype=float)
    kappa_fit = np.array([float(r["kappa_fit"]) for r in rows], dtype=float)
    sigma = np.array([float(r["kappa_fp_sigma_1sigma"]) for r in rows], dtype=float)

    fig = plt.figure(figsize=(8.8, 5.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(
        x,
        np.ones_like(x),
        yerr=sigma,
        fmt="o",
        color="#1f77b4",
        capsize=5,
        label="κ_fp (1σ, first-principles transfer envelope)",
    )
    ax.scatter(x, kappa_fit, color="#d62728", marker="s", s=50, label="κ_fit (from observed ring)")
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("κ")
    ax.set_title("EHT κ: first-principles transfer envelope vs fitted κ")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: Sequence[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "key",
                "name",
                "kappa_fit",
                "kappa_fp_center",
                "kappa_fp_sigma_1sigma",
                "kappa_fp_min_1sigma",
                "kappa_fp_max_1sigma",
                "kappa_fp_min_3sigma",
                "kappa_fp_max_3sigma",
                "z_fit_vs_fp_center",
                "status",
                "sigma_emit_gradient",
                "sigma_absorption",
                "sigma_scattering",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["key"],
                    row["name"],
                    row["kappa_fit"],
                    row["kappa_fp_center"],
                    row["kappa_fp_sigma_1sigma"],
                    row["kappa_fp_interval_1sigma"][0],
                    row["kappa_fp_interval_1sigma"][1],
                    row["kappa_fp_interval_3sigma"][0],
                    row["kappa_fp_interval_3sigma"][1],
                    row["z_fit_vs_fp_center"],
                    row["status"],
                    row["sigma_components"]["emit_gradient_sigma"],
                    row["sigma_components"]["absorption_sigma"],
                    row["sigma_components"]["scattering_sigma"],
                ]
            )


# 関数: `_mirror_to_public` の入出力契約と処理意図を定義する。

def _mirror_to_public(*, src_json: Path, src_csv: Path, src_png: Path, public_dir: Path) -> Dict[str, str]:
    public_dir.mkdir(parents=True, exist_ok=True)
    dst_json = public_dir / src_json.name
    dst_csv = public_dir / src_csv.name
    dst_png = public_dir / src_png.name
    shutil.copy2(src_json, dst_json)
    shutil.copy2(src_csv, dst_csv)
    shutil.copy2(src_png, dst_png)
    return {
        "json": str(dst_json),
        "csv": str(dst_csv),
        "plot_png": str(dst_png),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _ROOT
    default_shadow = root / "output" / "private" / "eht" / "eht_shadow_compare.json"
    default_budget = root / "output" / "private" / "eht" / "eht_kappa_error_budget.json"
    default_outdir = root / "output" / "private" / "eht"
    default_public_outdir = root / "output" / "public" / "eht"

    ap = argparse.ArgumentParser(
        description="First-principles κ envelope from P-metric radiative transfer boundary conditions (non-GRMHD)."
    )
    ap.add_argument("--shadow-compare-json", type=str, default=str(default_shadow))
    ap.add_argument("--kappa-budget-json", type=str, default=str(default_budget))
    ap.add_argument("--outdir", type=str, default=str(default_outdir))
    ap.add_argument("--public-outdir", type=str, default=str(default_public_outdir))
    ap.add_argument("--skip-public-copy", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_shadow = Path(args.shadow_compare_json)
    in_budget = Path(args.kappa_budget_json)
    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_kappa_first_principles_transfer.json"
    out_csv = outdir / "eht_kappa_first_principles_transfer.csv"
    out_png = outdir / "eht_kappa_first_principles_transfer.png"

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "shadow_compare_json": str(in_shadow),
            "kappa_budget_json": str(in_budget),
        },
        "equations": {
            "optical_metric": "g_opt_{μν}=g_{μν}(P)+(1-n(P)^{-2})U_μU_ν,  n(P)=(P/P0)^(2β)",
            "ray_equation": "g_opt^{μν}k_μk_ν=0,  dk_μ/dλ=-(1/2)∂_μg_opt^{αβ}k_αk_β",
            "transfer_equation": "k^μ∇^{(g(P))}_μĨ_ν = J̃_ν - Ã_ν Ĩ_ν,  Ĩ_ν=I_ν/ν^3",
            "formal_solution": "Ĩ_obs(b)=∫ J̃_ν(λ,b) exp[-τ_ν(λ,b)] dλ,  τ_ν=∫ Ã_ν dλ",
            "kappa_definition": "κ=b_peak/b_sh(P),  dI_obs(b)/db|_{b=b_peak}=0",
            "first_order_closure": "κ≈1 + (Δ_emit-Δ_abs-Δ_scat),  σ_κ^2≈σ_emit^2+σ_abs^2+σ_scat^2",
        },
        "boundary_conditions": {
            "horizon": "r→r_H^+: k^r<0, Ĩ_ν finite, τ_ν finite",
            "infinity": "r→∞: u→0, P_φ→0, Ĩ_ν→0, τ_ν→0",
            "axis": "θ=0,π: P_φ=0, ∂_θĨ_ν=0",
        },
        "rows": [],
        "overall_status": "watch",
        "outputs": {
            "json": str(out_json),
            "csv": str(out_csv),
            "plot_png": str(out_png),
        },
    }

    # 条件分岐: `not in_shadow.exists()` を満たす経路を評価する。
    if not in_shadow.exists():
        payload["overall_status"] = "reject"
        payload["reason"] = "missing_shadow_compare_json"
        _write_json(out_json, payload)
        print(f"[warn] missing input: {in_shadow}")
        print(f"[ok] json: {out_json}")
        return 0

    shadow_payload = _read_json(in_shadow)
    rows_shadow = shadow_payload.get("rows") or []
    # 条件分岐: `not isinstance(rows_shadow, list)` を満たす経路を評価する。
    if not isinstance(rows_shadow, list):
        rows_shadow = []

    budget_payload: Dict[str, Any] = {}
    # 条件分岐: `in_budget.exists()` を満たす経路を評価する。
    if in_budget.exists():
        budget_payload = _read_json(in_budget)

    sgra_scat_sigma = _budget_scattering_proxy(budget_payload) if budget_payload else None

    rows_out: List[Dict[str, Any]] = []
    for key in ("m87", "sgra"):
        row = _find_row(rows_shadow, key)
        # 条件分岐: `not row` を満たす経路を評価する。
        if not row:
            continue

        sigma_scat = sgra_scat_sigma if key == "sgra" else None
        row_out = _predict_row(row, sigma_scat_override=sigma_scat)
        # 条件分岐: `row_out` を満たす経路を評価する。
        if row_out:
            rows_out.append(row_out)

    payload["rows"] = rows_out
    # 条件分岐: `not rows_out` を満たす経路を評価する。
    if not rows_out:
        payload["overall_status"] = "reject"
        payload["reason"] = "no_usable_rows"
        _write_json(out_json, payload)
        print(f"[warn] no usable rows in {in_shadow}")
        print(f"[ok] json: {out_json}")
        return 0

    max_abs_z = max(abs(float(r["z_fit_vs_fp_center"])) for r in rows_out)
    payload["summary"] = {
        "n_rows": len(rows_out),
        "max_abs_z_fit_vs_fp": max_abs_z,
        "all_within_3sigma": bool(max_abs_z <= 3.0),
        "scattering_sigma_sgra": sgra_scat_sigma,
    }
    payload["overall_status"] = "pass" if max_abs_z <= 3.0 else "watch"

    _write_json(out_json, payload)
    _write_csv(rows_out, out_csv)
    _plot(rows_out, out_png)

    public_outputs: Optional[Dict[str, str]] = None
    # 条件分岐: `not args.skip_public_copy` を満たす経路を評価する。
    if not args.skip_public_copy:
        public_outputs = _mirror_to_public(
            src_json=out_json, src_csv=out_csv, src_png=out_png, public_dir=public_outdir
        )
        payload["outputs_public"] = public_outputs
        _write_json(out_json, payload)

    try:
        metrics = {
            "overall_status": payload["overall_status"],
            "n_rows": len(rows_out),
            "max_abs_z_fit_vs_fp": max_abs_z,
        }
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_kappa_first_principles_transfer",
                "inputs": {
                    "shadow_compare_json": str(in_shadow.relative_to(root)).replace("\\", "/"),
                    "kappa_budget_json": (
                        str(in_budget.relative_to(root)).replace("\\", "/") if in_budget.exists() else None
                    ),
                },
                "outputs": {
                    "json": str(out_json.relative_to(root)).replace("\\", "/"),
                    "csv": str(out_csv.relative_to(root)).replace("\\", "/"),
                    "png": str(out_png.relative_to(root)).replace("\\", "/"),
                    "public_json": (
                        str(Path(public_outputs["json"]).relative_to(root)).replace("\\", "/")
                        if public_outputs and "json" in public_outputs
                        else None
                    ),
                },
                "metrics": metrics,
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    # 条件分岐: `public_outputs` を満たす経路を評価する。
    if public_outputs:
        print(f"[ok] public json: {public_outputs['json']}")
        print(f"[ok] public csv : {public_outputs['csv']}")
        print(f"[ok] public png : {public_outputs['plot_png']}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
