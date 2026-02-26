#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_sn_time_dilation_constraints.py

Step 14.2.3（独立プローブで相互検証）:
距離指標（D_L/D_A）と独立に、「時間伸長（time dilation）」を一次ソースの制約として固定する。

目的：
- 背景P（宇宙膨張なし）でも赤方偏移が出るなら、時間伸長も同じ指数で出るのか（p_t）を観測で拘束する。
- DDR（距離二重性）の“回復”を p_t の変更だけで行う逃げ道（例：p_t≈3）を、SN time dilation で排除できるかを示す。

定義：
  Δt_obs = (1+z)^(p_t) Δt_em
  aging rate g(z) ≡ Δt_em/Δt_obs = (1+z)^(-p_t)

入力（固定）:
  - data/cosmology/sn_time_dilation_constraints.json

出力（固定名）:
  - output/private/cosmology/cosmology_sn_time_dilation_constraints.png
  - output/private/cosmology/cosmology_sn_time_dilation_constraints_metrics.json
  - output/private/cosmology/cosmology_sn_time_dilation_pt_fit.png
  - output/private/cosmology/cosmology_sn_time_dilation_pt_fit.json

補足（Phase 4 / Step 4.3.1）：
- Blondin+2008（arXiv:0804.3595）の Table 3（aging rate）を一次PDFから抽出し、
  g(z)=1/(1+z)^(p_t)（p_t自由; zero-point固定）で再fitして「(1+z) を前処理に埋め込まない」形を固定出力として残す。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。
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


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_fmt_float` の入出力契約と処理意図を定義する。

def _fmt_float(x: float, *, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# クラス: `Constraint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class Constraint:
    id: str
    short_label: str
    title: str
    p_t: float
    p_t_sigma: float
    sigma_note: str
    source: Dict[str, Any]

    # 関数: `from_json` の入出力契約と処理意図を定義する。
    @staticmethod
    def from_json(j: Dict[str, Any]) -> "Constraint":
        return Constraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            p_t=float(j["p_t"]),
            p_t_sigma=float(j["p_t_sigma"]),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


# クラス: `AgingRatePoint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class AgingRatePoint:
    sn: str
    z: float
    inv_1pz: float
    aging_rate: float
    sigma: float

    # 関数: `sample` の入出力契約と処理意図を定義する。
    @property
    def sample(self) -> str:
        # 条件分岐: `self.z < 0.04` を満たす経路を評価する。
        if self.z < 0.04:
            return "lowz"

        # 条件分岐: `self.z > 0.2` を満たす経路を評価する。

        if self.z > 0.2:
            return "highz"

        return "midz"


# 関数: `compute` の入出力契約と処理意図を定義する。

def compute(rows: Sequence[Constraint]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # Model benchmarks
    p_t_frw = 1.0
    p_t_tired = 0.0
    # If one tries to recover DDR (ε0≈0) only by changing p_t, with p_e=1 and no opacity/evolution,
    # ε0_model = (1+p_t)/2 - 2 = (p_t-3)/2  =>  ε0=0 -> p_t=3.
    p_t_ddr_fix_only = 3.0

    for r in rows:
        sig = float(r.p_t_sigma)
        # 条件分岐: `not (sig > 0.0)` を満たす経路を評価する。
        if not (sig > 0.0):
            raise ValueError(f"p_t_sigma must be >0: {r.id}")

        # 関数: `z` の入出力契約と処理意図を定義する。

        def z(model_pt: float) -> float:
            return (model_pt - float(r.p_t)) / sig

        out.append(
            {
                "id": r.id,
                "short_label": r.short_label,
                "title": r.title,
                "p_t_obs": float(r.p_t),
                "p_t_sigma": sig,
                "z_frw": z(p_t_frw),
                "z_tired_light": z(p_t_tired),
                "z_ddr_fix_only": z(p_t_ddr_fix_only),
                "benchmarks": {
                    "p_t_frw": p_t_frw,
                    "p_t_tired_light": p_t_tired,
                    "p_t_ddr_fix_only": p_t_ddr_fix_only,
                },
                "sigma_note": r.sigma_note,
                "source": r.source,
            }
        )

    return out


# 関数: `_extract_text_from_pdf` の入出力契約と処理意図を定義する。

def _extract_text_from_pdf(pdf_path: Path) -> str:
    # pypdf may emit cryptography deprecation warnings in some environments; silence for clean logs.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)


# 関数: `_extract_blondin2008_table3` の入出力契約と処理意図を定義する。

def _extract_blondin2008_table3(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract Table 3 (aging rate measurements) from Blondin+2008 primary PDF (arXiv:0804.3595).

    Table 3 columns (as extracted by pypdf text):
      SN, z, 1/(1+z), aging rate g, (sigma)
    """
    text = _extract_text_from_pdf(pdf_path)
    i0 = text.find("TABLE 3")
    # 条件分岐: `i0 < 0` を満たす経路を評価する。
    if i0 < 0:
        raise ValueError("TABLE 3 not found in PDF text")

    i1 = text.find("TABLE 4", i0)
    # 条件分岐: `i1 < 0` を満たす経路を評価する。
    if i1 < 0:
        # Fallback: table end not found; take a conservative tail window.
        i1 = min(len(text), i0 + 12000)

    block = text[i0:i1]

    row_re = re.compile(
        r"(?P<sn>[0-9A-Za-z]+)\s+"
        r"(?P<z>[0-9]+\.[0-9]+)\s+"
        r"(?P<inv>[0-9]+\.[0-9]+)\s+"
        r"(?P<g>[0-9]+\.[0-9]+)\s*"
        r"\((?P<sig>[0-9]+\.[0-9]+)\)"
    )

    points: List[AgingRatePoint] = []
    for m in row_re.finditer(block):
        points.append(
            AgingRatePoint(
                sn=str(m.group("sn")),
                z=float(m.group("z")),
                inv_1pz=float(m.group("inv")),
                aging_rate=float(m.group("g")),
                sigma=float(m.group("sig")),
            )
        )

    # 条件分岐: `not points` を満たす経路を評価する。

    if not points:
        raise ValueError("no rows parsed from Blondin+2008 Table 3 block")

    # Sanity: extracted inv_1pz should match 1/(1+z)

    diffs = [abs(p.inv_1pz - 1.0 / (1.0 + p.z)) for p in points]
    sanity = {
        "n_points": int(len(points)),
        "max_abs_inv_1pz_diff": float(max(diffs)),
        "median_abs_inv_1pz_diff": float(np.median(diffs)),
    }

    return {
        "pdf": str(pdf_path).replace("\\", "/"),
        "table": "Blondin2008 Table 3",
        "points": [p.__dict__ | {"sample": p.sample} for p in points],
        "sanity": sanity,
    }


# 関数: `_fit_pt_from_aging_rates` の入出力契約と処理意図を定義する。

def _fit_pt_from_aging_rates(points: Sequence[AgingRatePoint]) -> Dict[str, Any]:
    # 条件分岐: `not points` を満たす経路を評価する。
    if not points:
        raise ValueError("no points to fit")

    z = np.array([p.z for p in points], dtype=float)
    g = np.array([p.aging_rate for p in points], dtype=float)
    sig = np.array([p.sigma for p in points], dtype=float)

    # 条件分岐: `np.any(sig <= 0.0)` を満たす経路を評価する。
    if np.any(sig <= 0.0):
        raise ValueError("sigma must be > 0")

    one_pz = 1.0 + z

    # Fit single parameter p_t in g = 1/(1+z)^(p_t), with fixed zero-point g(z=0)=1.
    # Low-z sample alone does not constrain p_t well; keep a wide grid so the 1σ interval is captured.
    b_min, b_max, n_grid = (-0.5, 6.0, 13001)
    b_grid = np.linspace(b_min, b_max, n_grid, dtype=float)
    pred = one_pz[:, None] ** (-b_grid[None, :])
    chi2 = np.sum(((g[:, None] - pred) / sig[:, None]) ** 2, axis=0)

    i_best = int(np.argmin(chi2))
    b_best = float(b_grid[i_best])
    chi2_best = float(chi2[i_best])
    dof = max(1, int(len(points) - 1))

    target = chi2_best + 1.0

    # Find 1σ interval by chi2=chi2_min+1 (1 parameter)
    b_lo: Optional[float] = None
    b_hi: Optional[float] = None

    i = i_best
    while i > 0 and float(chi2[i]) <= target:
        i -= 1

    # 条件分岐: `i < i_best` を満たす経路を評価する。

    if i < i_best:
        x0, y0 = float(b_grid[i]), float(chi2[i])
        x1, y1 = float(b_grid[i + 1]), float(chi2[i + 1])
        # 条件分岐: `y1 != y0` を満たす経路を評価する。
        if y1 != y0:
            b_lo = x0 + (target - y0) * (x1 - x0) / (y1 - y0)

    i = i_best
    while i < n_grid - 1 and float(chi2[i]) <= target:
        i += 1

    # 条件分岐: `i > i_best` を満たす経路を評価する。

    if i > i_best:
        x0, y0 = float(b_grid[i - 1]), float(chi2[i - 1])
        x1, y1 = float(b_grid[i]), float(chi2[i])
        # 条件分岐: `y1 != y0` を満たす経路を評価する。
        if y1 != y0:
            b_hi = x0 + (target - y0) * (x1 - x0) / (y1 - y0)

    sig_minus = None if b_lo is None else float(b_best - b_lo)
    sig_plus = None if b_hi is None else float(b_hi - b_best)
    sig_sym = None
    # 条件分岐: `sig_minus is not None and sig_plus is not None` を満たす経路を評価する。
    if sig_minus is not None and sig_plus is not None:
        sig_sym = 0.5 * (sig_minus + sig_plus)

    return {
        "model": "g(z)=1/(1+z)^(p_t) (zero-point fixed at g(z=0)=1)",
        "n_points": int(len(points)),
        "p_t_fit": b_best,
        "p_t_sigma_minus": sig_minus,
        "p_t_sigma_plus": sig_plus,
        "p_t_sigma_sym": sig_sym,
        "chi2_min": chi2_best,
        "dof": dof,
        "reduced_chi2": float(chi2_best / dof),
        "grid": {"b_min": b_min, "b_max": b_max, "n": n_grid},
    }


# 関数: `_plot_pt_fit` の入出力契約と処理意図を定義する。

def _plot_pt_fit(
    *,
    table: Dict[str, Any],
    fit_all: Dict[str, Any],
    fit_highz: Dict[str, Any],
    fit_lowz: Dict[str, Any],
    out_png: Path,
) -> None:
    pts = table["points"]
    z = np.array([float(p["z"]) for p in pts], dtype=float)
    g = np.array([float(p["aging_rate"]) for p in pts], dtype=float)
    sig = np.array([float(p["sigma"]) for p in pts], dtype=float)
    sample = [str(p.get("sample") or "") for p in pts]

    one_pz = 1.0 + z

    p_fit = float(fit_all["p_t_fit"])
    p_sig = fit_all.get("p_t_sigma_sym")
    p_sig = float(p_sig) if isinstance(p_sig, (int, float)) else None

    x = np.linspace(float(np.min(one_pz)) * 0.98, float(np.max(one_pz)) * 1.02, 250)
    y_frw = x ** (-1.0)
    y_tired = x ** (0.0)
    y_fit = x ** (-p_fit)
    y_fit_lo = None
    y_fit_hi = None
    # 条件分岐: `p_sig is not None` を満たす経路を評価する。
    if p_sig is not None:
        y_fit_lo = x ** (-(p_fit - p_sig))
        y_fit_hi = x ** (-(p_fit + p_sig))

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    # Panel 1: data + model curves
    colors = {"lowz": "#1f77b4", "highz": "#ff7f0e", "midz": "#7f7f7f"}
    for key in ("lowz", "highz", "midz"):
        idx = [i for i, s in enumerate(sample) if s == key]
        # 条件分岐: `not idx` を満たす経路を評価する。
        if not idx:
            continue

        ax1.errorbar(
            one_pz[idx],
            g[idx],
            yerr=sig[idx],
            fmt="o",
            capsize=3,
            color=colors[key],
            ecolor=colors[key],
            label={"lowz": "low-z", "highz": "high-z", "midz": "mid-z"}.get(key, key),
            alpha=0.95,
        )

    ax1.plot(x, y_frw, color="#2ca02c", linewidth=2.0, label="FRW: p_t=1")
    ax1.plot(x, y_tired, color="#d62728", linewidth=2.0, linestyle="--", label="no dilation: p_t=0")
    ax1.plot(x, y_fit, color="#111111", linewidth=2.0, label=f"fit (all): p_t={p_fit:.2f}")
    # 条件分岐: `y_fit_lo is not None and y_fit_hi is not None` を満たす経路を評価する。
    if y_fit_lo is not None and y_fit_hi is not None:
        ax1.fill_between(x, y_fit_hi, y_fit_lo, color="#111111", alpha=0.10, label="fit ±1σ (approx)")

    ax1.set_xlabel("1+z", fontsize=11)
    ax1.set_ylabel("aging rate g = Δt_em/Δt_obs", fontsize=11)
    ax1.set_title("Blondin+2008 Table 3: aging rate vs redshift", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: summary bars for fits (all/highz/lowz)
    labels = ["all", "high-z", "low-z"]
    vals = [float(fit_all["p_t_fit"]), float(fit_highz["p_t_fit"]), float(fit_lowz["p_t_fit"])]
    errs = []
    for f in (fit_all, fit_highz, fit_lowz):
        s = f.get("p_t_sigma_sym")
        errs.append(float(s) if isinstance(s, (int, float)) else 0.0)

    y_pos = np.arange(len(labels))[::-1]
    ax2.axvline(1.0, color="#2ca02c", linewidth=1.6, alpha=0.85, label="p_t=1")
    ax2.axvline(0.0, color="#d62728", linewidth=1.2, alpha=0.85, linestyle="--", label="p_t=0")
    ax2.errorbar(vals, y_pos, xerr=errs, fmt="o", capsize=4, color="#111111", ecolor="#111111")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("p_t (fit)", fontsize=11)
    ax2.set_title("p_t fit (zero-point fixed)", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle("SN time dilation audit: p_t refit from primary Table 3 (non-circular I/F)", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "注：g(z) は観測時間軸を事前に (1+z) で割り戻さず、Table 3 の公表値（Δt_em/Δt_obs）をそのまま用いて p_t を自由に推定する。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(rows: Sequence[Dict[str, Any]], *, out_png: Path) -> None:
    labels = [str(r.get("short_label") or r.get("id") or "") for r in rows]
    y = np.arange(len(rows))[::-1]

    p_obs = np.array([float(r["p_t_obs"]) for r in rows], dtype=float)
    p_sig = np.array([float(r["p_t_sigma"]) for r in rows], dtype=float)

    z_frw = np.array([float(r["z_frw"]) for r in rows], dtype=float)
    z_tired = np.array([float(r["z_tired_light"]) for r in rows], dtype=float)
    z_ddr = np.array([float(r["z_ddr_fix_only"]) for r in rows], dtype=float)

    p_frw = float(rows[0]["benchmarks"]["p_t_frw"]) if rows else 1.0
    p_tired = float(rows[0]["benchmarks"]["p_t_tired_light"]) if rows else 0.0
    p_ddr = float(rows[0]["benchmarks"]["p_t_ddr_fix_only"]) if rows else 3.0

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    # Left: z-scores
    ax1.axvline(0.0, color="k", linewidth=1.0, alpha=0.6)
    ax1.axvline(-3.0, color="#999999", linewidth=1.0, alpha=0.7, linestyle="--")
    ax1.axvline(3.0, color="#999999", linewidth=1.0, alpha=0.7, linestyle="--")

    dy = 0.12
    ax1.scatter(z_frw, y + dy, s=45, label=f"FRW: p_t={_fmt_float(p_frw, digits=3)}", color="#1f77b4")
    ax1.scatter(
        z_tired,
        y,
        s=45,
        label=f"tired light: p_t={_fmt_float(p_tired, digits=3)}（時間伸長なし）",
        color="#d62728",
    )
    ax1.scatter(
        z_ddr,
        y - dy,
        s=45,
        label=f"DDRをp_tだけで回復: p_t={_fmt_float(p_ddr, digits=3)}",
        color="#2ca02c",
    )

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("z-score（(p_t_model - p_t_obs)/σ）", fontsize=11)
    ax1.set_title("SN time dilation：モデルのz-score", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax1.legend(fontsize=9, loc="lower right")

    # Right: observed p_t with reference lines
    ax2.axvline(p_frw, color="#1f77b4", linewidth=1.2, alpha=0.85, label="FRW: p_t=1")
    ax2.axvline(p_tired, color="#d62728", linewidth=1.2, alpha=0.85, label="tired light: p_t=0")
    ax2.axvline(p_ddr, color="#2ca02c", linewidth=1.2, alpha=0.85, label="DDR回復（p_tのみ）: p_t=3")
    ax2.errorbar(
        p_obs,
        y,
        xerr=p_sig,
        fmt="o",
        capsize=4,
        color="#111111",
        ecolor="#111111",
        label="観測（一次ソースの公表値）",
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("p_t", fontsize=11)
    ax2.set_title("観測 p_t（時間伸長指数）の一次ソース制約", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle("宇宙論（独立プローブ）：SN time dilation の制約（p_t）", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "注：距離二重性（DDR）の回復を p_t の変更だけで行う場合、p_t≈3 が必要になるが、SN time dilation の一次ソース制約では強く排除される。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: SN time dilation constraints (p_t).")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "sn_time_dilation_constraints.json"),
        help="Input JSON (default: data/cosmology/sn_time_dilation_constraints.json)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    src = _read_json(data_path)
    constraints = [Constraint.from_json(c) for c in (src.get("constraints") or [])]
    # 条件分岐: `not constraints` を満たす経路を評価する。
    if not constraints:
        raise SystemExit(f"no constraints found in: {data_path}")

    rows = compute(constraints)

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cosmology_sn_time_dilation_constraints.png"
    _plot(rows, out_png=png_path)

    # 4.3.1: p_t refit + (1+z) circularity audit (Blondin+2008; Table 3 from primary PDF)
    pt_fit_png: Optional[Path] = None
    pt_fit_json: Optional[Path] = None
    pt_fit_payload: Optional[Dict[str, Any]] = None

    blondin_constraint: Optional[Constraint] = None
    for c in constraints:
        src_meta = c.source or {}
        # 条件分岐: `str(src_meta.get("arxiv_id") or "") == "0804.3595" or c.id.startswith("blondi...` を満たす経路を評価する。
        if str(src_meta.get("arxiv_id") or "") == "0804.3595" or c.id.startswith("blondin2008"):
            blondin_constraint = c
            break

    # 条件分岐: `blondin_constraint is not None` を満たす経路を評価する。

    if blondin_constraint is not None:
        local_pdf = blondin_constraint.source.get("local_pdf")
        pdf_path = (_ROOT / local_pdf) if local_pdf else None
        # 条件分岐: `pdf_path is not None and pdf_path.exists()` を満たす経路を評価する。
        if pdf_path is not None and pdf_path.exists():
            try:
                table = _extract_blondin2008_table3(pdf_path)
                pts_all = [
                    AgingRatePoint(
                        sn=str(p["sn"]),
                        z=float(p["z"]),
                        inv_1pz=float(p["inv_1pz"]),
                        aging_rate=float(p["aging_rate"]),
                        sigma=float(p["sigma"]),
                    )
                    for p in table["points"]
                ]
                pts_highz = [p for p in pts_all if p.sample == "highz"]
                pts_lowz = [p for p in pts_all if p.sample == "lowz"]

                fit_all = _fit_pt_from_aging_rates(pts_all)
                fit_highz = _fit_pt_from_aging_rates(pts_highz)
                fit_lowz = _fit_pt_from_aging_rates(pts_lowz)

                audit = {
                    "circularity_risk_main_fit": "low",
                    "where_1_plus_z_enters": [
                        "スペクトルを rest-frame に戻すための波長補正（λ_rest=λ_obs/(1+z)）。",
                        "モデルの独立変数としての (1+z)（g(z)=1/(1+z)^(p_t) の形）。",
                    ],
                    "explicitly_not_used_in_main_fit": [
                        "観測時間軸の事前割り戻し（t_rest=t_obs/(1+z)）を前処理として作らない。",
                        "light-curve age を用いた補助比較（Appendix 系）は主fitから分離する。",
                    ],
                    "note": "本再fitは Table 3 の公表値 g=Δt_em/Δt_obs とその誤差を用い、p_t を自由パラメータとして推定する（p_t=1 を埋め込まない）。",
                }

                pt_fit_png = out_dir / "cosmology_sn_time_dilation_pt_fit.png"
                _plot_pt_fit(
                    table=table,
                    fit_all=fit_all,
                    fit_highz=fit_highz,
                    fit_lowz=fit_lowz,
                    out_png=pt_fit_png,
                )

                pt_fit_json = out_dir / "cosmology_sn_time_dilation_pt_fit.json"
                pt_fit_payload = {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "source_constraint_id": blondin_constraint.id,
                    "source": blondin_constraint.source,
                    "table3_extraction": table,
                    "fits": {"all": fit_all, "highz": fit_highz, "lowz": fit_lowz},
                    "audit": audit,
                    "outputs": {
                        "png": str(pt_fit_png).replace("\\", "/"),
                        "json": str(pt_fit_json).replace("\\", "/"),
                    },
                }
                _write_json(pt_fit_json, pt_fit_payload)
            except Exception as e:
                pt_fit_payload = {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "source_constraint_id": blondin_constraint.id,
                    "source": blondin_constraint.source,
                    "error": f"{type(e).__name__}: {e}",
                }

    out_json = out_dir / "cosmology_sn_time_dilation_constraints_metrics.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "definition": src.get("definition") or {},
        "assumptions": {
            "time_dilation": "Δt_obs=(1+z)^(p_t)Δt_em",
            "aging_rate": "g(z)=Δt_em/Δt_obs=(1+z)^(-p_t)",
            "benchmarks": {
                "FRW": {"p_t": 1.0},
                "tired_light": {"p_t": 0.0},
                "DDR_fix_only": {"p_t": 3.0, "note": "p_e=1, opacity/evolutionなしでε0=0を満たすための目安"},
            },
            "theory_branch_summary": {
                "if_phase_rate_preserved": "可変時計 dτ/dt=P0/P と、自由伝播で座標位相進み dθ/dt が保存される仮定の下では、Δt_obs=(1+z)Δt_em（p_t=1）を要求する。",
                "if_not_preserved": "dθ/dt 保存が崩れる場合は、同じ定義系のままでは赤方偏移写像（ν_obs/ν_em=P_obs/P_em）と両立せず、p_t は未確定（追加の伝播則が必要）。",
            },
        },
        "rows": rows,
        "audit": {
            "pt_refit": None if pt_fit_payload is None else pt_fit_payload.get("fits"),
            "circularity_risk": None if pt_fit_payload is None else (pt_fit_payload.get("audit") or {}).get("circularity_risk_main_fit"),
            "note": None if pt_fit_payload is None else (pt_fit_payload.get("audit") or {}).get("note"),
            "refit_outputs": None if pt_fit_payload is None else pt_fit_payload.get("outputs"),
            "refit_error": None if pt_fit_payload is None else pt_fit_payload.get("error"),
        },
        "outputs": {"png": str(png_path).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }

    # Attach audit to the relevant row for convenience.
    if pt_fit_payload is not None:
        for r in payload["rows"]:
            # 条件分岐: `str(r.get("id")) == str(pt_fit_payload.get("source_constraint_id"))` を満たす経路を評価する。
            if str(r.get("id")) == str(pt_fit_payload.get("source_constraint_id")):
                r["audit"] = {
                    "circularity_risk_main_fit": (pt_fit_payload.get("audit") or {}).get("circularity_risk_main_fit"),
                    "pt_fit_all": (pt_fit_payload.get("fits") or {}).get("all"),
                    "refit_outputs": pt_fit_payload.get("outputs"),
                    "refit_error": pt_fit_payload.get("error"),
                }
                break

    _write_json(out_json, payload)

    print(f"[ok] png : {png_path}")
    print(f"[ok] json: {out_json}")
    # 条件分岐: `pt_fit_png is not None and pt_fit_json is not None` を満たす経路を評価する。
    if pt_fit_png is not None and pt_fit_json is not None:
        print(f"[ok] pt-fit png : {pt_fit_png}")
        print(f"[ok] pt-fit json: {pt_fit_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_sn_time_dilation_constraints",
                "argv": list(sys.argv),
                "inputs": {"data": data_path},
                "outputs": {
                    "png": png_path,
                    "metrics_json": out_json,
                    **({} if pt_fit_png is None else {"pt_fit_png": pt_fit_png}),
                    **({} if pt_fit_json is None else {"pt_fit_json": pt_fit_json}),
                },
                "metrics": {
                    "n_constraints": len(rows),
                    **(
                        {}
                        if (pt_fit_payload is None or pt_fit_payload.get("fits") is None)
                        else {"pt_refit": {"p_t_fit": pt_fit_payload["fits"]["all"]["p_t_fit"], "n": pt_fit_payload["fits"]["all"]["n_points"]}}
                    ),
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
