"""
目的: 量子 topic の molecular isotopic scaling に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


from figure_japanese_localizer import enable_japanese_figure_localization
from scripts.utils.plot_style import apply_wavep_figure_layout, get_wavep_font_size

enable_japanese_figure_localization()

# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(x: object) -> float | None:
    try:
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None


# 関数: `_reduced_mass` の入出力契約と処理意図を定義する。

def _reduced_mass(m1: float, m2: float) -> float:
    return (m1 * m2) / (m1 + m2)


# 関数: `_load_nist_h_isotope_masses_u` の入出力契約と処理意図を定義する。

def _load_nist_h_isotope_masses_u(root: Path) -> tuple[dict[str, float], dict[str, Any]] | None:
    path = root / "data" / "quantum" / "sources" / "nist_isotopic_compositions_h" / "extracted_values.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    j = _read_json(path)
    isotopes = j.get("isotopes")
    # 条件分岐: `not isinstance(isotopes, list)` を満たす経路を評価する。
    if not isinstance(isotopes, list):
        return None

    masses: dict[str, float] = {}
    for iso in isotopes:
        # 条件分岐: `not isinstance(iso, dict)` を満たす経路を評価する。
        if not isinstance(iso, dict):
            continue

        sym = iso.get("symbol")
        a = iso.get("mass_number")
        m_u = iso.get("relative_atomic_mass_u")
        # 条件分岐: `not isinstance(sym, str) or not isinstance(a, int)` を満たす経路を評価する。
        if not isinstance(sym, str) or not isinstance(a, int):
            continue

        m = _as_float(m_u)
        # 条件分岐: `m is None` を満たす経路を評価する。
        if m is None:
            continue

        masses[f"{sym}{a}"] = m

    meta = {
        "source": "NIST Atomic Weights and Isotopic Compositions (stand_alone.pl)",
        "path": str(path),
        "query_url": j.get("query_url"),
        "raw_sha256": j.get("raw_sha256"),
    }
    return masses, meta


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal isotopologues for Step 7.12.
    # Prefer primary-source-backed isotope masses (NIST). Fallback to mass-number approximation.
    mass_model: dict[str, Any] = {"kind": "mass_number_approx", "note": "Fallback: H=1, D=2 (dimensionless)."}
    m = _load_nist_h_isotope_masses_u(root)
    # 条件分岐: `m is not None` を満たす経路を評価する。
    if m is not None:
        masses_u, meta = m
        # 条件分岐: `"H1" in masses_u and "D2" in masses_u` を満たす経路を評価する。
        if "H1" in masses_u and "D2" in masses_u:
            mass_model = {
                "kind": "nist_relative_atomic_mass_u",
                "note": "Uses relative atomic masses (u) as a primary-source-backed reduced-mass model.",
                "meta": meta,
                "masses_u": {"H1": masses_u["H1"], "D2": masses_u["D2"]},
            }

    # 関数: `_get_mass` の入出力契約と処理意図を定義する。

    def _get_mass(symbol_a: str) -> float:
        # 条件分岐: `mass_model["kind"] == "nist_relative_atomic_mass_u"` を満たす経路を評価する。
        if mass_model["kind"] == "nist_relative_atomic_mass_u":
            return float(mass_model["masses_u"][symbol_a])
        # mass-number approximation

        return {"H1": 1.0, "D2": 2.0}[symbol_a]

    species = [
        {"slug": "h2", "label": "H2", "m1": _get_mass("H1"), "m2": _get_mass("H1"), "isotopes": ("H1", "H1")},
        {"slug": "hd", "label": "HD", "m1": _get_mass("H1"), "m2": _get_mass("D2"), "isotopes": ("H1", "D2")},
        {"slug": "d2", "label": "D2", "m1": _get_mass("D2"), "m2": _get_mass("D2"), "isotopes": ("D2", "D2")},
    ]

    rows: list[dict[str, Any]] = []
    for s in species:
        path = out_dir / f"molecular_{s['slug']}_baseline_metrics.json"
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            raise SystemExit(f"[fail] missing baseline metrics: {path}\nRun molecular_h2_baseline.py first.")

        j = _read_json(path)
        consts = j.get("constants")
        # 条件分岐: `not isinstance(consts, dict)` を満たす経路を評価する。
        if not isinstance(consts, dict):
            raise SystemExit(f"[fail] constants missing in: {path}")

        omega_e = _as_float(consts.get("omega_e_cm^-1"))
        be = _as_float(consts.get("B_e_cm^-1"))
        # 条件分岐: `omega_e is None or be is None` を満たす経路を評価する。
        if omega_e is None or be is None:
            raise SystemExit(f"[fail] missing ωe or Be in: {path}")

        mu = _reduced_mass(float(s["m1"]), float(s["m2"]))
        rows.append(
            {
                "slug": s["slug"],
                "label": s["label"],
                "m1": s["m1"],
                "m2": s["m2"],
                "isotopes": s.get("isotopes"),
                "mu": mu,
                "omega_e_cm^-1": omega_e,
                "B_e_cm^-1": be,
                "source_metrics": str(path),
            }
        )

    # Use H2 as reference for reduced-mass scaling.

    ref = next(r for r in rows if r["slug"] == "h2")
    mu_ref = float(ref["mu"])
    omega_ref = float(ref["omega_e_cm^-1"])
    be_ref = float(ref["B_e_cm^-1"])

    for r in rows:
        mu = float(r["mu"])
        omega_pred = omega_ref * math.sqrt(mu_ref / mu)
        be_pred = be_ref * (mu_ref / mu)
        r["omega_e_pred_cm^-1"] = omega_pred
        r["B_e_pred_cm^-1"] = be_pred
        r["omega_e_ratio_meas_over_pred"] = float(r["omega_e_cm^-1"]) / omega_pred
        r["B_e_ratio_meas_over_pred"] = float(r["B_e_cm^-1"]) / be_pred

    # ---- Figure ----

    labels = [r["label"] for r in rows]
    x = list(range(len(labels)))
    omega_ratios = [r["omega_e_ratio_meas_over_pred"] for r in rows]
    be_ratios = [r["B_e_ratio_meas_over_pred"] for r in rows]

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 11.8), dpi=190)
    apply_wavep_figure_layout(fig, template="part2_two_panel_spacious")
    fig.set_size_inches(fig.get_figwidth(), 7.20, forward=True)
    panel_title_font = get_wavep_font_size("title") * 0.82
    axis_label_font = get_wavep_font_size("axis")
    tick_font = get_wavep_font_size("tick")
    suptitle_font = get_wavep_font_size("suptitle") + 2.0
    fig.suptitle("Isotopic reduced-mass scaling (WebBook diatomic constants)", fontsize=suptitle_font, y=0.992)

    ax = axes[0]
    ax.set_title("ωe scaling: ωe ∝ μ^{-1/2} (ratio vs prediction)", fontsize=panel_title_font, pad=6.0)
    ax.axhline(1.0, color="#666666", lw=1.2, alpha=0.8)
    ax.plot(x, omega_ratios, "o-", color="#2b6cb0", lw=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("measured / predicted", fontsize=axis_label_font)
    ax.set_ylim(0.98, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=tick_font)

    ax = axes[1]
    ax.set_title("Be scaling: Be ∝ μ^{-1} (ratio vs prediction)", fontsize=panel_title_font, pad=6.0)
    ax.axhline(1.0, color="#666666", lw=1.2, alpha=0.8)
    ax.plot(x, be_ratios, "o-", color="#c53030", lw=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("measured / predicted", fontsize=axis_label_font)
    ax.set_ylim(0.98, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=tick_font)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.920, bottom=0.095, hspace=0.64)

    out_pdf = out_dir / "molecular_isotopic_scaling.pdf"
    out_png = out_dir / "molecular_isotopic_scaling.png"
    prev_disable_normalize = os.environ.get("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE")
    os.environ["WAVEP_MPL_DISABLE_CANVAS_NORMALIZE"] = "1"
    try:
        with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
            fig.savefig(out_pdf)
            fig.savefig(out_png)
    finally:
        if prev_disable_normalize is None:
            os.environ.pop("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE", None)
        else:
            os.environ["WAVEP_MPL_DISABLE_CANVAS_NORMALIZE"] = prev_disable_normalize

    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": "Isotopic reduced-mass scaling check (WebBook diatomic constants)",
        "mass_model": mass_model,
        "note": (
            "Checks leading reduced-mass scaling (ωe∝μ^{-1/2}, B_e∝μ^{-1}). "
            "Prefer NIST primary-source-backed isotope masses when available; otherwise uses mass-number approximation."
        ),
        "rows": rows,
        "reference": {"slug": "h2", "mu_ref": mu_ref, "omega_e_ref_cm^-1": omega_ref, "B_e_ref_cm^-1": be_ref},
        "outputs": {"pdf": str(out_pdf), "png": str(out_png)},
    }
    out_json = out_dir / "molecular_isotopic_scaling_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_pdf}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
