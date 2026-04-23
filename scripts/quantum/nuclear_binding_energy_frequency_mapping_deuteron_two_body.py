"""
目的: 量子 topic の nuclear binding energy frequency mapping deuteron two body に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_japanese_localizer import enable_japanese_figure_localization
from scripts.utils.figure_locale_paths import localize_figure_output_path
from scripts.utils.plot_style import apply_wavep_figure_layout, get_wavep_font_size, install_wavep_font_profile

enable_japanese_figure_localization()

_PROFILE_NAME = "part3b_quantum_verification"

# 関数: `_sha256` の入出力契約と処理意図を定義する。
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


# 関数: `_load_json` の入出力契約と処理意図を定義する。

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_load_nist_codata_constants` の入出力契約と処理意図を定義する。

def _load_nist_codata_constants(*, root: Path) -> dict[str, dict[str, object]]:
    src_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    extracted = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted.exists()` を満たす経路を評価する。
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted CODATA constants.\n"
            "Run:\n"
            "  python -B scripts/quantum/fetch_nuclear_binding_sources.py\n"
            f"Expected: {extracted}"
        )

    payload = _load_json(extracted)
    consts = payload.get("constants")
    # 条件分岐: `not isinstance(consts, dict)` を満たす経路を評価する。
    if not isinstance(consts, dict):
        raise SystemExit(f"[fail] invalid extracted_values.json: constants is not a dict: {extracted}")

    return {k: v for k, v in consts.items() if isinstance(v, dict)}


# 関数: `_solve_bound_x` の入出力契約と処理意図を定義する。

def _solve_bound_x(*, kappa_fm1: float, r_fm: float) -> float:
    """
    Solve x in (pi/2, pi) for the s-wave square-well bound-state condition:

      k cot(kR) = -kappa, with k = x/R

    i.e.

      x cot x + kappa R = 0.
    """
    # 条件分岐: `not (math.isfinite(kappa_fm1) and kappa_fm1 > 0 and math.isfinite(r_fm) and r...` を満たす経路を評価する。
    if not (math.isfinite(kappa_fm1) and kappa_fm1 > 0 and math.isfinite(r_fm) and r_fm > 0):
        raise ValueError("invalid kappa or R")

    lo = (math.pi / 2.0) + 1e-7
    hi = math.pi - 1e-7

    # 関数: `f` の入出力契約と処理意図を定義する。
    def f(x: float) -> float:
        return (x / math.tan(x)) + (kappa_fm1 * r_fm)

    flo = f(lo)
    fhi = f(hi)
    # 条件分岐: `not (flo > 0 and fhi < 0)` を満たす経路を評価する。
    if not (flo > 0 and fhi < 0):
        raise ValueError(f"no bracket for bound x: f(lo)={flo}, f(hi)={fhi}, kappaR={kappa_fm1*r_fm}")

    for _ in range(96):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        # 条件分岐: `fmid == 0 or (hi - lo) < 1e-15` を満たす経路を評価する。
        if fmid == 0 or (hi - lo) < 1e-15:
            return mid

        # 条件分岐: `fmid > 0` を満たす経路を評価する。

        if fmid > 0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# 関数: `_square_well_from_r` の入出力契約と処理意図を定義する。

def _square_well_from_r(*, mu_mev: float, b_mev: float, r_fm: float, hbarc_mev_fm: float) -> dict[str, float]:
    """
    Given B (fixed) and R, solve the well depth V0 by the s-wave bound-state condition.

    Returns: V0 (MeV), x (dimensionless), k (fm^-1), kappa (fm^-1).
    """
    # 条件分岐: `not (mu_mev > 0 and b_mev > 0 and r_fm > 0 and hbarc_mev_fm > 0)` を満たす経路を評価する。
    if not (mu_mev > 0 and b_mev > 0 and r_fm > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid inputs")

    kappa = math.sqrt(2.0 * mu_mev * b_mev) / hbarc_mev_fm
    x = _solve_bound_x(kappa_fm1=kappa, r_fm=r_fm)
    k = x / r_fm
    v0 = b_mev + (hbarc_mev_fm**2) * (k**2) / (2.0 * mu_mev)
    return {"V0_mev": float(v0), "x": float(x), "k_fm1": float(k), "kappa_fm1": float(kappa)}


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    install_wavep_font_profile(profile_name=_PROFILE_NAME)
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    consts = _load_nist_codata_constants(root=root)
    need = ["mp", "mn", "md", "rd"]
    for k in need:
        # 条件分岐: `k not in consts` を満たす経路を評価する。
        if k not in consts:
            raise SystemExit(f"[fail] missing constant {k!r} in extracted_values.json")

    mp = float(consts["mp"]["value_si"])
    mn = float(consts["mn"]["value_si"])
    md = float(consts["md"]["value_si"])
    sigma_mp = float(consts["mp"]["sigma_si"])
    sigma_mn = float(consts["mn"]["sigma_si"])
    sigma_md = float(consts["md"]["sigma_si"])
    rd_m = float(consts["rd"]["value_si"])

    # Exact SI constants:
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    h = 6.626_070_15e-34
    hbar = h / (2.0 * math.pi)

    # Binding energy (CODATA baseline)
    dm = (mp + mn - md)
    sigma_dm = math.sqrt(sigma_mp**2 + sigma_mn**2 + sigma_md**2)
    b_j = dm * (c**2)
    sigma_b_j = sigma_dm * (c**2)
    b_mev = b_j / (1e6 * e_charge)
    sigma_b_mev = sigma_b_j / (1e6 * e_charge)

    # Reduced mass
    mu_kg = (mp * mn) / (mp + mn)
    mu_mev = (mu_kg * (c**2)) / (1e6 * e_charge)

    # Tail scale (kappa) and frequency mapping
    kappa_si = math.sqrt(2.0 * mu_kg * abs(b_j)) / hbar if b_j > 0 else float("nan")
    inv_kappa_fm = (1.0 / kappa_si) * 1e15 if (math.isfinite(kappa_si) and kappa_si > 0) else float("nan")
    delta_omega_per_s = (b_j / hbar) if (b_j > 0) else float("nan")
    j_freq_per_s = 0.5 * delta_omega_per_s if math.isfinite(delta_omega_per_s) else float("nan")

    # Scale proxies used in nuclear steps
    qcd_metrics_path = root / "output" / "public" / "quantum" / "qcd_hadron_masses_baseline_metrics.json"
    qcd_metrics = _load_json(qcd_metrics_path) if qcd_metrics_path.exists() else {}
    hbarc_mev_fm = float(qcd_metrics.get("constants", {}).get("hbar_c_mev_fm", 197.3269804))

    lambda_pi_fm: float | None = None
    # 条件分岐: `isinstance(qcd_metrics.get("rows"), list)` を満たす経路を評価する。
    if isinstance(qcd_metrics.get("rows"), list):
        for row in qcd_metrics["rows"]:
            # 条件分岐: `isinstance(row, dict) and row.get("label") == "π±"` を満たす経路を評価する。
            if isinstance(row, dict) and row.get("label") == "π±":
                try:
                    lambda_pi_fm = float(row.get("compton_lambda_fm"))
                except Exception:
                    lambda_pi_fm = None

                break

    rd_fm = rd_m * 1e15

    ranges: list[dict[str, object]] = [
        {"label": "R = λπ (π± Compton)", "R_fm": lambda_pi_fm},
        {"label": "R = r_d (charge rms)", "R_fm": rd_fm},
        {"label": "R = 2.0 fm (proxy)", "R_fm": 2.0},
        {"label": "R = 1/κ (tail scale)", "R_fm": inv_kappa_fm},
    ]

    fits: list[dict[str, object]] = []
    for r in ranges:
        r_fm = r.get("R_fm")
        # 条件分岐: `r_fm is None` を満たす経路を評価する。
        if r_fm is None:
            continue

        r_fm = float(r_fm)
        # 条件分岐: `not (math.isfinite(r_fm) and r_fm > 0)` を満たす経路を評価する。
        if not (math.isfinite(r_fm) and r_fm > 0):
            continue

        sw = _square_well_from_r(mu_mev=mu_mev, b_mev=b_mev, r_fm=r_fm, hbarc_mev_fm=hbarc_mev_fm)
        fits.append(
            {
                "label": str(r.get("label")),
                "R_fm": r_fm,
                "kappaR": sw["kappa_fm1"] * r_fm,
                "x": sw["x"],
                "k_fm1": sw["k_fm1"],
                "V0_mev": sw["V0_mev"],
            }
        )

    # Plot

    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig = plt.figure(figsize=(16.6, 7.4), dpi=170)
    apply_wavep_figure_layout(fig, template="part2_side_by_side")
    fig.set_size_inches(fig.get_figwidth(), 5.9, forward=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.40], wspace=0.20)
    panel_title_font = get_wavep_font_size("title") * 0.80
    axis_label_font = get_wavep_font_size("axis")
    tick_font = get_wavep_font_size("tick")
    note_font = get_wavep_font_size("note") * 0.96
    suptitle_font = get_wavep_font_size("suptitle") + 1.0

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.text(
        0.0,
        1.0,
        "deuteron (pn) two-body: bound-state scales (frozen)",
        ha="left",
        va="top",
        fontsize=panel_title_font,
        weight="bold",
        transform=ax0.transAxes,
    )
    ax0.text(
        0.0,
        0.86,
        (
            f"B = {b_mev:.6f} ± {sigma_b_mev:.6f} MeV\n"
            "(CODATA mass defect)\n"
            f"Δω = B/ħ ≈ {delta_omega_per_s:.3e} s^-1\n"
            f"J (2-mode I/F) = Δω/2 ≈ {j_freq_per_s:.3e} s^-1\n"
            f"1/κ (tail) ≈ {inv_kappa_fm:.3f} fm\n"
            f"r_d ≈ {rd_fm:.3f} fm"
        ),
        ha="left",
        va="top",
        fontsize=note_font,
        transform=ax0.transAxes,
        bbox={"boxstyle": "round,pad=0.34", "facecolor": "white", "edgecolor": "0.85"},
    )
    ax0.text(
        -0.035,
        0.30,
        (
            "Square-well example (s-wave)\n"
            "V(r) = -V0 for r < R, and 0 for r ≥ R\n"
            "k cot(kR) = -κ,   κ = sqrt(2μB)/ħ\n"
            "\n"
            "Operational I/F for the standing-wave boundary condition.\n"
            "It does not assert that the nuclear force is literally\n"
            "a square-well potential."
        ),
        ha="left",
        va="top",
        fontsize=note_font,
        transform=ax0.transAxes,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "0.87", "alpha": 0.95},
    )

    ax1 = fig.add_subplot(gs[0, 1])
    xs = [float(f["R_fm"]) for f in fits]
    ys = [float(f["V0_mev"]) for f in fits]
    labels = [str(f["label"]) for f in fits]

    ax1.plot(xs, ys, marker="o", lw=1.8)
    y_max = max(ys) if ys else 0.0
    for x, y, lab in zip(xs, ys, labels):
        # Top-side overlap with title/grid text is avoided by shifting labels downward.
        y_shift_pts = -16 if y > (y_max - 4.0) else -8
        rot = 10 if y_shift_pts <= -16 else 15
        ax1.annotate(
            lab.replace("R = ", ""),
            xy=(x, y),
            xytext=(4, y_shift_pts),
            textcoords="offset points",
            fontsize=note_font,
            ha="left",
            va="top",
            rotation=rot,
        )

    ax1.set_xlabel("well range R (fm)", fontsize=axis_label_font)
    ax1.set_ylabel("required depth V0 (MeV)", fontsize=axis_label_font)
    ax1.set_title("Square-well depth required to support B (illustration)", fontsize=panel_title_font, pad=6.0)
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax1.tick_params(axis="both", labelsize=tick_font)

    fig.suptitle("deuteron Δω mapping via 2-body boundary condition", y=0.982, fontsize=suptitle_font)
    fig.subplots_adjust(left=0.070, right=0.985, top=0.875, bottom=0.115, wspace=0.20)

    out_pdf = localize_figure_output_path(out_dir / "nuclear_binding_energy_frequency_mapping_deuteron_two_body.pdf", root=root)
    out_png = localize_figure_output_path(out_dir / "nuclear_binding_energy_frequency_mapping_deuteron_two_body.png", root=root)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
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

    # Sources / traceability
    codata_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    codata_manifest = codata_dir / "manifest.json"
    codata_extracted = codata_dir / "extracted_values.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.13.17.2",
        "sources": [
            {
                "dataset": "NIST Cuu CODATA constants (mp,mn,md,rd)",
                "local_manifest": str(codata_manifest),
                "local_manifest_sha256": _sha256(codata_manifest) if codata_manifest.exists() else None,
                "local_extracted": str(codata_extracted),
                "local_extracted_sha256": _sha256(codata_extracted) if codata_extracted.exists() else None,
            },
            {
                "dataset": "PDG RPP 2024 mass baseline (for λπ proxy and ħc constant)",
                "local_metrics": str(qcd_metrics_path) if qcd_metrics_path.exists() else None,
                "local_metrics_sha256": _sha256(qcd_metrics_path) if qcd_metrics_path.exists() else None,
            },
        ],
        "constants": {
            "c_m_per_s": c,
            "h_J_s": h,
            "hbar_J_s": hbar,
            "e_C": e_charge,
            "hbarc_MeV_fm": hbarc_mev_fm,
        },
        "derived": {
            "binding_energy": {
                "B_J": {"value": b_j, "sigma": sigma_b_j},
                "B_MeV": {"value": b_mev, "sigma": sigma_b_mev},
            },
            "reduced_mass_mu_c2_MeV": mu_mev,
            "kappa_1_per_m": kappa_si,
            "inv_kappa_fm": inv_kappa_fm,
            "deuteron_charge_rms_radius_fm": rd_fm,
            "delta_omega_per_s": delta_omega_per_s,
            "two_mode_J_per_s": j_freq_per_s,
            "lambda_pi_pm_fm": lambda_pi_fm,
        },
        "square_well_example": {
            "condition": "k cot(kR) = -kappa (s-wave; bound state)",
            "fits_from_R": fits,
            "notes": [
                "This is an illustrative operational I/F for the boundary condition; it does not claim a square-well force model.",
                "R is treated as a knob (e.g., λπ, r_d, proxy R0, tail scale) to visualize how the required depth scales.",
            ],
        },
        "outputs": {"pdf": str(out_pdf), "png": str(out_png)},
    }

    out_json = localize_figure_output_path(
        out_dir / "nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json",
        root=root,
    )
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
