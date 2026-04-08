"""
目的: 量子 topic の nuclear binding deuteron に対応する公開図・表・監査指標を再生成する。
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


# 関数: `_configure_japanese_font` の入出力契約と処理意図を定義する。

def _configure_japanese_font() -> None:
    import matplotlib as mpl
    from scripts.utils.plot_style import install_wavep_cjk_font_override

    install_wavep_cjk_font_override(preferred_name="Noto Sans CJK JP")
    mpl.rcParams["axes.unicode_minus"] = False


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

    payload = json.loads(extracted.read_text(encoding="utf-8"))
    consts = payload.get("constants")
    # 条件分岐: `not isinstance(consts, dict)` を満たす経路を評価する。
    if not isinstance(consts, dict):
        raise SystemExit(f"[fail] invalid extracted_values.json: constants is not a dict: {extracted}")

    return {k: v for k, v in consts.items() if isinstance(v, dict)}


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
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
    sigma_rd_m = float(consts["rd"]["sigma_si"])

    # Exact SI constants:
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    h = 6.626_070_15e-34
    hbar = h / (2.0 * math.pi)

    # Deuteron binding energy from mass defect: B = (m_p + m_n - m_d) c^2
    dm = (mp + mn - md)
    sigma_dm = math.sqrt(sigma_mp**2 + sigma_mn**2 + sigma_md**2)
    b_j = dm * (c**2)
    sigma_b_j = sigma_dm * (c**2)
    b_mev = b_j / (1e6 * e_charge)
    sigma_b_mev = sigma_b_j / (1e6 * e_charge)

    # Reduced mass μ and bound-state tail length scale 1/κ where B = ħ^2 κ^2 / (2μ)
    mu = (mp * mn) / (mp + mn)
    kappa = math.sqrt(2.0 * mu * abs(b_j)) / hbar if b_j > 0 else float("nan")
    inv_kappa_m = (1.0 / kappa) if (kappa and math.isfinite(kappa) and kappa > 0) else float("nan")
    inv_kappa_fm = inv_kappa_m * 1e15

    rd_fm = rd_m * 1e15
    sigma_rd_fm = sigma_rd_m * 1e15

    # Effective potential scale if one writes (semi-classically) V = m φ:
    # |φ|/c^2 ~ B / (m c^2). This is only a bookkeeping number here.
    phi_over_c2 = b_j / (mu * c**2) if mu > 0 else float("nan")

    _configure_japanese_font()

    # Plot
    import matplotlib.pyplot as plt
    from scripts.utils.plot_style import apply_wavep_figure_layout, get_wavep_font_size
    title_font = get_wavep_font_size("title") * 0.88
    axis_font = get_wavep_font_size("axis")
    tick_font = get_wavep_font_size("tick")
    legend_font = get_wavep_font_size("legend")
    note_font = get_wavep_font_size("note")
    suptitle_font = get_wavep_font_size("suptitle") + 2.0

    fig = plt.figure(dpi=170)
    apply_wavep_figure_layout(fig, template="part2_two_panel_spacious")
    fig.set_size_inches(fig.get_figwidth(), 5.95, forward=True)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.915, bottom=0.105, hspace=0.44)
    gs = fig.add_gridspec(2, 1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.errorbar([0.0], [b_mev], yerr=[sigma_b_mev], fmt="o", capsize=5, lw=1.8)
    ax0.set_xticks([0.0])
    ax0.set_xticklabels(["重水素"])
    ax0.set_ylabel("束縛エネルギー B (MeV)", fontsize=axis_font)
    ax0.set_title("質量欠損ベースライン（CODATA/NIST）", fontsize=title_font, pad=5.0)
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax0.tick_params(axis="both", labelsize=tick_font)
    ax0.text(
        0.02,
        0.98,
        (
            "B = (m_p + m_n − m_d)c²\n"
            f"B ≈ {b_mev:.6f} ± {sigma_b_mev:.6f} MeV\n"
            f"|φ|/c²（記帳値）≈ {abs(phi_over_c2):.3e}"
        ),
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=note_font,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    ax1 = fig.add_subplot(gs[1, 0])
    x = [0.0, 1.0]
    y = [rd_fm, inv_kappa_fm]
    yerr = [sigma_rd_fm, 0.0]
    ax1.errorbar([x[0]], [y[0]], yerr=[yerr[0]], fmt="o", capsize=5, lw=1.8, label="r_d（電荷rms半径）")
    ax1.plot([x[1]], [y[1]], marker="s", lw=0.0, label="Bから得る 1/κ（テール尺度）")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["r_d", "1/κ"])
    ax1.set_ylabel("長さスケール (fm)", fontsize=axis_font)
    ax1.set_title("サイズ制約（半径と束縛テール）", fontsize=title_font, pad=5.0)
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax1.tick_params(axis="both", labelsize=tick_font)
    ax1.legend(frameon=True, fontsize=legend_font, loc="upper right")
    ax1.text(
        0.02,
        0.02,
        "κ = sqrt(2 μ B) / ħ",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
        fontsize=note_font,
    )

    fig.suptitle("重水素の核ベースライン（観測量固定）", y=0.992, fontsize=suptitle_font)

    out_png = out_dir / "nuclear_binding_deuteron.png"
    out_pdf = out_png.with_suffix(".pdf")
    prev_disable_normalize = os.environ.get("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE")
    os.environ["WAVEP_MPL_DISABLE_CANVAS_NORMALIZE"] = "1"
    try:
        with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
            fig.savefig(out_pdf)
            fig.savefig(out_png, dpi=180)
    finally:
        if prev_disable_normalize is None:
            os.environ.pop("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE", None)
        else:
            os.environ["WAVEP_MPL_DISABLE_CANVAS_NORMALIZE"] = prev_disable_normalize

    plt.close(fig)

    # Sources and traceability
    src_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    manifest = src_dir / "manifest.json"
    extracted = src_dir / "extracted_values.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.9",
        "sources": [
            {
                "dataset": "NIST Cuu CODATA constants (mp,mn,md,rd)",
                "local_manifest": str(manifest),
                "local_manifest_sha256": _sha256(manifest) if manifest.exists() else None,
                "local_extracted": str(extracted),
                "local_extracted_sha256": _sha256(extracted) if extracted.exists() else None,
            }
        ],
        "assumptions": [
            "Independent uncertainties for mp,mn,md when propagating σ(B) (no covariance provided).",
            "The length scale 1/κ is the bound-state tail scale from a single-channel nonrelativistic estimate.",
            "The 'phi_over_c2' value is a bookkeeping ratio B/(μc^2) under the minimal coupling V=mφ; not a derived nuclear field yet.",
        ],
        "constants_from_nist_codata": {
            "mp_kg": {"value": mp, "sigma": sigma_mp},
            "mn_kg": {"value": mn, "sigma": sigma_mn},
            "md_kg": {"value": md, "sigma": sigma_md},
            "rd_m": {"value": rd_m, "sigma": sigma_rd_m},
        },
        "derived": {
            "mass_defect_kg": {"value": dm, "sigma": sigma_dm},
            "binding_energy": {
                "B_J": {"value": b_j, "sigma": sigma_b_j},
                "B_MeV": {"value": b_mev, "sigma": sigma_b_mev},
            },
            "reduced_mass_kg": mu,
            "kappa_1_per_m": kappa,
            "inv_kappa_m": inv_kappa_m,
            "inv_kappa_fm": inv_kappa_fm,
            "deuteron_charge_rms_radius_fm": {"value": rd_fm, "sigma": sigma_rd_fm},
            "phi_over_c2_bookkeeping": phi_over_c2,
        },
        "falsification": {
            "acceptance_criteria": [
                "Any proposed P-model nuclear effective equation/potential must support a pn bound state (deuteron) with B within 5σ of this baseline value.",
                "Any proposed P-model prediction for the deuteron size scale must be compatible with r_d and the tail scale 1/κ (within stated model assumptions).",
            ]
        },
        "outputs": {"pdf": str(out_pdf), "png": str(out_png)},
        "notes": [
            "This step fixes nuclear baseline observables and traceable primary sources; it does not claim a first-principles derivation of nuclear forces yet.",
            "Next: extend primary data to np scattering (a_t, r_t, phase shifts) and derive the effective nuclear-scale constraint from the P-field model (to avoid 'just insert φ' criticism).",
        ],
    }

    out_json = out_dir / "nuclear_binding_deuteron_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
