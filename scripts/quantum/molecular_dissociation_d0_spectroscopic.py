from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_japanese_localizer import enable_japanese_figure_localization
from scripts.utils.figure_locale_paths import localize_figure_output_path
from scripts.utils.plot_style import apply_wavep_figure_layout, get_wavep_font_size, install_wavep_font_profile

enable_japanese_figure_localization()

_PROFILE_NAME = "part3b_quantum_verification"

# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_cm_inv_to_hz` の入出力契約と処理意図を定義する。

def _cm_inv_to_hz(cm_inv: float) -> float:
    c = 299_792_458.0
    return c * (cm_inv * 100.0)


# 関数: `_cm_inv_to_ev` の入出力契約と処理意図を定義する。

def _cm_inv_to_ev(cm_inv: float) -> float:
    h = 6.626_070_15e-34  # exact (SI)
    e_charge = 1.602_176_634e-19  # exact (J/eV)
    return (h * _cm_inv_to_hz(cm_inv)) / e_charge


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    install_wavep_font_profile(profile_name=_PROFILE_NAME)
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = (
        root
        / "data"
        / "quantum"
        / "sources"
        / "molecular_dissociation_d0_spectroscopic"
        / "extracted_values.json"
    )
    # 条件分岐: `not src.exists()` を満たす経路を評価する。
    if not src.exists():
        raise SystemExit(
            f"[fail] missing D0 cache: {src}\n"
            "Run: python -B scripts/quantum/fetch_molecular_dissociation_d0_spectroscopic.py"
        )

    j = _read_json(src)
    sources = j.get("sources")
    # 条件分岐: `not isinstance(sources, list) or not sources` を満たす経路を評価する。
    if not isinstance(sources, list) or not sources:
        raise SystemExit(f"[fail] invalid sources in: {src}")

    rows: list[dict[str, Any]] = []
    for rec in sources:
        # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
        if not isinstance(rec, dict):
            continue

        mol = str(rec.get("molecule") or "").strip()
        # 条件分岐: `not mol` を満たす経路を評価する。
        if not mol:
            continue

        d0 = rec.get("d0_cm^-1")
        # 条件分岐: `not isinstance(d0, (int, float))` を満たす経路を評価する。
        if not isinstance(d0, (int, float)):
            continue

        unc = rec.get("d0_unc_cm^-1")
        unc_f = None if unc is None else float(unc)
        n = rec.get("rotational_N")
        n_i = None if n is None else int(n)

        d0_f = float(d0)
        rows.append(
            {
                "molecule": mol,
                "rotational_N": n_i,
                "d0_cm^-1": d0_f,
                "d0_unc_cm^-1": unc_f,
                "d0_eV": _cm_inv_to_ev(d0_f),
                "d0_unc_eV": (None if unc_f is None else _cm_inv_to_ev(unc_f)),
                "source": rec.get("source"),
            }
        )

    # Stable ordering.

    order = {"H2": 0, "HD": 1, "D2": 2}
    rows.sort(key=lambda r: order.get(str(r["molecule"]), 99))

    # ---- Figure ----
    labels: list[str] = []
    y: list[float] = []
    yerr: list[float] = []
    for r in rows:
        lab = str(r["molecule"])
        n = r.get("rotational_N")
        # 条件分岐: `n is not None` を満たす経路を評価する。
        if n is not None:
            lab = f"{lab} (N={n})"

        labels.append(lab)
        y.append(float(r["d0_eV"]))
        yerr.append(0.0 if r["d0_unc_eV"] is None else float(r["d0_unc_eV"]))

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(1, 1, figsize=(10.4, 4.9), dpi=190)
    apply_wavep_figure_layout(fig, template="part2_single_panel_legend_bottom")
    title_font = get_wavep_font_size("title") * 0.92
    axis_label_font = get_wavep_font_size("axis") * 0.94
    tick_font = get_wavep_font_size("tick") * 0.94
    note_font = get_wavep_font_size("note") * 0.88
    ax.set_title("Spectroscopic D0 baseline at 0 K", fontsize=title_font, pad=7.0)
    ax.bar(labels, y, color=["#2b6cb0", "#805ad5", "#c53030"], alpha=0.92)
    ax.errorbar(labels, y, yerr=yerr, fmt="none", ecolor="#222222", elinewidth=1.2, capsize=4)
    ax.set_ylabel("D0 at 0 K [eV per molecule]", fontsize=axis_label_font)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=tick_font)

    for i, r in enumerate(rows):
        d0_cm = float(r["d0_cm^-1"])
        ax.text(i, y[i] + 0.008, f"{d0_cm:.5f} cm⁻¹", ha="center", va="bottom", fontsize=note_font)

    fig.text(
        0.11,
        0.015,
        "Baseline uses spectroscopic D0 at 0 K.\n"
        "Do not mix with 298 K dissociation enthalpy.\n"
        "H2 value is ortho-H2 (N=1) per the primary source.",
        fontsize=note_font,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.24)

    out_pdf = localize_figure_output_path(out_dir / "molecular_dissociation_d0_spectroscopic.pdf", root=root)
    out_png = localize_figure_output_path(out_dir / "molecular_dissociation_d0_spectroscopic.png", root=root)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": "Spectroscopic dissociation energy D0 (0 K; fixed baseline)",
        "note": (
            "This output fixes spectroscopic dissociation energies D0 at 0 K as an independent baseline for future "
            "P-model molecular-binding derivations. Track rotational state (N) where applicable."
        ),
        "inputs": {"extracted_values": str(src)},
        "rows": rows,
        "outputs": {"pdf": str(out_pdf), "png": str(out_png)},
    }
    out_json = localize_figure_output_path(out_dir / "molecular_dissociation_d0_spectroscopic_metrics.json", root=root)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_pdf}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
