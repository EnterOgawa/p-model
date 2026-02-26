from __future__ import annotations

import csv
import json
import math
from pathlib import Path


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


# 関数: `_solve_kappa_from_ere` の入出力契約と処理意図を定義する。

def _solve_kappa_from_ere(*, a_fm: float, r_fm: float) -> dict[str, float]:
    """
    Solve κ (fm^-1) from the truncated effective range expansion (ERE):

      k cot δ = -1/a + (r/2) k^2

    Bound-state pole at k = i κ (κ>0) implies:

      (r/2) κ^2 - κ + 1/a = 0

    There are two roots; the physical deuteron corresponds to the smaller κ.
    """
    # 条件分岐: `not (a_fm > 0 and r_fm > 0)` を満たす経路を評価する。
    if not (a_fm > 0 and r_fm > 0):
        raise ValueError("a_fm and r_fm must be positive")

    disc = 1.0 - 2.0 * r_fm / a_fm
    # 条件分岐: `disc <= 0.0` を満たす経路を評価する。
    if disc <= 0.0:
        raise ValueError(f"ERE discriminant <= 0: 1-2r/a = {disc}")

    s = math.sqrt(disc)
    kappa_small = (1.0 - s) / r_fm
    kappa_large = (1.0 + s) / r_fm
    # 条件分岐: `not (kappa_small > 0 and kappa_large > 0)` を満たす経路を評価する。
    if not (kappa_small > 0 and kappa_large > 0):
        raise ValueError("invalid roots (non-positive)")

    return {
        "disc": disc,
        "kappa_small_fm1": kappa_small,
        "kappa_large_fm1": kappa_large,
        "kappa_physical_fm1": kappa_small,
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs fixed by earlier steps.
    deut_path = root / "output" / "public" / "quantum" / "nuclear_binding_deuteron_metrics.json"
    # 条件分岐: `not deut_path.exists()` を満たす経路を評価する。
    if not deut_path.exists():
        raise SystemExit(
            "[fail] missing deuteron binding baseline metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_deuteron.py\n"
            f"Expected: {deut_path}"
        )

    deut = _load_json(deut_path)

    np_path = root / "output" / "public" / "quantum" / "nuclear_np_scattering_baseline_metrics.json"
    # 条件分岐: `not np_path.exists()` を満たす経路を評価する。
    if not np_path.exists():
        raise SystemExit(
            "[fail] missing np scattering baseline metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_np_scattering_baseline.py\n"
            f"Expected: {np_path}"
        )

    np_scatt = _load_json(np_path)

    # Constants (treat as exact here; consistent with other nuclear scripts).
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    hbarc_mev_fm = 197.326_980_4

    # Observed deuteron binding energy (primary fixed).
    b_mev_obj = (
        deut.get("derived", {}).get("binding_energy", {}).get("B_MeV", {})
        if isinstance(deut.get("derived", {}), dict)
        else {}
    )
    b_obs_mev = float(b_mev_obj.get("value", float("nan")))
    sigma_b_obs_mev = float(b_mev_obj.get("sigma", float("nan")))
    # 条件分岐: `not (math.isfinite(b_obs_mev) and b_obs_mev > 0 and math.isfinite(sigma_b_obs...` を満たす経路を評価する。
    if not (math.isfinite(b_obs_mev) and b_obs_mev > 0 and math.isfinite(sigma_b_obs_mev) and sigma_b_obs_mev >= 0):
        raise SystemExit("[fail] invalid/missing B_obs in nuclear_binding_deuteron_metrics.json derived.binding_energy.B_MeV")

    # Reduced mass (from CODATA mp,mn already fixed in deuteron baseline metrics).

    mu_kg = float(deut.get("derived", {}).get("reduced_mass_kg", float("nan")))
    # 条件分岐: `not (math.isfinite(mu_kg) and mu_kg > 0)` を満たす経路を評価する。
    if not (math.isfinite(mu_kg) and mu_kg > 0):
        raise SystemExit("[fail] invalid/missing reduced_mass_kg in nuclear_binding_deuteron_metrics.json derived.reduced_mass_kg")

    mu_c2_mev = (mu_kg * (c**2)) / (e_charge * 1e6)

    # Extract Eq.(18) and Eq.(19) (systematics proxy: analysis-dependent phase-shift sets).
    sets = np_scatt.get("np_low_energy_parameter_sets", [])
    # 条件分岐: `not isinstance(sets, list) or not sets` を満たす経路を評価する。
    if not isinstance(sets, list) or not sets:
        raise SystemExit("[fail] invalid np scattering metrics: np_low_energy_parameter_sets missing/empty")

    by_eq: dict[int, dict[str, object]] = {}
    for s in sets:
        # 条件分岐: `not isinstance(s, dict)` を満たす経路を評価する。
        if not isinstance(s, dict):
            continue

        try:
            eq = int(s.get("eq_label"))
        except Exception:
            continue

        by_eq[eq] = s

    need_eqs = [18, 19]
    missing = [eq for eq in need_eqs if eq not in by_eq]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise SystemExit(f"[fail] missing required eq sets in np metrics: {missing}")

    rows: list[dict[str, object]] = []
    preds_mev: list[float] = []
    for eq in need_eqs:
        s = by_eq[eq]
        a_t = float(s.get("a_t_fm", float("nan")))
        r_t = float(s.get("r_t_fm", float("nan")))
        v2t = s.get("v2t_fm3", None)
        v2t_val = float(v2t) if isinstance(v2t, (int, float)) else float("nan")

        # 条件分岐: `not (math.isfinite(a_t) and a_t > 0 and math.isfinite(r_t) and r_t > 0)` を満たす経路を評価する。
        if not (math.isfinite(a_t) and a_t > 0 and math.isfinite(r_t) and r_t > 0):
            raise SystemExit(f"[fail] invalid a_t/r_t for eq{eq}: a_t={a_t}, r_t={r_t}")

        sol = _solve_kappa_from_ere(a_fm=a_t, r_fm=r_t)
        kappa = float(sol["kappa_physical_fm1"])
        inv_kappa = 1.0 / kappa

        b_pred = (hbarc_mev_fm**2) * (kappa**2) / (2.0 * mu_c2_mev)
        delta_keV = (b_pred - b_obs_mev) * 1e3

        preds_mev.append(b_pred)
        rows.append(
            {
                "eq_label": int(eq),
                "kind": str(s.get("kind", "")),
                "a_t_fm": a_t,
                "r_t_fm": r_t,
                "v2t_fm3": v2t_val,
                "disc": float(sol["disc"]),
                "kappa_physical_fm1": kappa,
                "inv_kappa_fm": inv_kappa,
                "B_pred_MeV": b_pred,
                "Delta_B_pred_minus_obs_keV": delta_keV,
            }
        )

    b_min = min(preds_mev)
    b_max = max(preds_mev)
    b_mean = 0.5 * (b_min + b_max)
    sys_half_range = 0.5 * (b_max - b_min)

    delta_mean_keV = (b_mean - b_obs_mev) * 1e3
    sigma_tot_mev = math.sqrt((sigma_b_obs_mev**2) + (sys_half_range**2)) if sys_half_range > 0 else sigma_b_obs_mev
    z_total = (b_mean - b_obs_mev) / sigma_tot_mev if sigma_tot_mev > 0 else float("nan")
    z_sys = (b_mean - b_obs_mev) / sys_half_range if sys_half_range > 0 else float("nan")

    envelope_pass = (b_min <= b_obs_mev <= b_max)

    # CSV (freeze numerical comparison).
    out_csv = out_dir / "nuclear_binding_energy_frequency_mapping_deuteron_verification.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eq_label",
                "kind",
                "a_t_fm",
                "r_t_fm",
                "v2t_fm3",
                "kappa_physical_fm1",
                "inv_kappa_fm",
                "B_pred_MeV",
                "Delta_B_pred_minus_obs_keV",
                "B_obs_MeV",
                "sigma_B_obs_MeV",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    int(r["eq_label"]),
                    r["kind"],
                    f"{float(r['a_t_fm']):.12g}",
                    f"{float(r['r_t_fm']):.12g}",
                    "" if not math.isfinite(float(r["v2t_fm3"])) else f"{float(r['v2t_fm3']):.12g}",
                    f"{float(r['kappa_physical_fm1']):.12g}",
                    f"{float(r['inv_kappa_fm']):.12g}",
                    f"{float(r['B_pred_MeV']):.12g}",
                    f"{float(r['Delta_B_pred_minus_obs_keV']):.12g}",
                    f"{b_obs_mev:.12g}",
                    f"{sigma_b_obs_mev:.12g}",
                ]
            )

    # Plot

    import matplotlib.pyplot as plt

    x = [0, 1, 2]
    labels = ["Eq.(18)\nGWU/SAID", "Eq.(19)\nNijmegen", "mean\n±sys"]
    b_pred_list = [float(rows[0]["B_pred_MeV"]), float(rows[1]["B_pred_MeV"]), b_mean]
    b_err = [0.0, 0.0, sys_half_range]
    delta_keV_list = [
        float(rows[0]["Delta_B_pred_minus_obs_keV"]),
        float(rows[1]["Delta_B_pred_minus_obs_keV"]),
        delta_mean_keV,
    ]

    fig = plt.figure(figsize=(10.8, 4.2), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.errorbar(x, b_pred_list, yerr=b_err, fmt="o", color="tab:blue", capsize=4, lw=1.2)
    ax0.axhline(b_obs_mev, color="0.15", lw=1.2, ls="-", label="B_obs (CODATA)")
    ax0.fill_between([-0.4, 2.4], [b_min, b_min], [b_max, b_max], color="tab:orange", alpha=0.12, label="eq18–eq19 envelope")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("B (MeV)")
    ax0.set_title("Deuteron binding from triplet ERE pole (a_t, r_t)")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="lower right", fontsize=9)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar([0, 1, 2], delta_keV_list, color=["tab:blue", "tab:blue", "tab:orange"], alpha=0.85)
    ax1.axhline(0.0, color="0.15", lw=1.2, ls="-")
    ax1.axhspan(-1e3 * sys_half_range, 1e3 * sys_half_range, color="tab:orange", alpha=0.12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("ΔB = B_pred − B_obs (keV)")
    ax1.set_title("Residuals vs observation")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    # Summary annotation (avoid unicode in console; figure text can be unicode).
    verdict = "PASS" if envelope_pass else "FAIL"
    ax1.text(
        0.02,
        0.96,
        f"envelope: {verdict}\nΔmean={delta_mean_keV:+.1f} keV\nsys≈{1e3*sys_half_range:.1f} keV\nz_sys≈{z_sys:.2f}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.suptitle("Phase 7 / Step 7.13.17.4: deuteron numeric verification (ERE cross-check)", y=1.02)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.20, wspace=0.28)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_deuteron_verification.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Metrics (machine-readable freeze)
    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_deuteron_verification_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.17.4",
                "inputs": {
                    "deuteron_binding_metrics": {"path": str(deut_path), "sha256": _sha256(deut_path)},
                    "np_scattering_metrics": {"path": str(np_path), "sha256": _sha256(np_path)},
                },
                "constants": {
                    "c_m_per_s": c,
                    "e_C": e_charge,
                    "hbarc_MeV_fm": hbarc_mev_fm,
                },
                "observed": {
                    "B_obs_MeV": {"value": b_obs_mev, "sigma": sigma_b_obs_mev},
                    "mu_c2_MeV": mu_c2_mev,
                },
                "method": {
                    "ere_truncation": "k cot δ = -1/a_t + (r_t/2) k^2",
                    "bound_state_condition": "(r_t/2) κ^2 - κ + 1/a_t = 0  (κ>0; choose smaller root)",
                    "binding_from_kappa": "B_pred = (ħc)^2 κ^2 / (2 μ c^2)",
                    "notes": [
                        "Eq.(18) and Eq.(19) are treated as two analysis-dependent low-energy parameter sets (phase-shift analyses).",
                        "Their difference is used as a proxy for systematic uncertainty (not a statistical error bar).",
                        "Shape parameter v2t is ignored in the truncated ERE prediction; its effect is higher order for κ≈0.23 fm^-1.",
                    ],
                },
                "rows": rows,
                "summary": {
                    "B_pred_MeV": {
                        "min": b_min,
                        "max": b_max,
                        "mean": b_mean,
                        "sys_half_range": sys_half_range,
                    },
                    "residuals": {
                        "Delta_mean_keV": delta_mean_keV,
                        "z_total_proxy": z_total,
                        "z_sys_proxy": z_sys,
                    },
                    "acceptance": {
                        "envelope_pass": envelope_pass,
                        "criterion": "B_obs must lie within the envelope spanned by Eq.(18) and Eq.(19) predictions (ERE truncated).",
                    },
                },
                "outputs": {"png": str(out_png), "csv": str(out_csv)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    print(f"  {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

