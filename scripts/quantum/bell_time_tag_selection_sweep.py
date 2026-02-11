from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    # Simulation size (keep moderate for Windows; deterministic seed for reproducibility).
    n_events: int = 250_000
    seed: int = 0

    # "Time-tag" modulation strength; larger d makes delays more peaked near 0,
    # amplifying post-selection effects for small coincidence windows.
    delay_power_d: float = 4.0

    # Coincidence window sweep (dimensionless in this toy).
    windows: tuple[float, ...] = (
        1e-4,
        2e-4,
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
        1e-1,
        2e-1,
        5e-1,
        1.0,
    )

    # CHSH settings (photon polarization convention; degrees -> radians).
    # Common "quantum-optimal" set: 0°, 45°, 22.5°, 67.5°.
    a_deg: float = 0.0
    ap_deg: float = 45.0
    b_deg: float = 22.5
    bp_deg: float = 67.5


def _outcome(phi: np.ndarray, theta: float) -> np.ndarray:
    # Deterministic local output for polarization-like settings.
    # Using 2*(phi-theta) keeps the periodicity consistent with linear polarizers.
    return np.where(np.cos(2.0 * (phi - theta)) >= 0.0, 1, -1).astype(np.int8)


def _time_tag(rng: np.random.Generator, phi: np.ndarray, theta: float, *, d: float) -> np.ndarray:
    # Toy time-tag model: setting-dependent delay distribution.
    # t in [0,1) scaled by |sin(2*(phi-theta))|^d.
    u = rng.random(phi.shape[0])
    return u * (np.abs(np.sin(2.0 * (phi - theta))) ** d)


def _sweep_pair(
    rng: np.random.Generator,
    phi: np.ndarray,
    theta_a: float,
    theta_b: float,
    *,
    d: float,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a = _outcome(phi, theta_a)
    b = _outcome(phi, theta_b)
    prod = (a * b).astype(np.int16)

    t_a = _time_tag(rng, phi, theta_a, d=d)
    t_b = _time_tag(rng, phi, theta_b, d=d)
    dt = np.abs(t_a - t_b)

    e = np.empty(windows.shape[0], dtype=float)
    acc = np.empty(windows.shape[0], dtype=float)
    for i, w in enumerate(windows):
        m = dt < w
        n = int(np.count_nonzero(m))
        if n == 0:
            e[i] = float("nan")
            acc[i] = 0.0
            continue
        e[i] = float(prod[m].mean())
        acc[i] = n / float(phi.shape[0])
    return e, acc


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    windows = np.array(cfg.windows, dtype=float)
    a = np.deg2rad(cfg.a_deg)
    ap = np.deg2rad(cfg.ap_deg)
    b = np.deg2rad(cfg.b_deg)
    bp = np.deg2rad(cfg.bp_deg)

    # Shared hidden variable per emitted pair.
    phi = rng.random(cfg.n_events) * 2.0 * np.pi

    # Each setting-pair uses independent local time-tag randomness (still local).
    # Baseline (trial-based): no coincidence selection, use all indexed pairs.
    e_ab_all = float((_outcome(phi, a) * _outcome(phi, b)).mean())
    e_abp_all = float((_outcome(phi, a) * _outcome(phi, bp)).mean())
    e_apb_all = float((_outcome(phi, ap) * _outcome(phi, b)).mean())
    e_apbp_all = float((_outcome(phi, ap) * _outcome(phi, bp)).mean())
    s_all = float(abs(e_ab_all - e_abp_all + e_apb_all + e_apbp_all))

    e_ab, acc_ab = _sweep_pair(rng, phi, a, b, d=cfg.delay_power_d, windows=windows)
    e_abp, acc_abp = _sweep_pair(rng, phi, a, bp, d=cfg.delay_power_d, windows=windows)
    e_apb, acc_apb = _sweep_pair(rng, phi, ap, b, d=cfg.delay_power_d, windows=windows)
    e_apbp, acc_apbp = _sweep_pair(rng, phi, ap, bp, d=cfg.delay_power_d, windows=windows)

    # CHSH combination (one standard form).
    s = np.abs(e_ab - e_abp + e_apb + e_apbp)

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), dpi=150, sharex=True)

    ax = axes[0]
    ax.plot(windows, s, marker="o", lw=1.6, label="S(Δt) (toy; post-selection)")
    ax.axhline(s_all, color="0.15", lw=1.0, ls="-.", label=f"trial-based (no selection): S={s_all:.3f}")
    ax.axhline(2.0, color="0.3", lw=1.0, ls="--", label="Bell bound (2)")
    ax.axhline(2.0 * np.sqrt(2.0), color="0.3", lw=1.0, ls=":", label="2√2 (reference)")
    ax.set_xscale("log")
    ax.set_ylabel("CHSH S")
    ax.set_title("Toy: coincidence window can change S (local model + time-tag selection)")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    ax = axes[1]
    ax.plot(windows, acc_ab, marker="o", lw=1.2, label="accept frac (a,b)")
    ax.plot(windows, acc_abp, marker="o", lw=1.2, label="accept frac (a,b')")
    ax.plot(windows, acc_apb, marker="o", lw=1.2, label="accept frac (a',b)")
    ax.plot(windows, acc_apbp, marker="o", lw=1.2, label="accept frac (a',b')")
    ax.set_xscale("log")
    ax.set_xlabel("Δt (coincidence window; dimensionless in this toy)")
    ax.set_ylabel("accepted fraction")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=2)

    fig.tight_layout()

    out_png = out_dir / "bell_time_tag_selection_sweep.png"
    out_json = out_dir / "bell_time_tag_selection_sweep_metrics.json"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "config": {
            "n_events": cfg.n_events,
            "seed": cfg.seed,
            "delay_power_d": cfg.delay_power_d,
            "windows": list(map(float, windows.tolist())),
            "angles_deg": {"a": cfg.a_deg, "ap": cfg.ap_deg, "b": cfg.b_deg, "bp": cfg.bp_deg},
            "chsh_definition": "S=|E(a,b)-E(a,b')+E(a',b)+E(a',b')|",
        },
        "results": {
            "trial_based": {
                "E_ab": e_ab_all,
                "E_abp": e_abp_all,
                "E_apb": e_apb_all,
                "E_apbp": e_apbp_all,
                "S": s_all,
            },
            "E_ab": e_ab.tolist(),
            "E_abp": e_abp.tolist(),
            "E_apb": e_apb.tolist(),
            "E_apbp": e_apbp.tolist(),
            "S": s.tolist(),
            "accept_frac": {
                "ab": acc_ab.tolist(),
                "abp": acc_abp.tolist(),
                "apb": acc_apb.tolist(),
                "apbp": acc_apbp.tolist(),
            },
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "Toy model for logical structure only: local outputs + local time-tags + coincidence selection.",
            "Not evidence for P-model; used to define what Step 7.4 must check on real data.",
        ],
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
