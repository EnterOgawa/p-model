from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DemoConfig:
    L: float = 1.0
    n_x: int = 1200
    c: float = 1.0
    dt_cfl: float = 0.45
    steps: int = 1600
    snapshot_steps: tuple[int, ...] = (0, 400, 800, 1200, 1600)
    packet_center: float = 0.25
    packet_sigma: float = 0.03
    packet_k: float = 120.0  # rad / length


def _simulate_wave_packet(cfg: DemoConfig) -> dict:
    # 1D wave equation: u_tt = c^2 u_xx on [0,L] with Dirichlet boundaries u(0)=u(L)=0.
    # Leapfrog scheme.
    x = np.linspace(0.0, cfg.L, cfg.n_x, dtype=float)
    dx = float(x[1] - x[0])
    dt = float(cfg.dt_cfl * dx / cfg.c)
    r2 = (cfg.c * dt / dx) ** 2

    # Initial displacement: Gaussian packet modulated by cosine.
    f = np.exp(-0.5 * ((x - cfg.packet_center) / cfg.packet_sigma) ** 2) * np.cos(cfg.packet_k * x)
    f[0] = 0.0
    f[-1] = 0.0

    # Right-moving packet approx: u_t(x,0) = -c * f'(x)
    f_x = np.gradient(f, dx)
    u0 = f.copy()
    v0 = -cfg.c * f_x
    v0[0] = 0.0
    v0[-1] = 0.0

    # First step via Taylor: u1 = u0 + dt*v0 + 0.5*dt^2*c^2*u_xx
    u_xx0 = np.zeros_like(u0)
    u_xx0[1:-1] = (u0[2:] - 2.0 * u0[1:-1] + u0[:-2]) / (dx * dx)
    u1 = u0 + dt * v0 + 0.5 * (dt * dt) * (cfg.c * cfg.c) * u_xx0
    u1[0] = 0.0
    u1[-1] = 0.0

    snapshots: dict[int, np.ndarray] = {}
    if 0 in cfg.snapshot_steps:
        snapshots[0] = u0.copy()

    u_prev = u0
    u = u1
    for step in range(1, cfg.steps + 1):
        u_next = np.zeros_like(u)
        u_next[1:-1] = (
            2.0 * u[1:-1]
            - u_prev[1:-1]
            + r2 * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        )
        u_next[0] = 0.0
        u_next[-1] = 0.0

        if step in cfg.snapshot_steps:
            snapshots[step] = u_next.copy()

        u_prev, u = u, u_next

    return {
        "x": x,
        "dx": dx,
        "dt": dt,
        "snapshots": snapshots,
    }


def _plot(cfg: DemoConfig, sim: dict, *, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    x: np.ndarray = sim["x"]
    L = cfg.L

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150)

    # Left: eigenmodes (Dirichlet).
    ax = axes[0]
    for n in (1, 2, 3):
        y = np.sin(n * np.pi * x / L)
        ax.plot(x, y, label=f"mode n={n} (λ={2*L/n:.3g})")
    ax.set_title("Reflection (boundary) → discrete eigenmodes")
    ax.set_xlabel("x")
    ax.set_ylabel("u_n(x) (arb.)")
    ax.set_xlim(0.0, L)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # Right: wave packet snapshots (reflection in time).
    ax = axes[1]
    snapshots: dict[int, np.ndarray] = sim["snapshots"]
    steps_sorted = sorted(snapshots.keys())
    for step in steps_sorted:
        t = step * sim["dt"]
        ax.plot(x, snapshots[step], label=f"t={t:.3f}")
    ax.set_title("Wave packet reflects at boundaries (Dirichlet)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t) (arb.)")
    ax.set_xlim(0.0, L)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=1)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DemoConfig()
    sim = _simulate_wave_packet(cfg)

    out_png = out_dir / "particle_reflection_demo.png"
    out_metrics = out_dir / "particle_reflection_demo_metrics.json"

    _plot(cfg, sim, out_png=out_png)

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "config": {
            "L": cfg.L,
            "n_x": cfg.n_x,
            "c": cfg.c,
            "dt_cfl": cfg.dt_cfl,
            "steps": cfg.steps,
            "snapshot_steps": list(cfg.snapshot_steps),
            "packet_center": cfg.packet_center,
            "packet_sigma": cfg.packet_sigma,
            "packet_k": cfg.packet_k,
        },
        "derived": {
            "dx": float(sim["dx"]),
            "dt": float(sim["dt"]),
            "cfl": float(cfg.c * sim["dt"] / sim["dx"]),
        },
        "outputs": {
            "png": str(out_png),
        },
        "notes": [
            "This is a toy model: 1D wave equation with Dirichlet boundaries (reflection).",
            "It illustrates how boundary (reflection) conditions produce discrete modes and particle-like bounded behavior.",
        ],
    }
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_metrics}")


if __name__ == "__main__":
    main()

