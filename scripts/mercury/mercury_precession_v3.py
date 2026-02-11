import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog

# Constants
G = 6.67430e-11
M_SUN = 1.989e30
C = 299792458.0
AU = 1.496e11

# Mercury Parameters
a = 0.387098 * AU
e = 0.205630
T_orb = 87.969 * 24 * 3600 # seconds

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
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

def calc_theoretical_shift_approx(c_val):
    # Standard Einstein approximation (valid for small shifts)
    dw_rad = (6 * np.pi * G * M_SUN) / (c_val**2 * a * (1 - e**2))
    return np.degrees(dw_rad) * 3600

def deriv_gr(t, state, G, M, c):
    x, y, vx, vy = state
    r_vec = np.array([x, y])
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2)
    h = x*vy - y*vx
    factor = 1.0 + (3.0 * h**2) / (c**2 * r2)
    a_vec = -G * M / (r**3) * r_vec * factor
    return [vx, vy, a_vec[0], a_vec[1]]

def deriv_newton(t, state, G, M):
    x, y, vx, vy = state
    r_vec = np.array([x, y])
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2)
    a_vec = -G * M / (r**3) * r_vec
    return [vx, vy, a_vec[0], a_vec[1]]

def _perihelion_event(_t, state, *_args):
    x, y, vx, vy = state
    # d(r^2)/dt = 2*(x*vx + y*vy). Perihelion occurs when dr/dt changes from - to +.
    return x * vx + y * vy


_perihelion_event.terminal = False  # type: ignore[attr-defined]
_perihelion_event.direction = 1.0  # type: ignore[attr-defined]


def _simulate_perihelion_shifts(*, model: str, num_orbits: int, c_val: float):
    if model not in ("pmodel", "newton"):
        raise ValueError(f"Unknown model: {model}")

    r_p = a * (1 - e)
    v_p = np.sqrt(G * M_SUN / a * (1 + e) / (1 - e))
    y0 = [r_p, 0.0, 0.0, v_p]

    t_end = float(num_orbits) * float(T_orb)
    max_step = T_orb / 100.0  # safety: do not skip event crossings

    if model == "pmodel":
        f = deriv_gr
        f_args = (G, M_SUN, c_val)
    else:
        f = deriv_newton
        f_args = (G, M_SUN)

    sol = solve_ivp(
        f,
        (0.0, t_end),
        y0,
        method="DOP853",
        events=_perihelion_event,
        rtol=1e-12,
        atol=1e-12,
        max_step=max_step,
        args=f_args,
    )

    if not sol.t_events or len(sol.t_events[0]) == 0:
        raise RuntimeError("Perihelion events not detected.")

    t_ev = np.array(sol.t_events[0], dtype=float)
    y_ev = np.array(sol.y_events[0], dtype=float)  # shape: (n_events, 4)

    # Some solvers may record the initial point as an event (t=0). Drop it.
    keep = t_ev > 1.0
    t_ev = t_ev[keep]
    y_ev = y_ev[keep]

    if len(t_ev) == 0:
        raise RuntimeError("Perihelion events only at t=0; simulation window too short?")

    angles_rad = np.arctan2(y_ev[:, 1], y_ev[:, 0])
    angle0 = 0.0  # initial perihelion is on +x axis by construction
    shift_arcsec = np.degrees(np.unwrap(angles_rad - angle0)) * 3600.0

    orbit_nums = np.arange(1, len(shift_arcsec) + 1, dtype=int)
    return {
        "orbit_nums": orbit_nums,
        "shift_arcsec": shift_arcsec,
        "t_ev_s": t_ev,
    }

def main():
    print("--- Mercury Simulation (Perihelion Precession) ---")

    out_dir = _ROOT / "output" / "mercury"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Reference (GR residual after planetary perturbations), representative value.
    reference_arcsec_century = 42.98

    # Theoretical Einstein approximation
    orbits_per_century = (100.0 * 365.25 * 24.0 * 3600.0) / float(T_orb)
    einstein_arcsec_per_orbit = float(calc_theoretical_shift_approx(C))
    einstein_arcsec_century = einstein_arcsec_per_orbit * orbits_per_century
    
    # 1) Physical simulation (real C) for quantitative metrics
    num_orbits_physical = 200
    p_ph = _simulate_perihelion_shifts(model="pmodel", num_orbits=num_orbits_physical, c_val=C)
    n_ph = _simulate_perihelion_shifts(model="newton", num_orbits=num_orbits_physical, c_val=C)

    slope_p, intercept_p = np.polyfit(p_ph["orbit_nums"], p_ph["shift_arcsec"], 1)
    slope_n, intercept_n = np.polyfit(n_ph["orbit_nums"], n_ph["shift_arcsec"], 1)

    p_arcsec_century = float(slope_p) * orbits_per_century
    n_arcsec_century = float(slope_n) * orbits_per_century

    # 2) Exaggerated orbit visualization (for readability)
    c_fake = C / 1000.0
    
    # Run Simulation
    r_p = a * (1 - e)
    v_p = np.sqrt(G * M_SUN / a * (1 + e) / (1 - e))
    y0 = [r_p, 0, 0, v_p]
    
    num_orbits = 6
    t_eval = np.linspace(0, num_orbits * T_orb, 30000)
    
    sol_gr = solve_ivp(
        deriv_gr,
        (0, num_orbits * T_orb),
        y0,
        method="DOP853",
        t_eval=t_eval,
        args=(G, M_SUN, c_fake),
        rtol=1e-12,
        atol=1e-12,
    )
    sol_nw = solve_ivp(
        deriv_newton,
        (0, num_orbits * T_orb),
        y0,
        method="DOP853",
        t_eval=t_eval,
        args=(G, M_SUN),
        rtol=1e-12,
        atol=1e-12,
    )
    
    x_gr, y_gr = sol_gr.y[0], sol_gr.y[1]
    x_nw, y_nw = sol_nw.y[0], sol_nw.y[1]
    
    # --- Linear Trend for the physical simulation (P-model) ---
    trend_line_p = slope_p * p_ph["orbit_nums"] + intercept_p
    # Observed representative value (arcsec/century) expressed as cumulative shift vs orbit count.
    reference_arcsec_per_orbit = float(reference_arcsec_century) / float(orbits_per_century)
    reference_line = reference_arcsec_per_orbit * p_ph["orbit_nums"]
    
    # Plot
    _set_japanese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Orbit
    ax1.plot(0, 0, 'yo', markersize=12, label='太陽', zorder=10)
    ax1.plot(x_nw, y_nw, 'k--', linewidth=1.0, alpha=0.5, label='ニュートン')
    ax1.plot(x_gr, y_gr, 'b-', linewidth=1.5, label='P-model 軌道')
    
    ax1.set_title(
        "水星の近日点移動（誇張表示）\n"
        f"参考: 観測残差≈{reference_arcsec_century:.2f} 角秒/世紀, "
        f"P-model(実C)≈{p_arcsec_century:.2f} 角秒/世紀",
        fontsize=14,
    )
    
    ax1.axis('equal')
    ax1.grid(True, linestyle='--')
    ax1.legend(loc='lower left', fontsize=12)

    # Right: Physical shift vs orbit count
    ax2.plot(
        p_ph["orbit_nums"],
        p_ph["shift_arcsec"],
        "r-",
        linewidth=2.0,
        label="P-model（実C）",
    )
    ax2.plot(
        n_ph["orbit_nums"],
        n_ph["shift_arcsec"],
        "k--",
        linewidth=1.2,
        alpha=0.6,
        label="ニュートン（誤差目安）",
    )
    ax2.plot(
        p_ph["orbit_nums"],
        trend_line_p,
        "b--",
        linewidth=1.5,
        alpha=0.7,
        label="線形フィット（一定性）",
    )
    ax2.scatter(
        p_ph["orbit_nums"],
        reference_line,
        s=18,
        color="tab:purple",
        alpha=0.75,
        edgecolors="none",
        label=f"観測（代表値: {reference_arcsec_century:.2f} 角秒/世紀）",
        zorder=3,
    )

    ax2.set_title("近日点移動の累積（周回ごと）\n（実Cでの定量評価）", fontsize=14)
    ax2.set_xlabel("周回数", fontsize=12)
    ax2.set_ylabel("移動角 [角秒]", fontsize=12)
    ax2.grid(True, linestyle='--')
    ax2.legend(loc='lower right', fontsize=12)
    
    ax2.text(
        0.05,
        0.94,
        f"推定: {p_arcsec_century:.2f} 角秒/世紀\n"
        f"Einstein近似: {einstein_arcsec_century:.2f} 角秒/世紀\n"
        f"観測代表: {reference_arcsec_century:.2f} 角秒/世紀",
        transform=ax2.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    plt.tight_layout()
    out_file = out_dir / "mercury_orbit.png"
    plt.savefig(out_file, dpi=300)
    print(f"Graph saved to {out_file}")

    # Save perihelion shifts and metrics for the report
    csv_path = out_dir / "mercury_perihelion_shifts.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("orbit,shift_arcsec_pmodel,shift_arcsec_newton\n")
        for i in range(min(len(p_ph["orbit_nums"]), len(n_ph["orbit_nums"]))):
            f.write(f"{int(p_ph['orbit_nums'][i])},{p_ph['shift_arcsec'][i]:.10f},{n_ph['shift_arcsec'][i]:.10f}\n")
    print(f"CSV saved to {csv_path}")

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference_arcsec_century": reference_arcsec_century,
        "orbits_per_century": orbits_per_century,
        "einstein_approx": {
            "arcsec_per_orbit": einstein_arcsec_per_orbit,
            "arcsec_per_century": einstein_arcsec_century,
        },
        "simulation_physical": {
            "num_orbits": int(num_orbits_physical),
            "pmodel": {
                "slope_arcsec_per_orbit": float(slope_p),
                "arcsec_per_century": p_arcsec_century,
                "fit_intercept_arcsec": float(intercept_p),
            },
            "newton": {
                "slope_arcsec_per_orbit": float(slope_n),
                "arcsec_per_century": n_arcsec_century,
                "fit_intercept_arcsec": float(intercept_n),
            },
        },
        "visualization": {
            "c_fake_scale": 1000.0,
            "num_orbits": int(num_orbits),
        },
    }
    metrics_path = out_dir / "mercury_precession_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metrics saved to {metrics_path}")

    try:
        worklog.append_event(
            {
                "event_type": "mercury_precession",
                "argv": sys.argv,
                "metrics": {
                    "p_arcsec_century": float(p_arcsec_century),
                    "reference_arcsec_century": float(reference_arcsec_century),
                    "einstein_arcsec_century": float(einstein_arcsec_century),
                    "num_orbits_physical": int(num_orbits_physical),
                },
                "outputs": {
                    "mercury_orbit_png": out_file,
                    "perihelion_shifts_csv": csv_path,
                    "metrics_json": metrics_path,
                },
            }
        )
    except Exception:
        pass

if __name__ == "__main__":
    main()
