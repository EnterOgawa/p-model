from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_expect_sigma_z` の入出力契約と処理意図を定義する。

def _expect_sigma_z(psi: np.ndarray) -> float:
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return float(np.real(np.vdot(psi, sigma_z @ psi)))


# 関数: `_coherence_abs` の入出力契約と処理意図を定義する。

def _coherence_abs(psi: np.ndarray) -> float:
    return float(abs(np.conjugate(psi[0]) * psi[1]))


# 関数: `_safe_norm` の入出力契約と処理意図を定義する。

def _safe_norm(psi: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(psi))
    # 条件分岐: `not np.isfinite(norm) or norm < 1e-30` を満たす経路を評価する。
    if not np.isfinite(norm) or norm < 1e-30:
        return np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.complex128)

    return psi / norm


# 関数: `_safe_corr` の入出力契約と処理意図を定義する。

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    # 条件分岐: `(not np.isfinite(x_std)) or (not np.isfinite(y_std)) or x_std < 1e-15 or y_st...` を満たす経路を評価する。
    if (not np.isfinite(x_std)) or (not np.isfinite(y_std)) or x_std < 1e-15 or y_std < 1e-15:
        return 0.0

    value = float(np.corrcoef(x, y)[0, 1])
    # 条件分岐: `not np.isfinite(value)` を満たす経路を評価する。
    if not np.isfinite(value):
        return 0.0

    return value


# 関数: `_simulate_trajectory` の入出力契約と処理意図を定義する。

def _simulate_trajectory(
    *,
    rng: np.random.Generator,
    n_steps: int,
    dt_s: float,
    omega_rad_s: float,
    gamma_meas_s_inv: float,
    pointer_relax_s_inv: np.ndarray,
    pointer_gain_s_inv: np.ndarray,
    pointer_env_gain_s_inv: np.ndarray,
    pointer_noise_sqrt_s_inv: np.ndarray,
    env_relax_s_inv: float,
    env_noise_sqrt_s_inv: float,
    env_detune_rad_s: float,
    collapse_threshold_abs_z: float,
) -> dict[str, Any]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    ident = np.eye(2, dtype=np.complex128)

    psi = np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.complex128)
    n_pointer_channels = int(pointer_relax_s_inv.size)
    pointer = np.zeros(n_pointer_channels, dtype=float)
    env_state = 0.0

    z_hist = np.empty(n_steps + 1, dtype=float)
    p1_hist = np.empty(n_steps + 1, dtype=float)
    coherence_hist = np.empty(n_steps + 1, dtype=float)
    pointer_hist = np.empty((n_steps + 1, n_pointer_channels), dtype=float)
    env_hist = np.empty(n_steps + 1, dtype=float)

    z0 = _expect_sigma_z(psi)
    z_hist[0] = z0
    p1_hist[0] = float(abs(psi[1]) ** 2)
    coherence_hist[0] = _coherence_abs(psi)
    pointer_hist[0, :] = pointer
    env_hist[0] = env_state

    collapse_time_s = math.nan
    collapse_state = 0
    collapse_step = -1

    for step in range(1, n_steps + 1):
        d_w_env = float(rng.normal(loc=0.0, scale=math.sqrt(dt_s)))
        env_state += (-env_relax_s_inv * env_state) * dt_s + env_noise_sqrt_s_inv * d_w_env

        z_now = _expect_sigma_z(psi)
        m_dev = sigma_z - z_now * ident
        d_w = float(rng.normal(loc=0.0, scale=math.sqrt(dt_s)))

        hamiltonian_term = -1j * (
            0.5 * omega_rad_s * (sigma_x @ psi) + 0.5 * env_detune_rad_s * env_state * (sigma_z @ psi)
        )
        nonlinear_drift = -0.5 * gamma_meas_s_inv * (m_dev @ (m_dev @ psi))
        stochastic_term = math.sqrt(gamma_meas_s_inv) * (m_dev @ psi) * d_w
        psi = psi + (hamiltonian_term + nonlinear_drift) * dt_s + stochastic_term
        psi = _safe_norm(psi)

        z_new = _expect_sigma_z(psi)
        d_wp = rng.normal(loc=0.0, scale=math.sqrt(dt_s), size=n_pointer_channels)
        pointer += (
            (-pointer_relax_s_inv * pointer + pointer_gain_s_inv * z_new + pointer_env_gain_s_inv * env_state) * dt_s
            + pointer_noise_sqrt_s_inv * d_wp
        )

        z_hist[step] = z_new
        p1_hist[step] = float(abs(psi[1]) ** 2)
        coherence_hist[step] = _coherence_abs(psi)
        pointer_hist[step, :] = pointer
        env_hist[step] = env_state

        # 条件分岐: `not np.isfinite(collapse_time_s) and abs(z_new) >= collapse_threshold_abs_z` を満たす経路を評価する。
        if not np.isfinite(collapse_time_s) and abs(z_new) >= collapse_threshold_abs_z:
            collapse_time_s = step * dt_s
            collapse_state = 1 if z_new >= 0.0 else -1
            collapse_step = int(step)

    target_sign = 1 if z_hist[-1] >= 0.0 else -1
    # 条件分岐: `collapse_step >= 0` を満たす経路を評価する。
    if collapse_step >= 0:
        post_z = z_hist[collapse_step:]
    else:
        post_z = z_hist[int(0.8 * n_steps) :]

    post_sign = np.sign(post_z)
    post_sign = np.where(post_sign == 0, target_sign, post_sign)
    post_abs_z = np.abs(post_z)
    post_sign_consistency = float(np.mean(post_sign == target_sign)) if post_sign.size > 0 else 0.0
    post_abs_z_mean = float(np.mean(post_abs_z)) if post_abs_z.size > 0 else 0.0
    branch_reversal = bool(np.any((post_sign != target_sign) & (post_abs_z >= 0.2))) if post_sign.size > 0 else True

    return {
        "z_hist": z_hist,
        "p1_hist": p1_hist,
        "coherence_hist": coherence_hist,
        "pointer_hist": pointer_hist,
        "env_hist": env_hist,
        "collapse_time_s": collapse_time_s,
        "collapse_state": int(collapse_state),
        "collapse_step": int(collapse_step),
        "branch_reversal": branch_reversal,
        "post_sign_consistency": post_sign_consistency,
        "post_abs_z_mean": post_abs_z_mean,
        "final_expect_z": float(z_hist[-1]),
        "final_p1": float(p1_hist[-1]),
        "final_p0": float(1.0 - p1_hist[-1]),
        "final_coherence_abs": float(coherence_hist[-1]),
        "final_pointer_channels": [float(v) for v in pointer_hist[-1, :]],
        "final_pointer": float(pointer_hist[-1, 0]),
        "final_pointer_mean": float(np.mean(pointer_hist[-1, :])),
        "final_env": float(env_hist[-1]),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase = 8
    step = "8.7.20"

    seed = 20260216
    n_trajectories = 320
    n_samples_for_plot = 12
    total_time_s = 1.6
    dt_s = 4.0e-4
    n_steps = int(round(total_time_s / dt_s))
    collapse_threshold_abs_z = 0.95

    omega_rad_s = 0.8
    gamma_meas_s_inv = 9.0
    pointer_relax_s_inv = np.array([2.4, 2.0, 2.8], dtype=float)
    pointer_gain_s_inv = np.array([5.0, 4.3, 5.7], dtype=float)
    pointer_env_gain_s_inv = np.array([0.85, -0.55, 1.05], dtype=float)
    pointer_noise_sqrt_s_inv = np.array([0.30, 0.34, 0.27], dtype=float)
    env_relax_s_inv = 1.6
    env_noise_sqrt_s_inv = 0.42
    env_detune_rad_s = 0.65
    n_pointer_channels = int(pointer_relax_s_inv.size)

    rng = np.random.default_rng(seed)
    times_s = np.linspace(0.0, n_steps * dt_s, n_steps + 1, dtype=float)

    trajectories: list[dict[str, Any]] = []
    for _ in range(n_trajectories):
        trajectories.append(
            _simulate_trajectory(
                rng=rng,
                n_steps=n_steps,
                dt_s=dt_s,
                omega_rad_s=omega_rad_s,
                gamma_meas_s_inv=gamma_meas_s_inv,
                pointer_relax_s_inv=pointer_relax_s_inv,
                pointer_gain_s_inv=pointer_gain_s_inv,
                pointer_env_gain_s_inv=pointer_env_gain_s_inv,
                pointer_noise_sqrt_s_inv=pointer_noise_sqrt_s_inv,
                env_relax_s_inv=env_relax_s_inv,
                env_noise_sqrt_s_inv=env_noise_sqrt_s_inv,
                env_detune_rad_s=env_detune_rad_s,
                collapse_threshold_abs_z=collapse_threshold_abs_z,
            )
        )

    collapse_times = np.array([float(t["collapse_time_s"]) for t in trajectories], dtype=float)
    collapse_states = np.array([int(t["collapse_state"]) for t in trajectories], dtype=int)
    final_expect_z = np.array([float(t["final_expect_z"]) for t in trajectories], dtype=float)
    final_pointer_matrix = np.array([t["final_pointer_channels"] for t in trajectories], dtype=float)
    final_pointer = final_pointer_matrix[:, 0]
    final_pointer_mean = np.mean(final_pointer_matrix, axis=1)
    final_env = np.array([float(t["final_env"]) for t in trajectories], dtype=float)
    final_coherence = np.array([float(t["final_coherence_abs"]) for t in trajectories], dtype=float)
    branch_reversal_flags = np.array([bool(t["branch_reversal"]) for t in trajectories], dtype=bool)
    post_sign_consistency_values = np.array([float(t["post_sign_consistency"]) for t in trajectories], dtype=float)
    post_abs_z_mean_values = np.array([float(t["post_abs_z_mean"]) for t in trajectories], dtype=float)

    collapsed_mask = np.isfinite(collapse_times)
    collapsed_n = int(np.sum(collapsed_mask))
    collapse_fraction = float(collapsed_n / max(1, n_trajectories))
    collapse_time_median_s = float(np.nanmedian(collapse_times)) if collapsed_n > 0 else math.nan
    collapse_time_p90_s = float(np.nanquantile(collapse_times[collapsed_mask], 0.9)) if collapsed_n > 1 else math.nan
    collapse_time_mean_s = float(np.nanmean(collapse_times[collapsed_mask])) if collapsed_n > 0 else math.nan

    branch_plus_fraction = float(np.mean(collapse_states == 1))
    branch_minus_fraction = float(np.mean(collapse_states == -1))
    unresolved_fraction = float(np.mean(collapse_states == 0))
    final_expect_z_mean = float(np.mean(final_expect_z))
    final_expect_z_abs_mean = float(np.mean(np.abs(final_expect_z)))
    final_coherence_median = float(np.median(final_coherence))
    initial_coherence = 0.5
    coherence_suppression_ratio = float(final_coherence_median / initial_coherence)
    pointer_z_correlation = _safe_corr(final_pointer, final_expect_z)
    pointer_mean_z_correlation = _safe_corr(final_pointer_mean, final_expect_z)
    env_z_correlation = _safe_corr(final_env, final_expect_z)
    env_pointer_mean_correlation = _safe_corr(final_env, final_pointer_mean)
    pointer_channel_z_correlations = [
        _safe_corr(final_pointer_matrix[:, idx], final_expect_z) for idx in range(n_pointer_channels)
    ]
    pointer_channel_abs_corr_mean = float(np.mean(np.abs(pointer_channel_z_correlations)))
    branch_reversal_fraction = float(np.mean(branch_reversal_flags))
    branch_stable_fraction = float(1.0 - branch_reversal_fraction)
    post_sign_consistency_median = float(np.median(post_sign_consistency_values))
    post_abs_z_mean_median = float(np.median(post_abs_z_mean_values))

    # 条件分岐: `n_pointer_channels > 1` を満たす経路を評価する。
    if n_pointer_channels > 1:
        corr_matrix = np.corrcoef(final_pointer_matrix.T)
        finite_corr = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        upper_indices = np.triu_indices(n_pointer_channels, k=1)
        pointer_interchannel_corr_abs_mean = float(np.mean(np.abs(finite_corr[upper_indices])))
    else:
        pointer_interchannel_corr_abs_mean = 1.0

    state_sign = np.where(collapse_states != 0, collapse_states, np.where(final_expect_z >= 0.0, 1, -1))
    pointer_sign_matrix = np.sign(final_pointer_matrix)
    pointer_vote = np.sum(pointer_sign_matrix, axis=1)
    pointer_majority_sign = np.sign(pointer_vote)
    pointer_majority_sign = np.where(pointer_majority_sign == 0, np.sign(final_pointer_mean), pointer_majority_sign)
    pointer_majority_sign = np.where(pointer_majority_sign == 0, state_sign, pointer_majority_sign)
    pointer_consensus_fraction = float(np.mean(pointer_majority_sign == state_sign))
    pointer_majority_unresolved_fraction = float(np.mean(pointer_vote == 0))

    out_csv = out_dir / "quantum_measurement_dynamic_collapse_simulation.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "trajectory_id",
            "collapse_time_s",
            "collapse_state",
            "final_expect_z",
            "final_p0",
            "final_p1",
            "final_coherence_abs",
            "final_pointer",
            "final_pointer_mean",
            "final_env",
            "pointer_majority_sign",
        ]
        cols.extend([f"final_pointer_ch{idx}" for idx in range(n_pointer_channels)])
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for idx, traj in enumerate(trajectories):
            row = {
                "trajectory_id": idx,
                "collapse_time_s": traj["collapse_time_s"],
                "collapse_state": traj["collapse_state"],
                "final_expect_z": traj["final_expect_z"],
                "final_p0": traj["final_p0"],
                "final_p1": traj["final_p1"],
                "final_coherence_abs": traj["final_coherence_abs"],
                "final_pointer": traj["final_pointer"],
                "final_pointer_mean": traj["final_pointer_mean"],
                "final_env": traj["final_env"],
                "pointer_majority_sign": int(pointer_majority_sign[idx]),
            }
            for channel_index in range(n_pointer_channels):
                row[f"final_pointer_ch{channel_index}"] = float(traj["final_pointer_channels"][channel_index])

            writer.writerow(row)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=160)

    sample_count = min(n_samples_for_plot, n_trajectories)
    sample_ids = np.linspace(0, n_trajectories - 1, sample_count, dtype=int)
    for traj_id in sample_ids:
        axes[0, 0].plot(times_s, trajectories[traj_id]["z_hist"], lw=1.1, alpha=0.85)

    axes[0, 0].axhline(collapse_threshold_abs_z, color="black", ls="--", lw=1.0, alpha=0.7)
    axes[0, 0].axhline(-collapse_threshold_abs_z, color="black", ls="--", lw=1.0, alpha=0.7)
    axes[0, 0].set_title("State collapse trajectory: expectation z(t)")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("expectation z")
    axes[0, 0].set_ylim(-1.05, 1.05)
    axes[0, 0].grid(ls=":", alpha=0.35)

    sample_traj_id = int(sample_ids[0]) if sample_count > 0 else 0
    sample_pointer_hist = trajectories[sample_traj_id]["pointer_hist"]
    for channel_index in range(n_pointer_channels):
        axes[0, 1].plot(
            times_s,
            sample_pointer_hist[:, channel_index],
            lw=1.2,
            alpha=0.9,
            label=f"channel {channel_index}",
        )

    axes[0, 1].plot(
        times_s,
        np.mean(sample_pointer_hist, axis=1),
        color="black",
        lw=1.4,
        ls="--",
        label="channel mean",
    )
    axes[0, 1].set_title("Detector pointer channels x_k(t) (single trajectory)")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("pointer x_k")
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[0, 1].grid(ls=":", alpha=0.35)

    # 条件分岐: `collapsed_n > 0` を満たす経路を評価する。
    if collapsed_n > 0:
        axes[1, 0].hist(collapse_times[collapsed_mask], bins=24, color="#4c78a8", alpha=0.85)
        # 条件分岐: `np.isfinite(collapse_time_median_s)` を満たす経路を評価する。
        if np.isfinite(collapse_time_median_s):
            axes[1, 0].axvline(collapse_time_median_s, color="black", ls="--", lw=1.0, label="median")
            axes[1, 0].legend(loc="upper right", fontsize=9)
    else:
        axes[1, 0].text(0.5, 0.5, "no collapse event in this run", ha="center", va="center", transform=axes[1, 0].transAxes)

    axes[1, 0].set_title("Collapse-time distribution")
    axes[1, 0].set_xlabel("collapse time [s]")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(ls=":", alpha=0.35)

    color_values = np.where(collapse_states == 1, "#2ca02c", np.where(collapse_states == -1, "#d62728", "#7f7f7f"))
    axes[1, 1].scatter(final_pointer_mean, final_expect_z, s=20, alpha=0.75, c=color_values)
    axes[1, 1].axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.6)
    axes[1, 1].set_title("Final detector pointer mean vs state branch")
    axes[1, 1].set_xlabel("final pointer mean x̄(T)")
    axes[1, 1].set_ylabel("final expectation z(T)")
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].grid(ls=":", alpha=0.35)

    fig.suptitle(
        "Dynamic collapse simulation (nonlinear measurement equation; two-state + multi-DOF detector pointer)",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / "quantum_measurement_dynamic_collapse_simulation.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_metrics = out_dir / "quantum_measurement_dynamic_collapse_simulation_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "step": step,
                "model": {
                    "equation": "d|psi> = [-iH dt - (gamma/2)(M-<M>)^2 dt + sqrt(gamma)(M-<M>) dW]|psi>",
                    "detector_pointer": "dx_k = (-Gamma_k*x_k + chi_k*<M> + eta_k*E)dt + sigma_xi,k dW_k",
                    "environment": "dE = (-lambda_E*E)dt + sigma_E dW_E",
                    "hamiltonian": "H=(omega/2)*sigma_x + (omega_env/2)*E*sigma_z",
                    "measurement_operator": "M=sigma_z",
                },
                "parameters": {
                    "seed": seed,
                    "n_trajectories": n_trajectories,
                    "total_time_s": total_time_s,
                    "dt_s": dt_s,
                    "n_steps": n_steps,
                    "collapse_threshold_abs_z": collapse_threshold_abs_z,
                    "omega_rad_s": omega_rad_s,
                    "gamma_meas_s_inv": gamma_meas_s_inv,
                    "pointer_channels_n": n_pointer_channels,
                    "pointer_relax_s_inv": [float(v) for v in pointer_relax_s_inv],
                    "pointer_gain_s_inv": [float(v) for v in pointer_gain_s_inv],
                    "pointer_env_gain_s_inv": [float(v) for v in pointer_env_gain_s_inv],
                    "pointer_noise_sqrt_s_inv": [float(v) for v in pointer_noise_sqrt_s_inv],
                    "env_relax_s_inv": env_relax_s_inv,
                    "env_noise_sqrt_s_inv": env_noise_sqrt_s_inv,
                    "env_detune_rad_s": env_detune_rad_s,
                },
                "summary": {
                    "collapse_fraction": collapse_fraction,
                    "collapse_time_median_s": collapse_time_median_s,
                    "collapse_time_mean_s": collapse_time_mean_s,
                    "collapse_time_p90_s": collapse_time_p90_s,
                    "branch_plus_fraction": branch_plus_fraction,
                    "branch_minus_fraction": branch_minus_fraction,
                    "unresolved_fraction": unresolved_fraction,
                    "final_expect_z_mean": final_expect_z_mean,
                    "final_expect_z_abs_mean": final_expect_z_abs_mean,
                    "final_coherence_median": final_coherence_median,
                    "coherence_suppression_ratio": coherence_suppression_ratio,
                    "pointer_z_correlation": pointer_z_correlation,
                    "pointer_mean_z_correlation": pointer_mean_z_correlation,
                    "env_z_correlation": env_z_correlation,
                    "env_pointer_mean_correlation": env_pointer_mean_correlation,
                    "pointer_channel_z_correlations": pointer_channel_z_correlations,
                    "pointer_channel_abs_corr_mean": pointer_channel_abs_corr_mean,
                    "pointer_interchannel_corr_abs_mean": pointer_interchannel_corr_abs_mean,
                    "pointer_consensus_fraction": pointer_consensus_fraction,
                    "pointer_majority_unresolved_fraction": pointer_majority_unresolved_fraction,
                    "branch_reversal_fraction": branch_reversal_fraction,
                    "branch_stable_fraction": branch_stable_fraction,
                    "post_sign_consistency_median": post_sign_consistency_median,
                    "post_abs_z_mean_median": post_abs_z_mean_median,
                },
                "outputs": {
                    "csv": str(out_csv),
                    "png": str(out_png),
                },
                "notes": [
                    "Operational simulation for measurement dynamics; not yet a first-principles derivation from microscopic P-field interactions.",
                    "This revision extends detector dynamics to multi-DOF pointer channels with environment-dissipation coupling.",
                    "Branch-stability gates are included via reversal fraction and post-collapse sign consistency metrics.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
