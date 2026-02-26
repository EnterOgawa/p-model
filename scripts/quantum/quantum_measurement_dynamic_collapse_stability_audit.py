from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.quantum.quantum_measurement_dynamic_collapse_simulation import _safe_corr, _simulate_trajectory  # noqa: E402
from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_float_list(text: str) -> List[float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    out: List[float] = []
    for part in parts:
        out.append(float(part))

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise ValueError(f"empty numeric list: {text!r}")

    return out


def _parse_int_list(text: str) -> List[int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    out: List[int] = []
    for part in parts:
        out.append(int(part))

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise ValueError(f"empty integer list: {text!r}")

    return out


def _fmt(v: float, digits: int = 7) -> str:
    # 条件分岐: `not math.isfinite(float(v))` を満たす経路を評価する。
    if not math.isfinite(float(v)):
        return ""

    x = float(v)
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _run_one(
    *,
    seed: int,
    env_scale: float,
    n_trajectories: int,
    n_steps: int,
    dt_s: float,
    omega_rad_s: float,
    gamma_meas_s_inv: float,
    pointer_relax_s_inv: np.ndarray,
    pointer_gain_s_inv: np.ndarray,
    pointer_env_gain_s_inv_base: np.ndarray,
    pointer_noise_sqrt_s_inv: np.ndarray,
    env_relax_s_inv: float,
    env_noise_sqrt_s_inv_base: float,
    env_detune_rad_s_base: float,
    collapse_threshold_abs_z: float,
    tau50_reference_s: float,
    tau50_ratio_min: float,
    tau50_ratio_max: float,
    min_collapse_fraction: float,
    min_pointer_consensus: float,
    max_branch_reversal: float,
    min_post_sign_consistency: float,
    reject_collapse_fraction: float,
    reject_branch_reversal: float,
    reject_post_sign_consistency: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    pointer_env_gain_s_inv = pointer_env_gain_s_inv_base * float(env_scale)
    env_noise_sqrt_s_inv = float(env_noise_sqrt_s_inv_base) * float(env_scale)
    env_detune_rad_s = float(env_detune_rad_s_base) * float(env_scale)

    n_pointer_channels = int(pointer_relax_s_inv.size)
    collapse_times = np.full(n_trajectories, np.nan, dtype=float)
    collapse_states = np.zeros(n_trajectories, dtype=int)
    final_expect_z = np.zeros(n_trajectories, dtype=float)
    final_pointer_matrix = np.zeros((n_trajectories, n_pointer_channels), dtype=float)
    final_pointer_mean = np.zeros(n_trajectories, dtype=float)
    final_env = np.zeros(n_trajectories, dtype=float)
    final_coherence = np.zeros(n_trajectories, dtype=float)
    branch_reversal_flags = np.zeros(n_trajectories, dtype=bool)
    post_sign_consistency_values = np.zeros(n_trajectories, dtype=float)
    post_abs_z_mean_values = np.zeros(n_trajectories, dtype=float)

    for idx in range(n_trajectories):
        tr = _simulate_trajectory(
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
        collapse_times[idx] = float(tr["collapse_time_s"])
        collapse_states[idx] = int(tr["collapse_state"])
        final_expect_z[idx] = float(tr["final_expect_z"])
        final_pointer_matrix[idx, :] = np.asarray(tr["final_pointer_channels"], dtype=float)
        final_pointer_mean[idx] = float(tr["final_pointer_mean"])
        final_env[idx] = float(tr["final_env"])
        final_coherence[idx] = float(tr["final_coherence_abs"])
        branch_reversal_flags[idx] = bool(tr["branch_reversal"])
        post_sign_consistency_values[idx] = float(tr["post_sign_consistency"])
        post_abs_z_mean_values[idx] = float(tr["post_abs_z_mean"])

    collapsed_mask = np.isfinite(collapse_times)
    collapsed_n = int(np.sum(collapsed_mask))
    collapse_fraction = float(collapsed_n / max(1, n_trajectories))
    tau50_s = float(np.nanmedian(collapse_times[collapsed_mask])) if collapsed_n > 0 else float("nan")
    tau90_s = float(np.nanquantile(collapse_times[collapsed_mask], 0.9)) if collapsed_n > 1 else float("nan")
    tau_mean_s = float(np.nanmean(collapse_times[collapsed_mask])) if collapsed_n > 0 else float("nan")

    branch_plus_fraction = float(np.mean(collapse_states == 1))
    branch_minus_fraction = float(np.mean(collapse_states == -1))
    unresolved_fraction = float(np.mean(collapse_states == 0))
    final_expect_z_mean = float(np.mean(final_expect_z))
    final_expect_z_abs_mean = float(np.mean(np.abs(final_expect_z)))
    final_coherence_median = float(np.median(final_coherence))
    initial_coherence = 0.5
    coherence_suppression_ratio = float(final_coherence_median / initial_coherence)
    pointer_mean_z_correlation = _safe_corr(final_pointer_mean, final_expect_z)
    env_z_correlation = _safe_corr(final_env, final_expect_z)

    pointer_channel_z_correlations = [
        _safe_corr(final_pointer_matrix[:, channel_index], final_expect_z) for channel_index in range(n_pointer_channels)
    ]
    pointer_channel_abs_corr_mean = float(np.mean(np.abs(pointer_channel_z_correlations)))
    # 条件分岐: `n_pointer_channels > 1` を満たす経路を評価する。
    if n_pointer_channels > 1:
        corr_matrix = np.corrcoef(final_pointer_matrix.T)
        finite_corr = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        upper_indices = np.triu_indices(n_pointer_channels, k=1)
        pointer_interchannel_corr_abs_mean = float(np.mean(np.abs(finite_corr[upper_indices])))
    else:
        pointer_interchannel_corr_abs_mean = 1.0

    branch_reversal_fraction = float(np.mean(branch_reversal_flags))
    branch_stable_fraction = float(1.0 - branch_reversal_fraction)
    post_sign_consistency_median = float(np.median(post_sign_consistency_values))
    post_abs_z_mean_median = float(np.median(post_abs_z_mean_values))

    state_sign = np.where(collapse_states != 0, collapse_states, np.where(final_expect_z >= 0.0, 1, -1))
    pointer_sign_matrix = np.sign(final_pointer_matrix)
    pointer_vote = np.sum(pointer_sign_matrix, axis=1)
    pointer_majority_sign = np.sign(pointer_vote)
    pointer_majority_sign = np.where(pointer_majority_sign == 0, np.sign(final_pointer_mean), pointer_majority_sign)
    pointer_majority_sign = np.where(pointer_majority_sign == 0, state_sign, pointer_majority_sign)
    pointer_consensus_fraction = float(np.mean(pointer_majority_sign == state_sign))

    tau50_ratio = float(tau50_s / tau50_reference_s) if math.isfinite(tau50_s) and tau50_reference_s > 0 else float("nan")

    gate_failures: List[str] = []
    # 条件分岐: `collapse_fraction < min_collapse_fraction` を満たす経路を評価する。
    if collapse_fraction < min_collapse_fraction:
        gate_failures.append("collapse_fraction")

    # 条件分岐: `pointer_consensus_fraction < min_pointer_consensus` を満たす経路を評価する。

    if pointer_consensus_fraction < min_pointer_consensus:
        gate_failures.append("pointer_consensus")

    # 条件分岐: `branch_reversal_fraction > max_branch_reversal` を満たす経路を評価する。

    if branch_reversal_fraction > max_branch_reversal:
        gate_failures.append("branch_reversal")

    # 条件分岐: `post_sign_consistency_median < min_post_sign_consistency` を満たす経路を評価する。

    if post_sign_consistency_median < min_post_sign_consistency:
        gate_failures.append("post_sign_consistency")

    # 条件分岐: `(not math.isfinite(tau50_ratio)) or tau50_ratio < tau50_ratio_min or tau50_ra...` を満たす経路を評価する。

    if (not math.isfinite(tau50_ratio)) or tau50_ratio < tau50_ratio_min or tau50_ratio > tau50_ratio_max:
        gate_failures.append("tau50_ratio")

    hard_reject = (
        (collapse_fraction < reject_collapse_fraction)
        or (branch_reversal_fraction > reject_branch_reversal)
        or (post_sign_consistency_median < reject_post_sign_consistency)
    )
    # 条件分岐: `hard_reject` を満たす経路を評価する。
    if hard_reject:
        status = "reject"
        decision = "hard_instability_detected"
    # 条件分岐: 前段条件が不成立で、`not gate_failures` を追加評価する。
    elif not gate_failures:
        status = "pass"
        decision = "all_branch_gates_pass"
    else:
        status = "watch"
        decision = "partial_gate_mismatch"

    return {
        "seed": int(seed),
        "env_scale": float(env_scale),
        "n_trajectories": int(n_trajectories),
        "status": status,
        "decision": decision,
        "gate_failures": gate_failures,
        "collapse_fraction": collapse_fraction,
        "collapse_time_median_s": tau50_s,
        "collapse_time_mean_s": tau_mean_s,
        "collapse_time_p90_s": tau90_s,
        "collapse_time_ratio_vs_ref": tau50_ratio,
        "branch_plus_fraction": branch_plus_fraction,
        "branch_minus_fraction": branch_minus_fraction,
        "unresolved_fraction": unresolved_fraction,
        "pointer_consensus_fraction": pointer_consensus_fraction,
        "branch_reversal_fraction": branch_reversal_fraction,
        "branch_stable_fraction": branch_stable_fraction,
        "post_sign_consistency_median": post_sign_consistency_median,
        "post_abs_z_mean_median": post_abs_z_mean_median,
        "pointer_mean_z_correlation": pointer_mean_z_correlation,
        "pointer_channel_abs_corr_mean": pointer_channel_abs_corr_mean,
        "pointer_interchannel_corr_abs_mean": pointer_interchannel_corr_abs_mean,
        "env_z_correlation": env_z_correlation,
        "coherence_suppression_ratio": coherence_suppression_ratio,
        "final_expect_z_mean": final_expect_z_mean,
        "final_expect_z_abs_mean": final_expect_z_abs_mean,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "run_id",
        "seed",
        "env_scale",
        "n_trajectories",
        "status",
        "decision",
        "gate_failures",
        "collapse_fraction",
        "collapse_time_median_s",
        "collapse_time_mean_s",
        "collapse_time_p90_s",
        "collapse_time_ratio_vs_ref",
        "pointer_consensus_fraction",
        "branch_reversal_fraction",
        "branch_stable_fraction",
        "post_sign_consistency_median",
        "post_abs_z_mean_median",
        "pointer_mean_z_correlation",
        "pointer_channel_abs_corr_mean",
        "pointer_interchannel_corr_abs_mean",
        "env_z_correlation",
        "coherence_suppression_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            values: List[Any] = []
            for h in headers:
                v = row.get(h, "")
                # 条件分岐: `isinstance(v, list)` を満たす経路を評価する。
                if isinstance(v, list):
                    values.append(",".join(str(x) for x in v))
                # 条件分岐: 前段条件が不成立で、`isinstance(v, float)` を追加評価する。
                elif isinstance(v, float):
                    values.append(_fmt(v))
                else:
                    values.append(v)

            writer.writerow(values)


def _status_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"pass": 0, "watch": 0, "reject": 0}
    for row in rows:
        s = str(row.get("status") or "").lower()
        # 条件分岐: `s not in out` を満たす経路を評価する。
        if s not in out:
            continue

        out[s] += 1

    return out


def _plot(
    *,
    rows: List[Dict[str, Any]],
    tau50_ref_s: float,
    tau50_ratio_min: float,
    tau50_ratio_max: float,
    min_collapse_fraction: float,
    min_pointer_consensus: float,
    max_branch_reversal: float,
    out_png: Path,
) -> None:
    labels = [f"s{int(r['seed'])}-e{float(r['env_scale']):.2f}" for r in rows]
    x = np.arange(len(rows), dtype=float)

    tau50_s = np.array([float(r.get("collapse_time_median_s", math.nan)) for r in rows], dtype=float)
    collapse_fraction = np.array([float(r.get("collapse_fraction", math.nan)) for r in rows], dtype=float)
    pointer_consensus = np.array([float(r.get("pointer_consensus_fraction", math.nan)) for r in rows], dtype=float)
    branch_reversal = np.array([float(r.get("branch_reversal_fraction", math.nan)) for r in rows], dtype=float)

    status_color = {"pass": "#2ca02c", "watch": "#e6b422", "reject": "#d62728"}
    colors = [status_color.get(str(r.get("status", "watch")), "#808080") for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.6))

    axes[0, 0].bar(x, tau50_s * 1000.0, color=colors, alpha=0.9)
    axes[0, 0].axhline(float(tau50_ref_s) * 1000.0, color="#4c78a8", ls="--", lw=1.2, label="ref tau50")
    axes[0, 0].axhline(float(tau50_ref_s) * float(tau50_ratio_min) * 1000.0, color="#999999", ls=":", lw=1.0)
    axes[0, 0].axhline(float(tau50_ref_s) * float(tau50_ratio_max) * 1000.0, color="#999999", ls=":", lw=1.0)
    axes[0, 0].set_ylabel("tau50 [ms]")
    axes[0, 0].set_title("Collapse-time median stability")
    axes[0, 0].grid(True, axis="y", alpha=0.25)
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].bar(x, branch_reversal, color=colors, alpha=0.9)
    axes[0, 1].axhline(float(max_branch_reversal), color="#d62728", ls="--", lw=1.2, label="max branch reversal")
    axes[0, 1].set_ylabel("branch reversal fraction")
    axes[0, 1].set_title("Branch-stability gate")
    axes[0, 1].grid(True, axis="y", alpha=0.25)
    axes[0, 1].legend(loc="upper right", fontsize=8)

    axes[1, 0].bar(x, pointer_consensus, color=colors, alpha=0.9)
    axes[1, 0].axhline(float(min_pointer_consensus), color="#d62728", ls="--", lw=1.2, label="min pointer consensus")
    axes[1, 0].set_ylabel("pointer consensus")
    axes[1, 0].set_title("Pointer majority consensus")
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].bar(x, collapse_fraction, color=colors, alpha=0.9)
    axes[1, 1].axhline(float(min_collapse_fraction), color="#d62728", ls="--", lw=1.2, label="min collapse fraction")
    axes[1, 1].set_ylabel("collapse fraction")
    axes[1, 1].set_title("Collapse success gate")
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(loc="upper right", fontsize=8)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("run (seed-env_scale)")

    fig.suptitle("Dynamic-collapse env-coupled stability audit (Step 8.7.20)", fontsize=14)
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 8.7.20: rerun env-coupled dynamic-collapse stability cycle.")
    ap.add_argument(
        "--base-metrics-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "quantum" / "quantum_measurement_dynamic_collapse_simulation_metrics.json"),
    )
    ap.add_argument("--seeds", type=str, default="20260216,20260217,20260218,20260219,20260220,20260221")
    ap.add_argument("--env-scales", type=str, default="0.90,1.00,1.10")
    ap.add_argument("--n-trajectories-per-run", type=int, default=192)
    ap.add_argument("--tau50-ratio-min", type=float, default=0.50)
    ap.add_argument("--tau50-ratio-max", type=float, default=2.00)
    ap.add_argument("--min-collapse-fraction", type=float, default=0.95)
    ap.add_argument("--min-pointer-consensus", type=float, default=0.90)
    ap.add_argument("--max-branch-reversal", type=float, default=0.20)
    ap.add_argument("--min-post-sign-consistency", type=float, default=0.90)
    ap.add_argument("--reject-collapse-fraction", type=float, default=0.90)
    ap.add_argument("--reject-branch-reversal", type=float, default=0.30)
    ap.add_argument("--reject-post-sign-consistency", type=float, default=0.80)
    ap.add_argument("--max-tau50-cv", type=float, default=0.35)
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "quantum"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "quantum"))
    ap.add_argument("--prefix", type=str, default="quantum_measurement_dynamic_collapse_stability_audit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    base_metrics_path = Path(str(args.base_metrics_json))
    # 条件分岐: `not base_metrics_path.exists()` を満たす経路を評価する。
    if not base_metrics_path.exists():
        raise FileNotFoundError(f"missing base metrics json: {base_metrics_path}")

    base_metrics = json.loads(base_metrics_path.read_text(encoding="utf-8"))

    params = base_metrics.get("parameters", {})
    summary_ref = base_metrics.get("summary", {})
    seeds = _parse_int_list(str(args.seeds))
    env_scales = _parse_float_list(str(args.env_scales))

    total_time_s = float(params.get("total_time_s", 1.6))
    dt_s = float(params.get("dt_s", 4.0e-4))
    n_steps = int(params.get("n_steps", int(round(total_time_s / dt_s))))
    omega_rad_s = float(params.get("omega_rad_s", 0.8))
    gamma_meas_s_inv = float(params.get("gamma_meas_s_inv", 9.0))
    collapse_threshold_abs_z = float(params.get("collapse_threshold_abs_z", 0.95))

    pointer_relax_s_inv = np.asarray(params.get("pointer_relax_s_inv", [2.4, 2.0, 2.8]), dtype=float)
    pointer_gain_s_inv = np.asarray(params.get("pointer_gain_s_inv", [5.0, 4.3, 5.7]), dtype=float)
    pointer_env_gain_s_inv_base = np.asarray(params.get("pointer_env_gain_s_inv", [0.85, -0.55, 1.05]), dtype=float)
    pointer_noise_sqrt_s_inv = np.asarray(params.get("pointer_noise_sqrt_s_inv", [0.30, 0.34, 0.27]), dtype=float)
    env_relax_s_inv = float(params.get("env_relax_s_inv", 1.6))
    env_noise_sqrt_s_inv_base = float(params.get("env_noise_sqrt_s_inv", 0.42))
    env_detune_rad_s_base = float(params.get("env_detune_rad_s", 0.65))

    tau50_reference_s = float(summary_ref.get("collapse_time_median_s", float("nan")))
    # 条件分岐: `not math.isfinite(tau50_reference_s) or tau50_reference_s <= 0` を満たす経路を評価する。
    if not math.isfinite(tau50_reference_s) or tau50_reference_s <= 0:
        tau50_reference_s = 0.04

    rows: List[Dict[str, Any]] = []
    run_id = 0
    for env_scale in env_scales:
        for seed in seeds:
            run_id += 1
            row = _run_one(
                seed=seed,
                env_scale=env_scale,
                n_trajectories=max(16, int(args.n_trajectories_per_run)),
                n_steps=n_steps,
                dt_s=dt_s,
                omega_rad_s=omega_rad_s,
                gamma_meas_s_inv=gamma_meas_s_inv,
                pointer_relax_s_inv=pointer_relax_s_inv,
                pointer_gain_s_inv=pointer_gain_s_inv,
                pointer_env_gain_s_inv_base=pointer_env_gain_s_inv_base,
                pointer_noise_sqrt_s_inv=pointer_noise_sqrt_s_inv,
                env_relax_s_inv=env_relax_s_inv,
                env_noise_sqrt_s_inv_base=env_noise_sqrt_s_inv_base,
                env_detune_rad_s_base=env_detune_rad_s_base,
                collapse_threshold_abs_z=collapse_threshold_abs_z,
                tau50_reference_s=tau50_reference_s,
                tau50_ratio_min=float(args.tau50_ratio_min),
                tau50_ratio_max=float(args.tau50_ratio_max),
                min_collapse_fraction=float(args.min_collapse_fraction),
                min_pointer_consensus=float(args.min_pointer_consensus),
                max_branch_reversal=float(args.max_branch_reversal),
                min_post_sign_consistency=float(args.min_post_sign_consistency),
                reject_collapse_fraction=float(args.reject_collapse_fraction),
                reject_branch_reversal=float(args.reject_branch_reversal),
                reject_post_sign_consistency=float(args.reject_post_sign_consistency),
            )
            row["run_id"] = int(run_id)
            rows.append(row)

    counts = _status_counts(rows)
    tau50_values = np.array([float(r.get("collapse_time_median_s", math.nan)) for r in rows], dtype=float)
    tau50_finite = tau50_values[np.isfinite(tau50_values)]
    tau50_cv = float(np.std(tau50_finite) / np.mean(tau50_finite)) if tau50_finite.size >= 2 and float(np.mean(tau50_finite)) > 0 else 0.0

    gate_fail_counts: Dict[str, int] = {}
    for row in rows:
        for fail_key in row.get("gate_failures", []):
            gate_fail_counts[fail_key] = int(gate_fail_counts.get(fail_key, 0)) + 1

    # 条件分岐: `counts["reject"] > 0` を満たす経路を評価する。

    if counts["reject"] > 0:
        overall_status = "reject"
        decision = "stability_break_detected"
    # 条件分岐: 前段条件が不成立で、`tau50_cv > float(args.max_tau50_cv)` を追加評価する。
    elif tau50_cv > float(args.max_tau50_cv):
        overall_status = "watch"
        decision = "tau50_spread_watch"
    # 条件分岐: 前段条件が不成立で、`counts["watch"] > max(1, int(math.floor(0.25 * len(rows))))` を追加評価する。
    elif counts["watch"] > max(1, int(math.floor(0.25 * len(rows)))):
        overall_status = "watch"
        decision = "partial_gate_mismatch"
    else:
        overall_status = "pass"
        decision = "env_seed_stability_confirmed"

    outdir = Path(str(args.outdir))
    public_outdir = Path(str(args.public_outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"

    rows_sorted = sorted(rows, key=lambda r: (float(r.get("env_scale", 1.0)), int(r.get("seed", 0))))
    _write_csv(out_csv, rows_sorted)
    _plot(
        rows=rows_sorted,
        tau50_ref_s=tau50_reference_s,
        tau50_ratio_min=float(args.tau50_ratio_min),
        tau50_ratio_max=float(args.tau50_ratio_max),
        min_collapse_fraction=float(args.min_collapse_fraction),
        min_pointer_consensus=float(args.min_pointer_consensus),
        max_branch_reversal=float(args.max_branch_reversal),
        out_png=out_png,
    )

    summary = {
        "overall_status": overall_status,
        "decision": decision,
        "n_runs": int(len(rows_sorted)),
        "status_counts": counts,
        "tau50_reference_s": tau50_reference_s,
        "tau50_stats_s": {
            "min": float(np.nanmin(tau50_values)),
            "median": float(np.nanmedian(tau50_values)),
            "max": float(np.nanmax(tau50_values)),
            "cv": tau50_cv,
        },
        "branch_reversal_stats": {
            "min": float(np.nanmin([float(r.get("branch_reversal_fraction", math.nan)) for r in rows_sorted])),
            "median": float(np.nanmedian([float(r.get("branch_reversal_fraction", math.nan)) for r in rows_sorted])),
            "max": float(np.nanmax([float(r.get("branch_reversal_fraction", math.nan)) for r in rows_sorted])),
        },
        "pointer_consensus_stats": {
            "min": float(np.nanmin([float(r.get("pointer_consensus_fraction", math.nan)) for r in rows_sorted])),
            "median": float(np.nanmedian([float(r.get("pointer_consensus_fraction", math.nan)) for r in rows_sorted])),
            "max": float(np.nanmax([float(r.get("pointer_consensus_fraction", math.nan)) for r in rows_sorted])),
        },
        "gate_fail_counts": gate_fail_counts,
    }

    payload = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.quantum.dynamic_collapse_stability_audit.v1",
        "phase": 8,
        "step": "8.7.20",
        "inputs": {
            "base_metrics_json": str(base_metrics_path).replace("\\", "/"),
            "seeds": seeds,
            "env_scales": env_scales,
            "n_trajectories_per_run": int(args.n_trajectories_per_run),
            "tau50_ratio_gate": [float(args.tau50_ratio_min), float(args.tau50_ratio_max)],
            "branch_gates": {
                "min_collapse_fraction": float(args.min_collapse_fraction),
                "min_pointer_consensus": float(args.min_pointer_consensus),
                "max_branch_reversal": float(args.max_branch_reversal),
                "min_post_sign_consistency": float(args.min_post_sign_consistency),
            },
            "reject_gates": {
                "collapse_fraction_lt": float(args.reject_collapse_fraction),
                "branch_reversal_gt": float(args.reject_branch_reversal),
                "post_sign_consistency_lt": float(args.reject_post_sign_consistency),
            },
        },
        "summary": summary,
        "rows": rows_sorted,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    copied: List[str] = []
    for src in [out_json, out_csv, out_png]:
        dst = public_outdir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst).replace("\\", "/"))

    payload["outputs"]["public_copies"] = copied
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, public_outdir / out_json.name)

    try:
        worklog.append_event(
            {
                "event_type": "quantum_measurement_dynamic_collapse_stability_audit",
                "argv": list(sys.argv),
                "outputs": {"audit_json": out_json, "audit_csv": out_csv, "audit_png": out_png},
                "metrics": payload.get("summary", {}),
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
