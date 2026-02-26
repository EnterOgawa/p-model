from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


PHASE = 7
STEP = "7.16.19"
Z_GATE = 3.0


def _parse_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")

    return out if math.isfinite(out) else float("nan")


def _safe_median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return float(median(finite))


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        # 条件分岐: `not rows` を満たす経路を評価する。
        if not rows:
            f.write("")
            return

        headers = list(rows[0].keys())
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h) for h in headers])


def _sigma_sys_required_for_gate(*, residual_scale: float, sigma_stat: float, z_gate: float = Z_GATE) -> float:
    # 条件分岐: `not (math.isfinite(residual_scale) and math.isfinite(sigma_stat) and residual...` を満たす経路を評価する。
    if not (math.isfinite(residual_scale) and math.isfinite(sigma_stat) and residual_scale > 0.0 and sigma_stat >= 0.0):
        return float("nan")

    threshold = residual_scale / z_gate
    # 条件分岐: `threshold <= sigma_stat` を満たす経路を評価する。
    if threshold <= sigma_stat:
        return 0.0

    return float(math.sqrt(max(threshold * threshold - sigma_stat * sigma_stat, 0.0)))


def _build_figure(
    *,
    summary_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    scenarios = [str(row["scenario"]) for row in summary_rows]
    median_z = [float(row["median_abs_z"]) for row in summary_rows]
    max_z = [float(row["max_abs_z"]) for row in summary_rows]
    fail_counts = [int(row["n_gate_fail"]) for row in summary_rows]

    channels = sorted({str(r["channel"]) for r in full_rows})
    upper_by_channel: dict[str, float] = {}
    required_by_channel: dict[str, float] = {}
    for channel in channels:
        sub = [r for r in full_rows if str(r["channel"]) == channel]
        upper_by_channel[channel] = max(float(r["sigma_sys_MeV"]) for r in sub)
        required_by_channel[channel] = _safe_median([float(r["sigma_sys_required_for_z3_MeV"]) for r in sub])

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    ax00.bar(scenarios, median_z, color="#4c78a8", alpha=0.9, label="median |z|")
    ax00.scatter(scenarios, max_z, color="#e45756", zorder=3, label="max |z|")
    ax00.axhline(Z_GATE, ls="--", lw=1.2, color="#444444", label="|z| gate=3")
    ax00.set_title("Scenario consistency against 3σ gate")
    ax00.set_ylabel("z = residual/sigma_total")
    ax00.tick_params(axis="x", rotation=25)
    ax00.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax00.legend(loc="upper right", fontsize=8)

    ax01.bar(scenarios, fail_counts, color="#f58518", alpha=0.9)
    ax01.set_title("Gate failures per scenario")
    ax01.set_ylabel("n channels with |z|>3")
    ax01.tick_params(axis="x", rotation=25)
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax10.bar(channels, [upper_by_channel[c] for c in channels], color="#72b7b2", alpha=0.9)
    ax10.set_title("Conservative sigma_sys upper bound by channel")
    ax10.set_ylabel("sigma_sys upper [MeV or unit]")
    ax10.tick_params(axis="x", rotation=25)
    ax10.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax11.bar(channels, [required_by_channel[c] for c in channels], color="#54a24b", alpha=0.9)
    ax11.set_title("Required sigma_sys to reach |z|<=3")
    ax11.set_ylabel("sigma_sys_required [MeV or unit]")
    ax11.tick_params(axis="x", rotation=25)
    ax11.grid(True, ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.16.19: conservative systematic upper bounds", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_stat_summary = out_dir / "nuclear_statistical_error_propagation_output_summary.csv"
    in_pairing_summary = out_dir / "nuclear_pairing_effect_systematics_summary.csv"
    in_sep_summary = out_dir / "nuclear_separation_energy_systematics_summary.csv"
    in_q_summary = out_dir / "nuclear_beta_decay_qvalue_prediction_summary.csv"
    in_beta2_summary = out_dir / "nuclear_deformation_parameter_prediction_summary.csv"

    for p in (in_stat_summary, in_pairing_summary, in_sep_summary, in_q_summary, in_beta2_summary):
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise SystemExit(f"[fail] missing input: {p}")

    sigma_stat: dict[str, float] = {}
    with in_stat_summary.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sigma_stat[str(row["output"])] = _parse_float(row["mc_std"])

    required_sigma_stat_keys = [
        "be_rms_MeV",
        "sep_rms_MeV",
        "q_beta_minus_rms_MeV",
        "q_beta_plus_rms_MeV",
        "beta2_rms",
    ]
    for key in required_sigma_stat_keys:
        # 条件分岐: `key not in sigma_stat or not math.isfinite(sigma_stat[key])` を満たす経路を評価する。
        if key not in sigma_stat or not math.isfinite(sigma_stat[key]):
            raise SystemExit(f"[fail] missing mc_std for {key}")

    residual_scale: dict[str, float] = {}
    with in_pairing_summary.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            # 条件分岐: `str(row.get("group_type")) == "all" and str(row.get("group")) == "all"` を満たす経路を評価する。
            if str(row.get("group_type")) == "all" and str(row.get("group")) == "all":
                residual_scale["be_rms_MeV"] = _parse_float(row.get("median_abs_resid_after_MeV"))
                break

    sep_medians: list[float] = []
    with in_sep_summary.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            group_type = str(row.get("group_type", ""))
            obs = str(row.get("group", ""))
            # 条件分岐: `group_type == "observable" and obs in {"S_n", "S_p", "S_2n", "S_2p"}` を満たす経路を評価する。
            if group_type == "observable" and obs in {"S_n", "S_p", "S_2n", "S_2p"}:
                value = _parse_float(row.get("median_abs_resid_after_MeV"))
                # 条件分岐: `math.isfinite(value)` を満たす経路を評価する。
                if math.isfinite(value):
                    sep_medians.append(float(value))

    residual_scale["sep_rms_MeV"] = _safe_median(sep_medians)

    with in_q_summary.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            group = str(row.get("group"))
            value = _parse_float(row.get("median_abs_resid_after_MeV"))
            # 条件分岐: `group == "beta_minus" and math.isfinite(value)` を満たす経路を評価する。
            if group == "beta_minus" and math.isfinite(value):
                residual_scale["q_beta_minus_rms_MeV"] = float(value)

            # 条件分岐: `group == "beta_plus" and math.isfinite(value)` を満たす経路を評価する。

            if group == "beta_plus" and math.isfinite(value):
                residual_scale["q_beta_plus_rms_MeV"] = float(value)

    with in_beta2_summary.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            # 条件分岐: `str(row.get("group_type")) == "overall" and str(row.get("group")) == "beta2"` を満たす経路を評価する。
            if str(row.get("group_type")) == "overall" and str(row.get("group")) == "beta2":
                value = _parse_float(row.get("median_abs_resid"))
                # 条件分岐: `math.isfinite(value)` を満たす経路を評価する。
                if math.isfinite(value):
                    residual_scale["beta2_rms"] = float(value)

                break

    for key in required_sigma_stat_keys:
        # 条件分岐: `key not in residual_scale or not math.isfinite(residual_scale[key])` を満たす経路を評価する。
        if key not in residual_scale or not math.isfinite(residual_scale[key]):
            raise SystemExit(f"[fail] missing residual scale for {key}")

    source_channel_fraction: dict[str, dict[str, float]] = {
        "pairing_model_mismatch": {
            "be_rms_MeV": 0.06,
            "sep_rms_MeV": 0.08,
            "q_beta_minus_rms_MeV": 0.06,
            "q_beta_plus_rms_MeV": 0.06,
            "beta2_rms": 0.07,
        },
        "shell_magic_region_bias": {
            "be_rms_MeV": 0.05,
            "sep_rms_MeV": 0.10,
            "q_beta_minus_rms_MeV": 0.04,
            "q_beta_plus_rms_MeV": 0.04,
            "beta2_rms": 0.08,
        },
        "dripline_edge_selection": {
            "be_rms_MeV": 0.04,
            "sep_rms_MeV": 0.12,
            "q_beta_minus_rms_MeV": 0.05,
            "q_beta_plus_rms_MeV": 0.05,
            "beta2_rms": 0.05,
        },
        "qvalue_mode_assignment": {
            "be_rms_MeV": 0.00,
            "sep_rms_MeV": 0.00,
            "q_beta_minus_rms_MeV": 0.10,
            "q_beta_plus_rms_MeV": 0.10,
            "beta2_rms": 0.00,
        },
        "beta4_proxy_mapping": {
            "be_rms_MeV": 0.00,
            "sep_rms_MeV": 0.00,
            "q_beta_minus_rms_MeV": 0.00,
            "q_beta_plus_rms_MeV": 0.00,
            "beta2_rms": 0.20,
        },
        "covariance_underestimation": {
            "be_rms_MeV": 0.05,
            "sep_rms_MeV": 0.05,
            "q_beta_minus_rms_MeV": 0.05,
            "q_beta_plus_rms_MeV": 0.05,
            "beta2_rms": 0.05,
        },
    }

    scenarios: list[dict[str, Any]] = [
        {"name": "baseline_stat_only", "mode": "quadrature", "sources": []},
        {"name": "pairing_plus_magic", "mode": "quadrature", "sources": ["pairing_model_mismatch", "shell_magic_region_bias"]},
        {"name": "edge_plus_covariance", "mode": "quadrature", "sources": ["dripline_edge_selection", "covariance_underestimation"]},
        {"name": "q_channel_stress", "mode": "quadrature", "sources": ["qvalue_mode_assignment", "covariance_underestimation"]},
        {"name": "beta2_proxy_stress", "mode": "quadrature", "sources": ["beta4_proxy_mapping", "covariance_underestimation"]},
        {
            "name": "all_sources_quadrature",
            "mode": "quadrature",
            "sources": list(source_channel_fraction.keys()),
        },
        {
            "name": "all_sources_linear_worst",
            "mode": "linear",
            "sources": list(source_channel_fraction.keys()),
        },
    ]

    full_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dominant_bound_rows: list[dict[str, Any]] = []

    by_channel_max_sigma_sys: dict[str, float] = defaultdict(float)

    channels = required_sigma_stat_keys
    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        mode = str(scenario["mode"])
        sources = [str(v) for v in scenario["sources"]]
        scenario_z_values: list[float] = []
        gate_fail = 0
        worst_channel = ""
        worst_abs_z = -1.0
        for channel in channels:
            resid = residual_scale[channel]
            stat = sigma_stat[channel]
            sys_components = []
            for source in sources:
                frac = float(source_channel_fraction[source][channel])
                sys_components.append(abs(frac * resid))

            # 条件分岐: `mode == "linear"` を満たす経路を評価する。

            if mode == "linear":
                sigma_sys = float(sum(sys_components))
            else:
                sigma_sys = float(math.sqrt(sum(v * v for v in sys_components)))

            sigma_total = float(math.sqrt(stat * stat + sigma_sys * sigma_sys))
            z_value = float(resid / sigma_total) if sigma_total > 0.0 else float("nan")
            gate_pass = bool(math.isfinite(z_value) and abs(z_value) <= Z_GATE)
            required_sigma_sys = _sigma_sys_required_for_gate(residual_scale=resid, sigma_stat=stat, z_gate=Z_GATE)
            inflation = float(required_sigma_sys / stat) if (math.isfinite(required_sigma_sys) and stat > 0.0) else float("nan")
            # 条件分岐: `math.isfinite(z_value)` を満たす経路を評価する。
            if math.isfinite(z_value):
                scenario_z_values.append(abs(z_value))
                # 条件分岐: `abs(z_value) > worst_abs_z` を満たす経路を評価する。
                if abs(z_value) > worst_abs_z:
                    worst_abs_z = abs(z_value)
                    worst_channel = channel

            # 条件分岐: `not gate_pass` を満たす経路を評価する。

            if not gate_pass:
                gate_fail += 1

            full_rows.append(
                {
                    "scenario": scenario_name,
                    "combine_mode": mode,
                    "channel": channel,
                    "residual_scale": resid,
                    "sigma_stat": stat,
                    "sigma_sys_MeV": sigma_sys,
                    "sigma_total": sigma_total,
                    "z_abs": abs(z_value) if math.isfinite(z_value) else float("nan"),
                    "gate_pass_abs_z_le_3": gate_pass,
                    "sigma_sys_required_for_z3_MeV": required_sigma_sys,
                    "required_sys_over_stat": inflation,
                    "sources": ",".join(sources),
                }
            )
            by_channel_max_sigma_sys[channel] = max(by_channel_max_sigma_sys[channel], sigma_sys)

        summary_rows.append(
            {
                "scenario": scenario_name,
                "combine_mode": mode,
                "n_channels": len(channels),
                "n_gate_pass": len(channels) - gate_fail,
                "n_gate_fail": gate_fail,
                "median_abs_z": _safe_median(scenario_z_values),
                "max_abs_z": max(scenario_z_values) if scenario_z_values else float("nan"),
                "worst_channel": worst_channel,
                "sources": ",".join(sources),
            }
        )

    for channel in channels:
        required_vals = [float(r["sigma_sys_required_for_z3_MeV"]) for r in full_rows if str(r["channel"]) == channel]
        dominant_bound_rows.append(
            {
                "channel": channel,
                "sigma_sys_upper_bound_worst_case": by_channel_max_sigma_sys[channel],
                "sigma_sys_required_for_z3_median": _safe_median(required_vals),
                "sigma_stat": sigma_stat[channel],
                "residual_scale": residual_scale[channel],
            }
        )

    out_full_csv = out_dir / "nuclear_systematic_error_upper_bounds_full.csv"
    out_summary_csv = out_dir / "nuclear_systematic_error_upper_bounds_summary.csv"
    out_bounds_csv = out_dir / "nuclear_systematic_error_upper_bounds_channel_bounds.csv"
    out_png = out_dir / "nuclear_systematic_error_upper_bounds_quantification.png"
    out_json = out_dir / "nuclear_systematic_error_upper_bounds_metrics.json"

    _write_csv(out_full_csv, full_rows)
    _write_csv(out_summary_csv, summary_rows)
    _write_csv(out_bounds_csv, dominant_bound_rows)
    _build_figure(summary_rows=summary_rows, full_rows=full_rows, out_png=out_png)

    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": PHASE,
                "step": STEP,
                "gate_abs_z_le": Z_GATE,
                "inputs": {
                    "statistical_summary_csv": {"path": str(in_stat_summary), "sha256": _sha256(in_stat_summary)},
                    "pairing_summary_csv": {"path": str(in_pairing_summary), "sha256": _sha256(in_pairing_summary)},
                    "separation_summary_csv": {"path": str(in_sep_summary), "sha256": _sha256(in_sep_summary)},
                    "qvalue_summary_csv": {"path": str(in_q_summary), "sha256": _sha256(in_q_summary)},
                    "deformation_summary_csv": {"path": str(in_beta2_summary), "sha256": _sha256(in_beta2_summary)},
                },
                "channel_residual_scale": residual_scale,
                "channel_sigma_stat": sigma_stat,
                "systematic_source_fraction_of_residual": source_channel_fraction,
                "scenarios": scenarios,
                "scenario_summary": summary_rows,
                "channel_bounds": dominant_bound_rows,
                "outputs": {
                    "full_csv": str(out_full_csv),
                    "summary_csv": str(out_summary_csv),
                    "channel_bounds_csv": str(out_bounds_csv),
                    "figure_png": str(out_png),
                },
                "notes": [
                    "This step fixes conservative systematic upper bounds by scenario composition (quadrature vs linear worst-case).",
                    "Input statistical sigma is taken from Step 7.16.18 Monte Carlo summary.",
                    "Residual scales are fixed from Step 7.16.3/7.16.15/7.16.16/7.16.17 summary outputs.",
                    "Gate logic: abs(z)=abs(residual/sigma_total)<=3.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_full_csv}")
    print(f"  {out_summary_csv}")
    print(f"  {out_bounds_csv}")
    print(f"  {out_png}")
    print(f"  {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
