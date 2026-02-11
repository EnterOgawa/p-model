from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = 7
STEP = "7.16.18"
N_MONTE_CARLO = 100000
SIGMA_ALPHA_Q = 0.10
SIGMA_ALPHA_BETA2 = 0.10
RNG_SEED = 71618


def _parse_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("")
            return
        headers = list(rows[0].keys())
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h) for h in headers])


def _write_matrix_csv(path: Path, labels: list[str], matrix: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"] + labels)
        for name, row in zip(labels, matrix):
            writer.writerow([name] + [float(v) for v in row])


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not ordered:
        return float("nan")
    if q <= 0.0:
        return ordered[0]
    if q >= 1.0:
        return ordered[-1]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / float(len(values)))


def _variance(values: list[float], mean_value: float | None = None) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean_value if mean_value is not None else _mean(values)
    return float(sum((float(v) - m) ** 2 for v in values) / float(len(values) - 1))


def _covariance(xs: list[float], ys: list[float], mx: float | None = None, my: float | None = None) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = mx if mx is not None else _mean(xs)
    mean_y = my if my is not None else _mean(ys)
    acc = 0.0
    for x, y in zip(xs, ys):
        acc += (float(x) - mean_x) * (float(y) - mean_y)
    return float(acc / float(len(xs) - 1))


def _moments(r0: list[float], c_n: list[float], c_p: list[float]) -> dict[str, float]:
    n = len(r0)
    if n == 0:
        raise SystemExit("[fail] empty moments input")
    inv_n = 1.0 / float(n)
    a = sum(v * v for v in r0) * inv_n
    b = 2.0 * sum(rv * cn for rv, cn in zip(r0, c_n)) * inv_n
    c = 2.0 * sum(rv * cp for rv, cp in zip(r0, c_p)) * inv_n
    d = sum(cn * cn for cn in c_n) * inv_n
    e = sum(cp * cp for cp in c_p) * inv_n
    f = 2.0 * sum(cn * cp for cn, cp in zip(c_n, c_p)) * inv_n
    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "n": float(n)}


def _rms_from_moments(m: dict[str, float], *, k_n: float, k_p: float, extra_var: float = 0.0) -> float:
    r2 = (
        float(m["a"])
        + float(m["b"]) * k_n
        + float(m["c"]) * k_p
        + float(m["d"]) * (k_n**2)
        + float(m["e"]) * (k_p**2)
        + float(m["f"]) * k_n * k_p
        + float(extra_var)
    )
    return math.sqrt(max(r2, 0.0))


def _dr_dk_n(m: dict[str, float], *, k_n: float, k_p: float, rms: float) -> float:
    if rms <= 0.0:
        return 0.0
    return float((float(m["b"]) + 2.0 * float(m["d"]) * k_n + float(m["f"]) * k_p) / (2.0 * rms))


def _dr_dk_p(m: dict[str, float], *, k_n: float, k_p: float, rms: float) -> float:
    if rms <= 0.0:
        return 0.0
    return float((float(m["c"]) + 2.0 * float(m["e"]) * k_p + float(m["f"]) * k_n) / (2.0 * rms))


def _fit_pairing_parameters(
    *,
    residuals_before: list[float],
    delta_n: list[float],
    delta_p: list[float],
    a_values: list[int],
) -> tuple[float, float, float, float, float]:
    by_a: dict[int, list[float]] = defaultdict(list)
    for a, r in zip(a_values, residuals_before):
        by_a[int(a)].append(float(r))
    med_by_a = {a: _percentile(vals, 0.5) for a, vals in by_a.items()}

    x1: list[float] = []
    x2: list[float] = []
    y: list[float] = []
    for a, r, dn, dp in zip(a_values, residuals_before, delta_n, delta_p):
        if not (math.isfinite(dn) and math.isfinite(dp)):
            continue
        center = med_by_a.get(int(a), 0.0)
        x1.append(float(dn))
        x2.append(float(dp))
        y.append(float(r - center))

    if len(x1) < 16:
        raise SystemExit("[fail] insufficient fit rows for k_n/k_p covariance")

    s11 = sum(v * v for v in x1)
    s22 = sum(v * v for v in x2)
    s12 = sum(v1 * v2 for v1, v2 in zip(x1, x2))
    sy1 = sum(vx * vy for vx, vy in zip(x1, y))
    sy2 = sum(vx * vy for vx, vy in zip(x2, y))

    det = s11 * s22 - s12 * s12
    if abs(det) < 1.0e-12:
        raise SystemExit("[fail] singular normal equation in k_n/k_p fit")

    k_n = (sy1 * s22 - sy2 * s12) / det
    k_p = (sy2 * s11 - sy1 * s12) / det

    sse = 0.0
    for xx1, xx2, yy in zip(x1, x2, y):
        diff = yy - (k_n * xx1 + k_p * xx2)
        sse += diff * diff
    dof = max(len(x1) - 2, 1)
    sigma2 = sse / float(dof)

    inv11 = s22 / det
    inv22 = s11 / det
    inv12 = -s12 / det
    var_k_n = sigma2 * inv11
    var_k_p = sigma2 * inv22
    cov_k_nk_p = sigma2 * inv12
    return float(k_n), float(k_p), float(var_k_n), float(var_k_p), float(cov_k_nk_p)


def _read_pairing_per_nucleus(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_zn: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            n = int(row["N"])
            item = {
                "Z": z,
                "N": n,
                "A": int(row["A"]),
                "B_obs_MeV": _parse_float(row["B_obs_MeV"]),
                "B_pred_before_MeV": _parse_float(row["B_pred_before_MeV"]),
                "delta_n_3pt_MeV": _parse_float(row["delta_n_3pt_MeV"]),
                "delta_p_3pt_MeV": _parse_float(row["delta_p_3pt_MeV"]),
            }
            rows.append(item)
            by_zn[(z, n)] = item
    if not rows:
        raise SystemExit(f"[fail] empty input: {path}")
    return {"rows": rows, "by_zn": by_zn}


def _build_separation_moments(
    *,
    separation_csv: Path,
    pairing_by_zn: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, float]:
    r0: list[float] = []
    c_n: list[float] = []
    c_p: list[float] = []
    with separation_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z_parent"])
            n = int(row["N_parent"])
            obs = str(row["observable"])
            if obs == "S_n":
                key_ref = (z, n - 1)
            elif obs == "S_p":
                key_ref = (z - 1, n)
            elif obs == "S_2n":
                key_ref = (z, n - 2)
            elif obs == "S_2p":
                key_ref = (z - 2, n)
            else:
                continue
            parent = pairing_by_zn.get((z, n))
            ref = pairing_by_zn.get(key_ref)
            if parent is None or ref is None:
                continue
            resid = _parse_float(row["resid_before_MeV"])
            if not math.isfinite(resid):
                continue
            dn_parent = parent["delta_n_3pt_MeV"]
            dn_ref = ref["delta_n_3pt_MeV"]
            dp_parent = parent["delta_p_3pt_MeV"]
            dp_ref = ref["delta_p_3pt_MeV"]
            dn = (dn_parent if math.isfinite(dn_parent) else 0.0) - (dn_ref if math.isfinite(dn_ref) else 0.0)
            dp = (dp_parent if math.isfinite(dp_parent) else 0.0) - (dp_ref if math.isfinite(dp_ref) else 0.0)
            r0.append(float(resid))
            c_n.append(float(dn))
            c_p.append(float(dp))
    if not r0:
        raise SystemExit(f"[fail] no usable separation rows: {separation_csv}")
    return _moments(r0, c_n, c_p)


def _build_q_moments(
    *,
    q_csv: Path,
    pairing_by_zn: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, dict[str, float]]:
    by_channel: dict[str, dict[str, list[float]]] = {
        "beta_minus": {"r0": [], "c_n": [], "c_p": [], "sigma": []},
        "beta_plus": {"r0": [], "c_n": [], "c_p": [], "sigma": []},
    }
    with q_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            channel = str(row["channel"])
            if channel not in by_channel:
                continue
            z = int(row["Z"])
            n = int(row["N"])
            zd = int(row["daughter_Z"])
            nd = int(row["daughter_N"])
            parent = pairing_by_zn.get((z, n))
            daughter = pairing_by_zn.get((zd, nd))
            if parent is None or daughter is None:
                continue
            resid = _parse_float(row["resid_before_MeV"])
            if not math.isfinite(resid):
                continue
            dn_parent = parent["delta_n_3pt_MeV"]
            dn_daughter = daughter["delta_n_3pt_MeV"]
            dp_parent = parent["delta_p_3pt_MeV"]
            dp_daughter = daughter["delta_p_3pt_MeV"]
            dn = (dn_daughter if math.isfinite(dn_daughter) else 0.0) - (dn_parent if math.isfinite(dn_parent) else 0.0)
            dp = (dp_daughter if math.isfinite(dp_daughter) else 0.0) - (dp_parent if math.isfinite(dp_parent) else 0.0)
            sigma = _parse_float(row["q_obs_sigma_MeV"])
            if not math.isfinite(sigma) or sigma <= 0.0:
                sigma = float("nan")
            bucket = by_channel[channel]
            bucket["r0"].append(float(resid))
            bucket["c_n"].append(float(dn))
            bucket["c_p"].append(float(dp))
            bucket["sigma"].append(float(sigma))

    out: dict[str, dict[str, float]] = {}
    for channel, data in by_channel.items():
        if not data["r0"]:
            raise SystemExit(f"[fail] no usable q rows for channel={channel}")
        finite_sigma = [v for v in data["sigma"] if math.isfinite(v) and v > 0.0]
        sigma_fallback = _percentile(finite_sigma, 0.5) if finite_sigma else 0.0
        sigma_sq_mean = _mean([(v if math.isfinite(v) and v > 0.0 else sigma_fallback) ** 2 for v in data["sigma"]])
        out[channel] = {
            **_moments(data["r0"], data["c_n"], data["c_p"]),
            "sigma_sq_mean": float(sigma_sq_mean if math.isfinite(sigma_sq_mean) else 0.0),
        }
    return out


def _build_beta2_moments(beta2_csv: Path) -> dict[str, float]:
    residuals: list[float] = []
    sigmas: list[float] = []
    with beta2_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            resid = _parse_float(row["beta2_resid"])
            if not math.isfinite(resid):
                continue
            sigma = _parse_float(row["beta2_sigma"])
            residuals.append(float(resid))
            sigmas.append(float(sigma) if (math.isfinite(sigma) and sigma > 0.0) else float("nan"))
    if not residuals:
        raise SystemExit(f"[fail] no usable beta2 residual rows: {beta2_csv}")
    finite_sigma = [v for v in sigmas if math.isfinite(v) and v > 0.0]
    sigma_fallback = _percentile(finite_sigma, 0.5) if finite_sigma else 0.0
    sigma_sq_mean = _mean([(v if math.isfinite(v) and v > 0.0 else sigma_fallback) ** 2 for v in sigmas])
    return {
        "a": _mean([r * r for r in residuals]),
        "b": 0.0,
        "c": 0.0,
        "d": 0.0,
        "e": 0.0,
        "f": 0.0,
        "n": float(len(residuals)),
        "sigma_sq_mean": float(sigma_sq_mean if math.isfinite(sigma_sq_mean) else 0.0),
    }


def _build_figure(
    *,
    input_labels: list[str],
    input_cov: list[list[float]],
    output_summary: list[dict[str, Any]],
    contribution_rows: list[dict[str, Any]],
    be_rms_samples: list[float],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    matrix = input_cov
    vmax = max(abs(v) for row in matrix for v in row) if matrix else 1.0
    im = ax00.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax00.set_xticks(range(len(input_labels)))
    ax00.set_yticks(range(len(input_labels)))
    ax00.set_xticklabels(input_labels, rotation=35, ha="right")
    ax00.set_yticklabels(input_labels)
    ax00.set_title("Input covariance matrix")
    fig.colorbar(im, ax=ax00, fraction=0.046, pad=0.04)

    names = [str(r["output"]) for r in output_summary]
    sigmas = [float(r["mc_std"]) for r in output_summary]
    p05 = [float(r["mc_p05"]) for r in output_summary]
    p95 = [float(r["mc_p95"]) for r in output_summary]
    means = [float(r["mc_mean"]) for r in output_summary]
    lower = [max(m - lo, 0.0) for m, lo in zip(means, p05)]
    upper = [max(hi - m, 0.0) for hi, m in zip(p95, means)]
    ax01.bar(names, sigmas, color="#4c78a8", alpha=0.9)
    ax01.errorbar(names, sigmas, yerr=[lower, upper], fmt="none", ecolor="#222222", capsize=3)
    ax01.set_ylabel("sigma from Monte Carlo")
    ax01.set_title("Output uncertainty summary (n=100000)")
    ax01.tick_params(axis="x", rotation=25)
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)

    source_order = ["k_n", "k_p", "alpha_q", "alpha_beta2", "cov_k_n_k_p"]
    source_colors = {
        "k_n": "#4c78a8",
        "k_p": "#f58518",
        "alpha_q": "#54a24b",
        "alpha_beta2": "#e45756",
        "cov_k_n_k_p": "#b279a2",
    }
    by_output: dict[str, dict[str, float]] = defaultdict(dict)
    for row in contribution_rows:
        by_output[str(row["output"])][str(row["source"])] = float(row["pct_of_mc_var_abs"])
    bottoms = [0.0 for _ in names]
    for source in source_order:
        values = [by_output.get(name, {}).get(source, 0.0) for name in names]
        ax10.bar(names, values, bottom=bottoms, color=source_colors[source], alpha=0.9, label=source)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax10.set_ylabel("abs contribution [% of MC var]")
    ax10.set_title("Dominant source decomposition")
    ax10.tick_params(axis="x", rotation=25)
    ax10.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax10.legend(loc="upper right", fontsize=8)

    ax11.hist(be_rms_samples, bins=60, color="#72b7b2", alpha=0.85, edgecolor="none")
    ax11.set_title("B.E. RMS distribution (Monte Carlo)")
    ax11.set_xlabel("rms(B_pred_after - B_obs) [MeV]")
    ax11.set_ylabel("count")
    ax11.grid(True, ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.16.18: statistical error propagation matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_pairing_csv = out_dir / "nuclear_pairing_effect_systematics_per_nucleus.csv"
    in_separation_csv = out_dir / "nuclear_separation_energy_systematics_full.csv"
    in_q_csv = out_dir / "nuclear_beta_decay_qvalue_prediction_full.csv"
    in_beta2_csv = out_dir / "nuclear_deformation_parameter_prediction_full.csv"

    for p in (in_pairing_csv, in_separation_csv, in_q_csv, in_beta2_csv):
        if not p.exists():
            raise SystemExit(f"[fail] missing input: {p}")

    pairing = _read_pairing_per_nucleus(in_pairing_csv)
    pairing_rows = pairing["rows"]
    pairing_by_zn = pairing["by_zn"]

    a_values = [int(r["A"]) for r in pairing_rows]
    residual_before = [float(r["B_pred_before_MeV"] - r["B_obs_MeV"]) for r in pairing_rows]
    delta_n = [float(r["delta_n_3pt_MeV"]) for r in pairing_rows]
    delta_p = [float(r["delta_p_3pt_MeV"]) for r in pairing_rows]

    k_n_mean, k_p_mean, var_k_n, var_k_p, cov_k_nk_p = _fit_pairing_parameters(
        residuals_before=residual_before,
        delta_n=delta_n,
        delta_p=delta_p,
        a_values=a_values,
    )

    dn_safe = [v if math.isfinite(v) else 0.0 for v in delta_n]
    dp_safe = [v if math.isfinite(v) else 0.0 for v in delta_p]
    moments_be = _moments(residual_before, dn_safe, dp_safe)
    moments_sep = _build_separation_moments(separation_csv=in_separation_csv, pairing_by_zn=pairing_by_zn)
    moments_q = _build_q_moments(q_csv=in_q_csv, pairing_by_zn=pairing_by_zn)
    moments_beta2 = _build_beta2_moments(in_beta2_csv)

    input_labels = ["k_n", "k_p", "alpha_q", "alpha_beta2"]
    input_mean = [k_n_mean, k_p_mean, 1.0, 1.0]
    input_cov = [
        [var_k_n, cov_k_nk_p, 0.0, 0.0],
        [cov_k_nk_p, var_k_p, 0.0, 0.0],
        [0.0, 0.0, SIGMA_ALPHA_Q**2, 0.0],
        [0.0, 0.0, 0.0, SIGMA_ALPHA_BETA2**2],
    ]

    if var_k_n <= 0.0 or var_k_p <= 0.0:
        raise SystemExit("[fail] non-positive variance for k_n/k_p")
    l11 = math.sqrt(var_k_n)
    l21 = cov_k_nk_p / l11
    l22_sq = var_k_p - l21 * l21
    if l22_sq <= 0.0:
        l22_sq = 1.0e-15
    l22 = math.sqrt(l22_sq)

    rng = random.Random(RNG_SEED)
    samples_outputs: dict[str, list[float]] = {
        "be_rms_MeV": [],
        "sep_rms_MeV": [],
        "q_beta_minus_rms_MeV": [],
        "q_beta_plus_rms_MeV": [],
        "beta2_rms": [],
    }

    q_minus_sigma_sq = float(moments_q["beta_minus"]["sigma_sq_mean"])
    q_plus_sigma_sq = float(moments_q["beta_plus"]["sigma_sq_mean"])
    beta2_sigma_sq = float(moments_beta2["sigma_sq_mean"])

    for _ in range(N_MONTE_CARLO):
        z1 = rng.gauss(0.0, 1.0)
        z2 = rng.gauss(0.0, 1.0)
        k_n = k_n_mean + l11 * z1
        k_p = k_p_mean + l21 * z1 + l22 * z2
        alpha_q = max(0.0, 1.0 + SIGMA_ALPHA_Q * rng.gauss(0.0, 1.0))
        alpha_beta2 = max(0.0, 1.0 + SIGMA_ALPHA_BETA2 * rng.gauss(0.0, 1.0))

        samples_outputs["be_rms_MeV"].append(
            _rms_from_moments(moments_be, k_n=k_n, k_p=k_p, extra_var=0.0)
        )
        samples_outputs["sep_rms_MeV"].append(
            _rms_from_moments(moments_sep, k_n=k_n, k_p=k_p, extra_var=0.0)
        )
        samples_outputs["q_beta_minus_rms_MeV"].append(
            _rms_from_moments(
                moments_q["beta_minus"],
                k_n=k_n,
                k_p=k_p,
                extra_var=(alpha_q**2) * q_minus_sigma_sq,
            )
        )
        samples_outputs["q_beta_plus_rms_MeV"].append(
            _rms_from_moments(
                moments_q["beta_plus"],
                k_n=k_n,
                k_p=k_p,
                extra_var=(alpha_q**2) * q_plus_sigma_sq,
            )
        )
        samples_outputs["beta2_rms"].append(
            _rms_from_moments(
                moments_beta2,
                k_n=0.0,
                k_p=0.0,
                extra_var=(alpha_beta2**2) * beta2_sigma_sq,
            )
        )

    output_names = list(samples_outputs.keys())
    output_summary_rows: list[dict[str, Any]] = []
    output_means: dict[str, float] = {}
    output_vars: dict[str, float] = {}
    for name in output_names:
        vals = samples_outputs[name]
        mean_val = _mean(vals)
        var_val = _variance(vals, mean_val)
        std_val = math.sqrt(var_val) if math.isfinite(var_val) and var_val >= 0.0 else float("nan")
        output_means[name] = mean_val
        output_vars[name] = var_val
        output_summary_rows.append(
            {
                "output": name,
                "baseline_at_mean_inputs": _rms_from_moments(
                    moments_be if name == "be_rms_MeV" else
                    moments_sep if name == "sep_rms_MeV" else
                    moments_q["beta_minus"] if name == "q_beta_minus_rms_MeV" else
                    moments_q["beta_plus"] if name == "q_beta_plus_rms_MeV" else
                    moments_beta2,
                    k_n=k_n_mean if name != "beta2_rms" else 0.0,
                    k_p=k_p_mean if name != "beta2_rms" else 0.0,
                    extra_var=(
                        q_minus_sigma_sq if name == "q_beta_minus_rms_MeV" else
                        q_plus_sigma_sq if name == "q_beta_plus_rms_MeV" else
                        beta2_sigma_sq if name == "beta2_rms" else
                        0.0
                    ),
                ),
                "mc_mean": mean_val,
                "mc_std": std_val,
                "mc_p05": _percentile(vals, 0.05),
                "mc_p50": _percentile(vals, 0.50),
                "mc_p95": _percentile(vals, 0.95),
            }
        )

    output_cov_matrix: list[list[float]] = []
    for ni in output_names:
        row: list[float] = []
        for nj in output_names:
            row.append(_covariance(samples_outputs[ni], samples_outputs[nj], output_means[ni], output_means[nj]))
        output_cov_matrix.append(row)

    contribution_rows: list[dict[str, Any]] = []
    dominant_rows: list[dict[str, Any]] = []
    var_alpha_q = SIGMA_ALPHA_Q**2
    var_alpha_beta2 = SIGMA_ALPHA_BETA2**2
    for name in output_names:
        if name == "be_rms_MeV":
            m = moments_be
            rms0 = _rms_from_moments(m, k_n=k_n_mean, k_p=k_p_mean, extra_var=0.0)
            d_kn = _dr_dk_n(m, k_n=k_n_mean, k_p=k_p_mean, rms=rms0)
            d_kp = _dr_dk_p(m, k_n=k_n_mean, k_p=k_p_mean, rms=rms0)
            d_aq = 0.0
            d_ab = 0.0
        elif name == "sep_rms_MeV":
            m = moments_sep
            rms0 = _rms_from_moments(m, k_n=k_n_mean, k_p=k_p_mean, extra_var=0.0)
            d_kn = _dr_dk_n(m, k_n=k_n_mean, k_p=k_p_mean, rms=rms0)
            d_kp = _dr_dk_p(m, k_n=k_n_mean, k_p=k_p_mean, rms=rms0)
            d_aq = 0.0
            d_ab = 0.0
        elif name == "q_beta_minus_rms_MeV":
            m = moments_q["beta_minus"]
            rms0 = _rms_from_moments(m, k_n=k_n_mean, k_p=k_p_mean, extra_var=q_minus_sigma_sq)
            d_kn = _dr_dk_n(m, k_n=k_n_mean, k_p=k_p_mean, rms=rms0)
            d_kp = _dr_dk_p(m, k_n=k_n_mean, k_p=k_p_mean, rms=rms0)
            d_aq = (1.0 * q_minus_sigma_sq / rms0) if rms0 > 0.0 else 0.0
            d_ab = 0.0
        elif name == "q_beta_plus_rms_MeV":
            m = moments_q["beta_plus"]
            rms0 = _rms_from_moments(m, k_n=k_n_mean, k_p=k_p_mean, extra_var=q_plus_sigma_sq)
            d_kn = _dr_dk_n(m, k_n=k_n_mean, k_p=k_p_mean, rms=rms0)
            d_kp = _dr_dk_p(m, k_n=k_n_mean, k_p=k_p_mean, rms=rms0)
            d_aq = (1.0 * q_plus_sigma_sq / rms0) if rms0 > 0.0 else 0.0
            d_ab = 0.0
        else:
            m = moments_beta2
            rms0 = _rms_from_moments(m, k_n=0.0, k_p=0.0, extra_var=beta2_sigma_sq)
            d_kn = 0.0
            d_kp = 0.0
            d_aq = 0.0
            d_ab = (1.0 * beta2_sigma_sq / rms0) if rms0 > 0.0 else 0.0

        v_kn = (d_kn**2) * var_k_n
        v_kp = (d_kp**2) * var_k_p
        v_aq = (d_aq**2) * var_alpha_q
        v_ab = (d_ab**2) * var_alpha_beta2
        v_cov = 2.0 * d_kn * d_kp * cov_k_nk_p
        v_mc = output_vars.get(name, float("nan"))
        v_abs_total = abs(v_kn) + abs(v_kp) + abs(v_aq) + abs(v_ab) + abs(v_cov)

        parts = {
            "k_n": v_kn,
            "k_p": v_kp,
            "alpha_q": v_aq,
            "alpha_beta2": v_ab,
            "cov_k_n_k_p": v_cov,
        }
        for source, val in parts.items():
            contribution_rows.append(
                {
                    "output": name,
                    "source": source,
                    "variance_contribution": val,
                    "pct_of_mc_var_signed": (100.0 * val / v_mc) if (math.isfinite(v_mc) and abs(v_mc) > 0.0) else float("nan"),
                    "pct_of_mc_var_abs": (100.0 * abs(val) / abs(v_mc)) if (math.isfinite(v_mc) and abs(v_mc) > 0.0) else float("nan"),
                    "pct_of_abs_sum": (100.0 * abs(val) / v_abs_total) if v_abs_total > 0.0 else float("nan"),
                }
            )
        dominant_source = max(parts.items(), key=lambda kv: abs(kv[1]))[0]
        dominant_rows.append(
            {
                "output": name,
                "mc_variance": v_mc,
                "dominant_source": dominant_source,
                "dominant_variance_contribution": parts[dominant_source],
                "dominant_pct_of_mc_var_abs": (100.0 * abs(parts[dominant_source]) / abs(v_mc))
                if (math.isfinite(v_mc) and abs(v_mc) > 0.0)
                else float("nan"),
                "linearized_variance_sum_signed": v_kn + v_kp + v_aq + v_ab + v_cov,
            }
        )

    out_input_cov_csv = out_dir / "nuclear_statistical_error_propagation_input_covariance.csv"
    out_output_cov_csv = out_dir / "nuclear_statistical_error_propagation_output_covariance.csv"
    out_summary_csv = out_dir / "nuclear_statistical_error_propagation_output_summary.csv"
    out_contrib_csv = out_dir / "nuclear_statistical_error_propagation_contribution_matrix.csv"
    out_dom_csv = out_dir / "nuclear_statistical_error_propagation_dominant_sources.csv"
    out_png = out_dir / "nuclear_statistical_error_propagation_quantification.png"
    out_json = out_dir / "nuclear_statistical_error_propagation_metrics.json"

    _write_matrix_csv(out_input_cov_csv, input_labels, input_cov)
    _write_matrix_csv(out_output_cov_csv, output_names, output_cov_matrix)
    _write_csv(out_summary_csv, output_summary_rows)
    _write_csv(out_contrib_csv, contribution_rows)
    _write_csv(out_dom_csv, dominant_rows)
    _build_figure(
        input_labels=input_labels,
        input_cov=input_cov,
        output_summary=output_summary_rows,
        contribution_rows=contribution_rows,
        be_rms_samples=samples_outputs["be_rms_MeV"],
        out_png=out_png,
    )

    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": PHASE,
                "step": STEP,
                "settings": {
                    "n_monte_carlo": N_MONTE_CARLO,
                    "rng_seed": RNG_SEED,
                    "alpha_q_sigma": SIGMA_ALPHA_Q,
                    "alpha_beta2_sigma": SIGMA_ALPHA_BETA2,
                },
                "inputs": {
                    "pairing_per_nucleus_csv": {"path": str(in_pairing_csv), "sha256": _sha256(in_pairing_csv)},
                    "separation_full_csv": {"path": str(in_separation_csv), "sha256": _sha256(in_separation_csv)},
                    "qvalue_full_csv": {"path": str(in_q_csv), "sha256": _sha256(in_q_csv)},
                    "deformation_full_csv": {"path": str(in_beta2_csv), "sha256": _sha256(in_beta2_csv)},
                },
                "input_parameter_mean": {
                    "k_n": k_n_mean,
                    "k_p": k_p_mean,
                    "alpha_q": 1.0,
                    "alpha_beta2": 1.0,
                },
                "input_covariance": {
                    "labels": input_labels,
                    "matrix": input_cov,
                },
                "output_summary": output_summary_rows,
                "output_covariance": {
                    "labels": output_names,
                    "matrix": output_cov_matrix,
                },
                "dominant_error_sources": dominant_rows,
                "outputs": {
                    "input_covariance_csv": str(out_input_cov_csv),
                    "output_covariance_csv": str(out_output_cov_csv),
                    "output_summary_csv": str(out_summary_csv),
                    "contribution_matrix_csv": str(out_contrib_csv),
                    "dominant_sources_csv": str(out_dom_csv),
                    "figure_png": str(out_png),
                },
                "notes": [
                    "Pairing coefficients (k_n, k_p) covariance is estimated from the Step 7.16.3 centered-residual linear fit.",
                    "Separation and beta-decay channels are propagated through linearized coefficients induced by k_n/k_p on parent-daughter differences.",
                    "alpha_q and alpha_beta2 are nuisance scale parameters for reported statistical sigma channels.",
                    "This step freezes Monte Carlo uncertainty propagation I/F for Step 7.16.18.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_input_cov_csv}")
    print(f"  {out_output_cov_csv}")
    print(f"  {out_summary_csv}")
    print(f"  {out_contrib_csv}")
    print(f"  {out_dom_csv}")
    print(f"  {out_png}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()

