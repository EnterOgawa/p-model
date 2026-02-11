from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


PHASE = 7
STEP = "7.16.20"
INCONSISTENT_MEDIAN_NORM_GATE = 3.0
INCONSISTENT_MAX_NORM_GATE = 5.0
LOG_EPS = 1.0e-12


def _parse_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(median(finite))


def _rms(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return math.sqrt(sum(v * v for v in finite) / float(len(finite)))


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mean_x = sum(xvals) / float(len(xvals))
    mean_y = sum(yvals) / float(len(yvals))
    var_x = sum((x - mean_x) ** 2 for x in xvals)
    var_y = sum((y - mean_y) ** 2 for y in yvals)
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return float(cov / math.sqrt(var_x * var_y))


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


def _key(z: int, n: int, a: int) -> tuple[int, int, int]:
    return (int(z), int(n), int(a))


def _label(symbol: str, a: int, z: int, n: int) -> str:
    if symbol:
        return f"{symbol}-{a}"
    return f"Z{z}N{n}A{a}"


def _build_figure(
    *,
    matrix_channels: list[str],
    corr_abs_log: list[list[float]],
    rows_joined: list[dict[str, Any]],
    inconsistent_rows: list[dict[str, Any]],
    precision_rows: list[dict[str, Any]],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    img = ax00.imshow(corr_abs_log, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax00.set_title("Cross-observable matrix (pearson of log10 abs residual)")
    ax00.set_xticks(range(len(matrix_channels)))
    ax00.set_yticks(range(len(matrix_channels)))
    ax00.set_xticklabels(matrix_channels, rotation=20, ha="right")
    ax00.set_yticklabels(matrix_channels)
    for i in range(len(matrix_channels)):
        for j in range(len(matrix_channels)):
            value = corr_abs_log[i][j]
            if math.isfinite(value):
                ax00.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(img, ax=ax00, fraction=0.046, pad=0.04)

    scatter_rows = [
        r
        for r in rows_joined
        if math.isfinite(float(r["be_norm"]))
        and math.isfinite(float(r["sep_norm"]))
    ]
    ax01.scatter(
        [float(r["be_norm"]) for r in scatter_rows],
        [float(r["sep_norm"]) for r in scatter_rows],
        s=10.0,
        alpha=0.35,
        color="#4c78a8",
    )
    ax01.axvline(1.0, ls="--", lw=1.0, color="#444444")
    ax01.axhline(1.0, ls="--", lw=1.0, color="#444444")
    ax01.set_xlabel("be_norm = abs(resid_BE) / median_abs_BE")
    ax01.set_ylabel("sep_norm = median_abs(resid_S*) / median_abs_S*")
    ax01.set_title("BE vs separation normalized residual overlap")
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)

    top_n = min(12, len(inconsistent_rows))
    top_rows = inconsistent_rows[:top_n]
    ax10.bar(
        [str(r["nuclide"]) for r in top_rows],
        [float(r["consistency_median_norm"]) for r in top_rows],
        color="#e45756",
        alpha=0.9,
    )
    ax10.axhline(INCONSISTENT_MEDIAN_NORM_GATE, ls="--", lw=1.0, color="#444444")
    ax10.set_title("Most inconsistent nuclei (median normalized residual)")
    ax10.set_ylabel("median normalized residual")
    ax10.tick_params(axis="x", rotation=25)
    ax10.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax11.bar(
        [str(r["channel"]) for r in precision_rows],
        [float(r["median_improvement_factor"]) for r in precision_rows],
        color="#54a24b",
        alpha=0.9,
    )
    ax11.axhline(1.0, ls="--", lw=1.0, color="#444444")
    ax11.set_title("Precision demand (median improvement factor to 3σ)")
    ax11.set_ylabel("improvement factor (current_proxy / required)")
    ax11.tick_params(axis="x", rotation=20)
    ax11.grid(True, ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.16.20: independent cross-observable consistency", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_be = out_dir / "nuclear_pairing_effect_systematics_per_nucleus.csv"
    in_radius = out_dir / "nuclear_charge_radius_consistency_full.csv"
    in_sep = out_dir / "nuclear_separation_energy_systematics_full.csv"
    in_q = out_dir / "nuclear_beta_decay_qvalue_prediction_full.csv"
    in_upper = out_dir / "nuclear_systematic_error_upper_bounds_channel_bounds.csv"

    for path in (in_be, in_radius, in_sep, in_q, in_upper):
        if not path.exists():
            raise SystemExit(f"[fail] missing input: {path}")

    be_map: dict[tuple[int, int, int], dict[str, Any]] = {}
    with in_be.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            n = int(row["N"])
            a = int(row["A"])
            key = _key(z, n, a)
            be_map[key] = {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": str(row.get("symbol", "")),
                "parity": str(row.get("parity", "")),
                "is_magic_any": str(row.get("is_magic_any", "False")).strip().lower() in {"true", "1", "yes"},
                "be_resid_after_MeV": _parse_float(row.get("resid_after_MeV")),
                "be_abs_resid_after_MeV": _parse_float(row.get("abs_resid_after_MeV")),
            }

    radius_map: dict[tuple[int, int, int], dict[str, float]] = {}
    with in_radius.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            n = int(row["N"])
            a = int(row["A"])
            key = _key(z, n, a)
            resid = _parse_float(row.get("resid_radius_a13_i_fm"))
            radius_map[key] = {
                "radius_resid_fm": resid,
                "radius_abs_resid_fm": abs(resid) if math.isfinite(resid) else float("nan"),
            }

    sep_acc: dict[tuple[int, int, int], list[float]] = {}
    sep_obs_count: dict[tuple[int, int, int], int] = {}
    with in_sep.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z_parent"])
            n = int(row["N_parent"])
            a = int(row["A_parent"])
            key = _key(z, n, a)
            value = _parse_float(row.get("resid_after_MeV"))
            if not math.isfinite(value):
                continue
            if key not in sep_acc:
                sep_acc[key] = []
                sep_obs_count[key] = 0
            sep_acc[key].append(float(value))
            sep_obs_count[key] += 1
    sep_map: dict[tuple[int, int, int], dict[str, float]] = {}
    for key, values in sep_acc.items():
        abs_values = [abs(v) for v in values]
        sep_map[key] = {
            "sep_resid_median_MeV": _safe_median(values),
            "sep_abs_resid_median_MeV": _safe_median(abs_values),
            "sep_resid_rms_MeV": _rms(values),
            "sep_obs_count": float(sep_obs_count.get(key, 0)),
        }

    q_acc: dict[tuple[int, int, int], list[float]] = {}
    q_channel_count: dict[tuple[int, int, int], int] = {}
    q_mode_count: dict[tuple[int, int, int], dict[str, int]] = {}
    with in_q.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            n = int(row["N"])
            a = int(row["A"])
            key = _key(z, n, a)
            value = _parse_float(row.get("resid_after_MeV"))
            if not math.isfinite(value):
                continue
            if key not in q_acc:
                q_acc[key] = []
                q_channel_count[key] = 0
                q_mode_count[key] = {"beta_minus": 0, "beta_plus": 0}
            q_acc[key].append(float(value))
            q_channel_count[key] += 1
            channel = str(row.get("channel", ""))
            if channel in {"beta_minus", "beta_plus"}:
                q_mode_count[key][channel] += 1
    q_map: dict[tuple[int, int, int], dict[str, float]] = {}
    for key, values in q_acc.items():
        abs_values = [abs(v) for v in values]
        q_map[key] = {
            "q_resid_median_MeV": _safe_median(values),
            "q_abs_resid_median_MeV": _safe_median(abs_values),
            "q_resid_rms_MeV": _rms(values),
            "q_obs_count": float(q_channel_count.get(key, 0)),
            "q_beta_minus_count": float(q_mode_count.get(key, {}).get("beta_minus", 0)),
            "q_beta_plus_count": float(q_mode_count.get(key, {}).get("beta_plus", 0)),
        }

    all_keys = sorted(set(be_map.keys()) | set(radius_map.keys()) | set(sep_map.keys()) | set(q_map.keys()))
    rows_joined: list[dict[str, Any]] = []
    for key in all_keys:
        z, n, a = key
        base = be_map.get(key, {})
        radius = radius_map.get(key, {})
        sep = sep_map.get(key, {})
        q = q_map.get(key, {})

        symbol = str(base.get("symbol", ""))
        nuclide = _label(symbol=symbol, a=a, z=z, n=n)
        row = {
            "nuclide": nuclide,
            "Z": z,
            "N": n,
            "A": a,
            "symbol": symbol,
            "parity": str(base.get("parity", "")),
            "is_magic_any": bool(base.get("is_magic_any", False)),
            "be_resid_after_MeV": _parse_float(base.get("be_resid_after_MeV")),
            "be_abs_resid_after_MeV": _parse_float(base.get("be_abs_resid_after_MeV")),
            "radius_resid_fm": _parse_float(radius.get("radius_resid_fm")),
            "radius_abs_resid_fm": _parse_float(radius.get("radius_abs_resid_fm")),
            "sep_resid_median_MeV": _parse_float(sep.get("sep_resid_median_MeV")),
            "sep_abs_resid_median_MeV": _parse_float(sep.get("sep_abs_resid_median_MeV")),
            "sep_resid_rms_MeV": _parse_float(sep.get("sep_resid_rms_MeV")),
            "sep_obs_count": int(float(sep.get("sep_obs_count", 0.0))),
            "q_resid_median_MeV": _parse_float(q.get("q_resid_median_MeV")),
            "q_abs_resid_median_MeV": _parse_float(q.get("q_abs_resid_median_MeV")),
            "q_resid_rms_MeV": _parse_float(q.get("q_resid_rms_MeV")),
            "q_obs_count": int(float(q.get("q_obs_count", 0.0))),
            "q_beta_minus_count": int(float(q.get("q_beta_minus_count", 0.0))),
            "q_beta_plus_count": int(float(q.get("q_beta_plus_count", 0.0))),
        }
        rows_joined.append(row)

    channel_abs_key = {
        "be": "be_abs_resid_after_MeV",
        "radius": "radius_abs_resid_fm",
        "sep": "sep_abs_resid_median_MeV",
        "q": "q_abs_resid_median_MeV",
    }
    channel_signed_key = {
        "be": "be_resid_after_MeV",
        "radius": "radius_resid_fm",
        "sep": "sep_resid_median_MeV",
        "q": "q_resid_median_MeV",
    }

    channel_scale: dict[str, float] = {}
    for channel, key_name in channel_abs_key.items():
        channel_scale[channel] = _safe_median(
            [float(row[key_name]) for row in rows_joined if math.isfinite(float(row[key_name]))]
        )
        if not math.isfinite(channel_scale[channel]) or channel_scale[channel] <= 0.0:
            raise SystemExit(f"[fail] invalid channel scale for {channel}")

    for row in rows_joined:
        available_norms: list[float] = []
        for channel, key_name in channel_abs_key.items():
            value = _parse_float(row.get(key_name))
            if math.isfinite(value):
                norm_value = float(value / channel_scale[channel])
                row[f"{channel}_norm"] = norm_value
                available_norms.append(norm_value)
            else:
                row[f"{channel}_norm"] = float("nan")
        row["n_channels_available"] = len(available_norms)
        row["consistency_median_norm"] = _safe_median(available_norms)
        row["consistency_max_norm"] = max(available_norms) if available_norms else float("nan")
        row["inconsistent"] = bool(
            len(available_norms) >= 3
            and (
                (math.isfinite(float(row["consistency_median_norm"])) and float(row["consistency_median_norm"]) > INCONSISTENT_MEDIAN_NORM_GATE)
                or (math.isfinite(float(row["consistency_max_norm"])) and float(row["consistency_max_norm"]) > INCONSISTENT_MAX_NORM_GATE)
            )
        )

    inconsistent_rows = sorted(
        [r for r in rows_joined if bool(r["inconsistent"])],
        key=lambda r: (float(r["consistency_median_norm"]), float(r["consistency_max_norm"])),
        reverse=True,
    )
    for rank, row in enumerate(inconsistent_rows, start=1):
        row["inconsistent_rank"] = rank

    matrix_channels = ["be", "radius", "sep", "q"]
    matrix_pair_rows: list[dict[str, Any]] = []
    corr_abs_log_matrix: list[list[float]] = []
    for ch_i in matrix_channels:
        matrix_row: list[float] = []
        for ch_j in matrix_channels:
            xs_signed = [
                float(row[channel_signed_key[ch_i]])
                for row in rows_joined
                if math.isfinite(float(row[channel_signed_key[ch_i]])) and math.isfinite(float(row[channel_signed_key[ch_j]]))
            ]
            ys_signed = [
                float(row[channel_signed_key[ch_j]])
                for row in rows_joined
                if math.isfinite(float(row[channel_signed_key[ch_i]])) and math.isfinite(float(row[channel_signed_key[ch_j]]))
            ]
            xs_log_abs = [
                math.log10(max(float(row[channel_abs_key[ch_i]]), LOG_EPS))
                for row in rows_joined
                if math.isfinite(float(row[channel_abs_key[ch_i]])) and math.isfinite(float(row[channel_abs_key[ch_j]]))
            ]
            ys_log_abs = [
                math.log10(max(float(row[channel_abs_key[ch_j]]), LOG_EPS))
                for row in rows_joined
                if math.isfinite(float(row[channel_abs_key[ch_i]])) and math.isfinite(float(row[channel_abs_key[ch_j]]))
            ]
            n_overlap = len(xs_signed)
            corr_signed = _pearson(xs_signed, ys_signed)
            corr_abs_log = _pearson(xs_log_abs, ys_log_abs)
            matrix_row.append(corr_abs_log if math.isfinite(corr_abs_log) else float("nan"))
            matrix_pair_rows.append(
                {
                    "channel_i": ch_i,
                    "channel_j": ch_j,
                    "n_overlap": n_overlap,
                    "pearson_signed_residual": corr_signed,
                    "pearson_log10_abs_residual": corr_abs_log,
                    "median_abs_i": _safe_median(
                        [float(row[channel_abs_key[ch_i]]) for row in rows_joined if math.isfinite(float(row[channel_abs_key[ch_i]]))]
                    ),
                    "median_abs_j": _safe_median(
                        [float(row[channel_abs_key[ch_j]]) for row in rows_joined if math.isfinite(float(row[channel_abs_key[ch_j]]))]
                    ),
                }
            )
        corr_abs_log_matrix.append(matrix_row)

    upper_scale_map: dict[str, float] = {}
    with in_upper.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            channel_name = str(row.get("channel", ""))
            upper = _parse_float(row.get("sigma_sys_upper_bound_worst_case"))
            if channel_name == "be_rms_MeV":
                upper_scale_map["be"] = upper
            elif channel_name == "sep_rms_MeV":
                upper_scale_map["sep"] = upper
            elif channel_name == "q_beta_minus_rms_MeV":
                upper_scale_map["q"] = upper
            elif channel_name == "beta2_rms":
                upper_scale_map["radius"] = upper

    precision_rows: list[dict[str, Any]] = []
    for channel in matrix_channels:
        values = [
            float(row[channel_abs_key[channel]])
            for row in rows_joined
            if math.isfinite(float(row[channel_abs_key[channel]]))
        ]
        required = [float(v / 3.0) for v in values]
        current_proxy = float(channel_scale[channel])
        improvement = [float(current_proxy / max(v, LOG_EPS)) for v in required]
        precision_rows.append(
            {
                "channel": channel,
                "n": len(values),
                "current_sigma_proxy": current_proxy,
                "sigma_sys_upper_bound_from_7_16_19": _parse_float(upper_scale_map.get(channel)),
                "median_required_sigma_for_3sigma": _safe_median(required),
                "p90_required_sigma_for_3sigma": sorted(required)[int(0.9 * (len(required) - 1))] if required else float("nan"),
                "median_improvement_factor": _safe_median(improvement),
                "p90_improvement_factor": sorted(improvement)[int(0.9 * (len(improvement) - 1))] if improvement else float("nan"),
            }
        )

    out_joined = out_dir / "nuclear_independent_cross_observable_consistency_joined.csv"
    out_matrix = out_dir / "nuclear_independent_cross_observable_consistency_matrix.csv"
    out_inconsistent = out_dir / "nuclear_independent_cross_observable_inconsistent_nuclei.csv"
    out_precision = out_dir / "nuclear_independent_cross_observable_precision_requirements.csv"
    out_png = out_dir / "nuclear_independent_cross_observable_consistency.png"
    out_json = out_dir / "nuclear_independent_cross_observable_consistency_metrics.json"

    _write_csv(out_joined, rows_joined)
    _write_csv(out_matrix, matrix_pair_rows)
    _write_csv(out_inconsistent, inconsistent_rows)
    _write_csv(out_precision, precision_rows)
    _build_figure(
        matrix_channels=matrix_channels,
        corr_abs_log=corr_abs_log_matrix,
        rows_joined=rows_joined,
        inconsistent_rows=inconsistent_rows,
        precision_rows=precision_rows,
        out_png=out_png,
    )

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "step": STEP,
        "inputs": {
            "pairing_per_nucleus_csv": {"path": str(in_be), "sha256": _sha256(in_be)},
            "charge_radius_full_csv": {"path": str(in_radius), "sha256": _sha256(in_radius)},
            "separation_full_csv": {"path": str(in_sep), "sha256": _sha256(in_sep)},
            "qvalue_full_csv": {"path": str(in_q), "sha256": _sha256(in_q)},
            "systematic_upper_bounds_csv": {"path": str(in_upper), "sha256": _sha256(in_upper)},
        },
        "counts": {
            "n_rows_joined": len(rows_joined),
            "n_with_all_4_channels": sum(1 for r in rows_joined if int(r["n_channels_available"]) == 4),
            "n_with_ge_3_channels": sum(1 for r in rows_joined if int(r["n_channels_available"]) >= 3),
            "n_inconsistent": len(inconsistent_rows),
        },
        "gates": {
            "inconsistent_median_norm_gt": INCONSISTENT_MEDIAN_NORM_GATE,
            "inconsistent_max_norm_gt": INCONSISTENT_MAX_NORM_GATE,
        },
        "channel_scale_median_abs": channel_scale,
        "cross_check_matrix_pairs": matrix_pair_rows,
        "precision_requirements": precision_rows,
        "top_inconsistent_nuclei": inconsistent_rows[:20],
        "outputs": {
            "joined_csv": str(out_joined),
            "matrix_csv": str(out_matrix),
            "inconsistent_csv": str(out_inconsistent),
            "precision_csv": str(out_precision),
            "figure_png": str(out_png),
        },
        "notes": [
            "Step 7.16.20 freezes a cross-observable consistency matrix over BE, charge radius, separation energies, and beta-decay Q values.",
            "Inconsistency is flagged when at least three channels are available and median_norm>3 or max_norm>5.",
            "Precision-demand rows report proxy improvement factors to reach a 3-sigma consistency gate under fixed residuals.",
        ],
    }

    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] wrote:")
    print(f"  {out_joined}")
    print(f"  {out_matrix}")
    print(f"  {out_inconsistent}")
    print(f"  {out_precision}")
    print(f"  {out_png}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
