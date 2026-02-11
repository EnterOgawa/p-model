from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


CHANNELS = ("beta_minus", "beta_plus")

CHANNEL_TO_QKEY = {
    "beta_minus": "betaMinus",
    "beta_plus": "positronEmission",
}


TIME_SCALE_TO_S = {
    "ys": 1.0e-24,
    "zs": 1.0e-21,
    "as": 1.0e-18,
    "fs": 1.0e-15,
    "ps": 1.0e-12,
    "ns": 1.0e-9,
    "us": 1.0e-6,
    "µs": 1.0e-6,
    "μs": 1.0e-6,
    "ms": 1.0e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "y": 31557600.0,
    "ky": 31557600.0e3,
    "my": 31557600.0e6,
    "gy": 31557600.0e9,
}


# Atomic-mass constants (u, MeV/u) for Q-value conversion constants.
M_NEUTRON_U = 1.00866491595
M_HYDROGEN_U = 1.00782503223
M_ELECTRON_U = 0.000548579909065
AMU_TO_MEV = 931.49410242

Q_CONST_BETA_MINUS_MEV = float((M_NEUTRON_U - M_HYDROGEN_U) * AMU_TO_MEV)
Q_CONST_BETA_PLUS_MEV = float((M_HYDROGEN_U - M_NEUTRON_U - 2.0 * M_ELECTRON_U) * AMU_TO_MEV)


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
    paired = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(paired) < 3:
        return float("nan")
    xvals = [p[0] for p in paired]
    yvals = [p[1] for p in paired]
    mx = sum(xvals) / float(len(xvals))
    my = sum(yvals) / float(len(yvals))
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in paired)
    return float(cov / math.sqrt(vx * vy))


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


def _parse_halflife_seconds(level: dict[str, Any]) -> float:
    half = level.get("halflife")
    if not isinstance(half, dict):
        return float("nan")
    value = _parse_float(half.get("value"))
    unit = str(half.get("unit", "")).strip().lower()
    if not math.isfinite(value):
        return float("nan")
    scale = TIME_SCALE_TO_S.get(unit)
    if scale is None:
        return float("nan")
    return float(value * scale)


def _extract_observed_modes(level: dict[str, Any]) -> list[str]:
    decay_modes = level.get("decayModes")
    if not isinstance(decay_modes, dict):
        return []
    observed = decay_modes.get("observed")
    if not isinstance(observed, list):
        return []
    out: list[str] = []
    for item in observed:
        if not isinstance(item, dict):
            continue
        mode = str(item.get("mode", "")).strip()
        if mode:
            out.append(mode)
    return out


def _mode_flags(observed_modes: list[str]) -> tuple[bool, bool]:
    mode_upper = [m.upper() for m in observed_modes]
    has_beta_minus = any("B-" in m for m in mode_upper)
    has_beta_plus = any(("EC" in m) or ("B+" in m) for m in mode_upper)
    return has_beta_minus, has_beta_plus


def _read_primary(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"[fail] invalid primary json: {path}")
    out: dict[str, dict[str, Any]] = {}
    for key, item in payload.items():
        if not isinstance(item, dict):
            continue
        z = int(item.get("z", -1))
        n = int(item.get("n", -1))
        a = int(item.get("a", -1))
        levels = item.get("levels")
        if not (z >= 1 and n >= 0 and a >= 2 and isinstance(levels, list) and levels):
            continue
        level0 = levels[0] if isinstance(levels[0], dict) else {}
        observed_modes = _extract_observed_modes(level0)
        has_bm, has_bp = _mode_flags(observed_modes)
        out[str(key)] = {
            "name": str(item.get("name", key)),
            "Z": z,
            "N": n,
            "A": a,
            "half_life_s": _parse_halflife_seconds(level0),
            "observed_modes": observed_modes,
            "has_beta_minus_mode": has_bm,
            "has_beta_plus_mode": has_bp,
        }
    return out


def _read_secondary_q(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"[fail] invalid secondary json: {path}")
    out: dict[str, dict[str, float]] = {}
    for key, item in payload.items():
        if not isinstance(item, dict):
            continue
        qvals = item.get("qValues")
        if not isinstance(qvals, dict):
            continue
        q_entry: dict[str, float] = {}
        for ch, qkey in CHANNEL_TO_QKEY.items():
            raw = qvals.get(qkey)
            if not isinstance(raw, dict):
                q_entry[f"{ch}_obs_MeV"] = float("nan")
                q_entry[f"{ch}_obs_sigma_MeV"] = float("nan")
                continue
            val_keV = _parse_float(raw.get("value"))
            sig_keV = _parse_float(raw.get("uncertainty"))
            q_entry[f"{ch}_obs_MeV"] = float(val_keV / 1000.0) if math.isfinite(val_keV) else float("nan")
            q_entry[f"{ch}_obs_sigma_MeV"] = float(sig_keV / 1000.0) if math.isfinite(sig_keV) else float("nan")
        out[str(key)] = q_entry
    return out


def _read_binding(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            n = int(row["N"])
            out[(z, n)] = {
                "B_obs_MeV": _parse_float(row.get("B_obs_MeV")),
                "B_pred_before_MeV": _parse_float(row.get("B_pred_before_MeV")),
                "B_pred_after_MeV": _parse_float(row.get("B_pred_after_MeV")),
            }
    return out


def _q_pred_from_binding(*, channel: str, b_parent: float, b_daughter: float) -> float:
    if channel == "beta_minus":
        return float(Q_CONST_BETA_MINUS_MEV + b_daughter - b_parent)
    if channel == "beta_plus":
        return float(Q_CONST_BETA_PLUS_MEV + b_daughter - b_parent)
    raise ValueError(f"unsupported channel: {channel}")


def _build_figure(*, rows: list[dict[str, Any]], summary_by_channel: dict[str, dict[str, Any]], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    colors = {"beta_minus": "#4c78a8", "beta_plus": "#f58518"}
    labels = {"beta_minus": "beta- (NuDat/ENSDF)", "beta_plus": "beta+ (NuDat/ENSDF)"}

    for channel in CHANNELS:
        sub = [r for r in rows if str(r["channel"]) == channel and math.isfinite(float(r["q_obs_MeV"])) and math.isfinite(float(r["q_pred_after_MeV"]))]
        xs = [float(r["q_obs_MeV"]) for r in sub]
        ys = [float(r["q_pred_after_MeV"]) for r in sub]
        ax00.scatter(xs, ys, s=10.0, alpha=0.25, color=colors[channel], label=f"{labels[channel]} (n={len(sub)})")
    all_q = [float(r["q_obs_MeV"]) for r in rows if math.isfinite(float(r["q_obs_MeV"]))]
    if all_q:
        q_min = float(min(all_q))
        q_max = float(max(all_q))
        ax00.plot([q_min, q_max], [q_min, q_max], color="#444444", lw=1.0, ls="--")
    ax00.set_xlabel("Q_obs [MeV]")
    ax00.set_ylabel("Q_pred_after [MeV]")
    ax00.set_title("Q-value prediction vs observed")
    ax00.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax00.legend(loc="best", fontsize=8)

    for channel in CHANNELS:
        vals = [
            float(r["resid_after_MeV"])
            for r in rows
            if str(r["channel"]) == channel and math.isfinite(float(r["resid_after_MeV"]))
        ]
        if vals:
            ax01.hist(vals, bins=80, alpha=0.45, label=labels[channel], color=colors[channel])
    ax01.axvline(0.0, color="#444444", lw=1.0)
    ax01.set_xlabel("Q_pred_after - Q_obs [MeV]")
    ax01.set_ylabel("count")
    ax01.set_title("Residual distribution")
    ax01.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax01.legend(loc="best", fontsize=8)

    for channel in CHANNELS:
        sub = [
            r
            for r in rows
            if str(r["channel"]) == channel
            and math.isfinite(float(r["log10_q_obs"]))
            and math.isfinite(float(r["log10_half_life_s"]))
        ]
        xs = [float(r["log10_q_obs"]) for r in sub]
        ys = [float(r["log10_half_life_s"]) for r in sub]
        ax10.scatter(xs, ys, s=10.0, alpha=0.25, color=colors[channel], label=f"{labels[channel]} (n={len(sub)})")
    ax10.set_xlabel("log10(Q_obs / MeV)")
    ax10.set_ylabel("log10(half-life / s)")
    ax10.set_title("Half-life correlation")
    ax10.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax10.legend(loc="best", fontsize=8)

    bars = []
    vals = []
    for channel in CHANNELS:
        stats = summary_by_channel.get(channel, {})
        bars.append(f"{channel}:acc")
        vals.append(float(stats.get("mode_consistency_accuracy", float("nan"))))
        bars.append(f"{channel}:fp")
        vals.append(float(stats.get("mode_false_positive_rate", float("nan"))))
        bars.append(f"{channel}:fn")
        vals.append(float(stats.get("mode_false_negative_rate", float("nan"))))
    ax11.bar(bars, vals, color=["#54a24b", "#e45756", "#72b7b2"] * 2)
    ax11.set_ylim(0.0, 1.0)
    ax11.set_ylabel("fraction")
    ax11.set_title("Decay-path consistency proxy (mode flags)")
    ax11.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.16.16: beta-decay Q-value prediction audit", y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_binding_csv = out_dir / "nuclear_pairing_effect_systematics_per_nucleus.csv"
    in_primary_json = root / "data" / "quantum" / "sources" / "nndc_nudat3_primary_secondary" / "primary.json"
    in_secondary_json = root / "data" / "quantum" / "sources" / "nndc_nudat3_primary_secondary" / "secondary.json"

    for p in (in_binding_csv, in_primary_json, in_secondary_json):
        if not p.exists():
            raise SystemExit(f"[fail] missing required input: {p}")

    binding_by_zn = _read_binding(in_binding_csv)
    primary_by_key = _read_primary(in_primary_json)
    secondary_q_by_key = _read_secondary_q(in_secondary_json)

    rows: list[dict[str, Any]] = []
    for key, meta in primary_by_key.items():
        qmeta = secondary_q_by_key.get(key)
        if qmeta is None:
            continue
        z = int(meta["Z"])
        n = int(meta["N"])
        a = int(meta["A"])
        parent = binding_by_zn.get((z, n))
        if parent is None:
            continue

        for channel in CHANNELS:
            q_obs = float(qmeta.get(f"{channel}_obs_MeV", float("nan")))
            q_sigma = float(qmeta.get(f"{channel}_obs_sigma_MeV", float("nan")))
            if not math.isfinite(q_obs):
                continue

            if channel == "beta_minus":
                zd, nd = z + 1, n - 1
                observed_flag = bool(meta["has_beta_minus_mode"])
            else:
                zd, nd = z - 1, n + 1
                observed_flag = bool(meta["has_beta_plus_mode"])

            daughter = binding_by_zn.get((zd, nd))
            if daughter is None:
                continue

            q_obs_from_binding = _q_pred_from_binding(
                channel=channel,
                b_parent=float(parent["B_obs_MeV"]),
                b_daughter=float(daughter["B_obs_MeV"]),
            )
            q_pred_before = _q_pred_from_binding(
                channel=channel,
                b_parent=float(parent["B_pred_before_MeV"]),
                b_daughter=float(daughter["B_pred_before_MeV"]),
            )
            q_pred_after = _q_pred_from_binding(
                channel=channel,
                b_parent=float(parent["B_pred_after_MeV"]),
                b_daughter=float(daughter["B_pred_after_MeV"]),
            )
            resid_before = float(q_pred_before - q_obs)
            resid_after = float(q_pred_after - q_obs)
            z_before = float(resid_before / q_sigma) if (math.isfinite(q_sigma) and q_sigma > 0.0) else float("nan")
            z_after = float(resid_after / q_sigma) if (math.isfinite(q_sigma) and q_sigma > 0.0) else float("nan")
            half_life_s = float(meta["half_life_s"])
            q_pred_pos = bool(q_pred_after > 0.0)

            rows.append(
                {
                    "nuclide_key": key,
                    "nuclide_name": str(meta["name"]),
                    "Z": z,
                    "N": n,
                    "A": a,
                    "channel": channel,
                    "daughter_Z": zd,
                    "daughter_N": nd,
                    "q_obs_MeV": q_obs,
                    "q_obs_sigma_MeV": q_sigma,
                    "q_obs_from_binding_MeV": q_obs_from_binding,
                    "q_pred_before_MeV": q_pred_before,
                    "q_pred_after_MeV": q_pred_after,
                    "resid_before_MeV": resid_before,
                    "resid_after_MeV": resid_after,
                    "abs_resid_before_MeV": abs(resid_before),
                    "abs_resid_after_MeV": abs(resid_after),
                    "z_resid_before": z_before,
                    "z_resid_after": z_after,
                    "half_life_s": half_life_s,
                    "log10_half_life_s": float(math.log10(half_life_s)) if (math.isfinite(half_life_s) and half_life_s > 0.0) else float("nan"),
                    "log10_q_obs": float(math.log10(q_obs)) if q_obs > 0.0 else float("nan"),
                    "log10_q_pred_after": float(math.log10(q_pred_after)) if q_pred_after > 0.0 else float("nan"),
                    "observed_decay_has_channel": observed_flag,
                    "predicted_q_positive": q_pred_pos,
                    "mode_consistent": bool(observed_flag == q_pred_pos),
                }
            )

    if not rows:
        raise SystemExit("[fail] no beta-decay rows produced")

    rows = sorted(rows, key=lambda r: (str(r["channel"]), int(r["Z"]), int(r["N"])))

    summary_rows: list[dict[str, Any]] = []
    summary_by_channel: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        sub = [r for r in rows if str(r["channel"]) == channel]
        if not sub:
            continue
        resid_before = [float(r["resid_before_MeV"]) for r in sub]
        resid_after = [float(r["resid_after_MeV"]) for r in sub]
        z_after = [float(r["z_resid_after"]) for r in sub if math.isfinite(float(r["z_resid_after"]))]
        q_obs = [float(r["q_obs_MeV"]) for r in sub]
        q_pred_after = [float(r["q_pred_after_MeV"]) for r in sub]
        logq_obs = [float(r["log10_q_obs"]) for r in sub]
        logq_pred = [float(r["log10_q_pred_after"]) for r in sub]
        logt = [float(r["log10_half_life_s"]) for r in sub]

        tp = sum(1 for r in sub if bool(r["predicted_q_positive"]) and bool(r["observed_decay_has_channel"]))
        tn = sum(1 for r in sub if (not bool(r["predicted_q_positive"])) and (not bool(r["observed_decay_has_channel"])))
        fp = sum(1 for r in sub if bool(r["predicted_q_positive"]) and (not bool(r["observed_decay_has_channel"])))
        fn = sum(1 for r in sub if (not bool(r["predicted_q_positive"])) and bool(r["observed_decay_has_channel"]))
        n_mode = tp + tn + fp + fn

        stats = {
            "group_type": "channel",
            "group": channel,
            "n_rows": len(sub),
            "n_sigma_rows": len(z_after),
            "n_half_life_rows": len([v for v in logt if math.isfinite(v)]),
            "median_abs_resid_before_MeV": _safe_median([abs(v) for v in resid_before]),
            "median_abs_resid_after_MeV": _safe_median([abs(v) for v in resid_after]),
            "rms_resid_before_MeV": _rms(resid_before),
            "rms_resid_after_MeV": _rms(resid_after),
            "median_abs_z_after": _safe_median([abs(v) for v in z_after]),
            "n_abs_z_after_gt3": sum(1 for v in z_after if abs(v) > 3.0),
            "pearson_q_obs_vs_q_pred_after": _pearson(q_obs, q_pred_after),
            "pearson_logq_obs_vs_logt_half": _pearson(logq_obs, logt),
            "pearson_logq_pred_vs_logt_half": _pearson(logq_pred, logt),
            "mode_tp": tp,
            "mode_tn": tn,
            "mode_fp": fp,
            "mode_fn": fn,
            "mode_consistency_accuracy": float((tp + tn) / n_mode) if n_mode > 0 else float("nan"),
            "mode_false_positive_rate": float(fp / n_mode) if n_mode > 0 else float("nan"),
            "mode_false_negative_rate": float(fn / n_mode) if n_mode > 0 else float("nan"),
        }
        summary_rows.append(stats)
        summary_by_channel[channel] = stats

    representative_rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        sub = [r for r in rows if str(r["channel"]) == channel and math.isfinite(float(r["abs_resid_after_MeV"]))]
        if not sub:
            continue
        ordered = sorted(sub, key=lambda r: float(r["abs_resid_after_MeV"]))
        best = ordered[:12]
        worst = ordered[-12:]
        for idx, row in enumerate(best, start=1):
            representative_rows.append(
                {
                    "channel": channel,
                    "rank_type": "best",
                    "rank": idx,
                    "nuclide_name": row["nuclide_name"],
                    "Z": row["Z"],
                    "N": row["N"],
                    "A": row["A"],
                    "q_obs_MeV": row["q_obs_MeV"],
                    "q_pred_after_MeV": row["q_pred_after_MeV"],
                    "resid_after_MeV": row["resid_after_MeV"],
                    "abs_resid_after_MeV": row["abs_resid_after_MeV"],
                    "q_obs_sigma_MeV": row["q_obs_sigma_MeV"],
                    "z_resid_after": row["z_resid_after"],
                }
            )
        for idx, row in enumerate(reversed(worst), start=1):
            representative_rows.append(
                {
                    "channel": channel,
                    "rank_type": "worst",
                    "rank": idx,
                    "nuclide_name": row["nuclide_name"],
                    "Z": row["Z"],
                    "N": row["N"],
                    "A": row["A"],
                    "q_obs_MeV": row["q_obs_MeV"],
                    "q_pred_after_MeV": row["q_pred_after_MeV"],
                    "resid_after_MeV": row["resid_after_MeV"],
                    "abs_resid_after_MeV": row["abs_resid_after_MeV"],
                    "q_obs_sigma_MeV": row["q_obs_sigma_MeV"],
                    "z_resid_after": row["z_resid_after"],
                }
            )

    out_full_csv = out_dir / "nuclear_beta_decay_qvalue_prediction_full.csv"
    out_summary_csv = out_dir / "nuclear_beta_decay_qvalue_prediction_summary.csv"
    out_representative_csv = out_dir / "nuclear_beta_decay_qvalue_prediction_representative.csv"
    out_png = out_dir / "nuclear_beta_decay_qvalue_prediction_quantification.png"
    out_json = out_dir / "nuclear_beta_decay_qvalue_prediction_metrics.json"

    _write_csv(out_full_csv, rows)
    _write_csv(out_summary_csv, summary_rows)
    _write_csv(out_representative_csv, representative_rows)
    _build_figure(rows=rows, summary_by_channel=summary_by_channel, out_png=out_png)

    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.16.16",
                "inputs": {
                    "pairing_per_nucleus_csv": {"path": str(in_binding_csv), "sha256": _sha256(in_binding_csv)},
                    "nudat_primary_json": {"path": str(in_primary_json), "sha256": _sha256(in_primary_json)},
                    "nudat_secondary_json": {"path": str(in_secondary_json), "sha256": _sha256(in_secondary_json)},
                },
                "constants": {
                    "M_neutron_u": M_NEUTRON_U,
                    "M_hydrogen_u": M_HYDROGEN_U,
                    "M_electron_u": M_ELECTRON_U,
                    "amu_to_MeV": AMU_TO_MEV,
                    "Q_const_beta_minus_MeV": Q_CONST_BETA_MINUS_MEV,
                    "Q_const_beta_plus_MeV": Q_CONST_BETA_PLUS_MEV,
                },
                "counts": {
                    "n_rows_total": len(rows),
                    "n_rows_beta_minus": len([r for r in rows if str(r["channel"]) == "beta_minus"]),
                    "n_rows_beta_plus": len([r for r in rows if str(r["channel"]) == "beta_plus"]),
                    "n_representative_rows": len(representative_rows),
                },
                "channel_summary": summary_rows,
                "outputs": {
                    "full_csv": str(out_full_csv),
                    "summary_csv": str(out_summary_csv),
                    "representative_csv": str(out_representative_csv),
                    "figure_png": str(out_png),
                },
                "notes": [
                    "Observed Q-values are taken from NuDat secondary.json qValues (betaMinus / positronEmission).",
                    "NuDat values are treated as ENSDF-linked operational observables in this cross-check I/F.",
                    "Predictions use frozen per-nucleus B from Step 7.16.3 and fixed Q-value conversion constants.",
                    "Decay-path consistency is a sign-based proxy (Q_pred_after>0) vs observed mode flags (B-/EC/B+).",
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
    print(f"  {out_representative_csv}")
    print(f"  {out_png}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
