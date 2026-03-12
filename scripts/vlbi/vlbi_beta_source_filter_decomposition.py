#!/usr/bin/env python3
"""
vlbi_beta_source_filter_decomposition.py

Decompose the all-vs-selected source-filter beta shift for one VLBI session.

Purpose:
- Quantify which source / baseline / time-band groups drive
  beta(all sources) - beta(selected sources).
- Keep analysis inside the same primary-data direct-fit pipeline used by
  vlbi_beta_direct_fit_from_vgosdb.py.

Method:
- Reconstruct the common filtered dataset from vgosDb observables.
- Compute baseline fit on:
  1) all sources
  2) selected sources
- Run drop-one refits on all-sources mask:
  - per source (top-N by removable points)
  - per baseline
  - per time quartile
- Report impact as:
  impact_beta = beta_all - beta_drop_group
  abs_z_impact = |impact_beta| / sqrt(sigma_all^2 + sigma_drop^2)

Output:
- output/vlbi/vlbi_<session>_beta_source_filter_decomposition_summary.csv
- output/vlbi/vlbi_<session>_beta_source_filter_decomposition_components.csv
- output/vlbi/vlbi_<session>_beta_source_filter_decomposition_metrics.json
- output/vlbi/vlbi_<session>_beta_source_filter_decomposition.pdf
- output/vlbi/vlbi_<session>_beta_source_filter_decomposition.png
- Synced copies under output/public/vlbi/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import vlbi_beta_direct_fit_from_vgosdb as core


# Function: Resolve repository root from this script location.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Normalize labels for stable output filenames.

def _slugify(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    return value or "session"


# Function: Parse comma-separated source allowlist.

def _parse_source_allowlist(text: str) -> List[str]:
    return core._parse_source_allowlist(text)


# Function: Convert arrays to finite aligned vectors and common length.

def _prepare_vectors(
    index: Dict[str, List[Path]],
    input_root: Path,
    band_index: int,
    observable_series: str,
    threshold_s: float,
    disable_flag_filter: bool,
) -> Dict[str, object]:
    obs_candidates = core.OBS_FULL_CANDIDATES if str(observable_series) == "full" else core.OBS_FRINGE_CANDIDATES
    obs_x_name, obs_x_file = core._pick_variable_preferred(index, obs_candidates, prefer_substring="_bx")
    obs_s_name, obs_s_file = core._pick_variable_preferred(index, obs_candidates, prefer_substring="_bs")
    theo_name, theo_file = core._pick_variable(index, core.THEO_CANDIDATES)
    bend_name, bend_file = core._pick_variable(index, core.BEND_CANDIDATES)
    sig_x_name, sig_x_file = core._pick_variable_preferred(index, core.SIGMA_CANDIDATES, prefer_substring="_bx")
    sig_s_name, sig_s_file = core._pick_variable_preferred(index, core.SIGMA_CANDIDATES, prefer_substring="_bs")
    flag_name, flag_file = core._pick_variable(index, core.FLAG_CANDIDATES)
    source_name, source_file = core._pick_variable(index, core.SOURCE_CANDIDATES)
    freq_x_name, freq_x_file = core._pick_variable_preferred(index, core.FREQ_GROUP_CANDIDATES, prefer_substring="_bx")
    freq_s_name, freq_s_file = core._pick_variable_preferred(index, core.FREQ_GROUP_CANDIDATES, prefer_substring="_bs")
    baseline_name, baseline_file = core._pick_variable(index, core.BASELINE_CANDIDATES)
    ymdhm_name, ymdhm_file = core._pick_variable_preferred(index, core.YMDHM_CANDIDATES, prefer_substring="observables")
    sec_name, sec_file = core._pick_variable_preferred(index, core.SECOND_CANDIDATES, prefer_substring="observables")
    if ymdhm_name is None or ymdhm_file is None:
        ymdhm_name, ymdhm_file = core._pick_variable(index, core.YMDHM_CANDIDATES)

    if sec_name is None or sec_file is None:
        sec_name, sec_file = core._pick_variable(index, core.SECOND_CANDIDATES)

    if obs_x_name is None or obs_x_file is None:
        raise RuntimeError(f"X-band observable variable not found. tried: {obs_candidates}")

    if obs_s_name is None or obs_s_file is None:
        raise RuntimeError("S-band observable variable not found; ionosphere-free mode requires S and X.")

    if theo_name is None or theo_file is None:
        raise RuntimeError(f"theoretical variable not found. tried: {core.THEO_CANDIDATES}")

    if bend_name is None or bend_file is None:
        raise RuntimeError(f"gravity template variable not found. tried: {core.BEND_CANDIDATES}")

    if source_name is None or source_file is None:
        raise RuntimeError("source variable not found; decomposition requires source labels.")

    if baseline_name is None or baseline_file is None:
        raise RuntimeError("baseline variable not found; decomposition requires baseline labels.")

    if ymdhm_name is None or ymdhm_file is None or sec_name is None or sec_file is None:
        raise RuntimeError("time variables not found; decomposition requires TimeUTC vectors.")

    if freq_x_name is None or freq_x_file is None or freq_s_name is None or freq_s_file is None:
        raise RuntimeError("effective frequency vectors not found; ionosphere-free mode requires FreqGroupIono.")

    obs_x = core._reduce_to_vector_numeric(core._read_variable(obs_x_file, obs_x_name), band_index)
    obs_s = core._reduce_to_vector_numeric(core._read_variable(obs_s_file, obs_s_name), band_index)
    theo = core._reduce_to_vector_numeric(core._read_variable(theo_file, theo_name), band_index)
    bend = core._reduce_to_vector_numeric(core._read_variable(bend_file, bend_name), band_index)
    sigma_x = (
        None
        if sig_x_name is None or sig_x_file is None
        else core._reduce_to_vector_numeric(core._read_variable(sig_x_file, sig_x_name), band_index)
    )
    sigma_s = (
        None
        if sig_s_name is None or sig_s_file is None
        else core._reduce_to_vector_numeric(core._read_variable(sig_s_file, sig_s_name), band_index)
    )
    flag = (
        None
        if flag_name is None or flag_file is None
        else core._reduce_to_vector_flag(core._read_variable(flag_file, flag_name), band_index)
    )
    source_vec = core._read_source_vector(core._read_variable(source_file, source_name))
    baseline_vec = core._read_baseline_vector(core._read_variable(baseline_file, baseline_name))
    freq_x = core._reduce_to_vector_numeric(core._read_variable(freq_x_file, freq_x_name), band_index)
    freq_s = core._reduce_to_vector_numeric(core._read_variable(freq_s_file, freq_s_name), band_index)
    ymdhm_raw = core._read_variable(ymdhm_file, ymdhm_name)
    sec_raw = core._read_variable(sec_file, sec_name)
    time_seconds = core._build_time_seconds(ymdhm_raw, sec_raw)
    if time_seconds is None:
        raise RuntimeError("failed to construct monotonic time vector from TimeUTC variables.")

    arrays_for_n: List[np.ndarray] = [
        obs_x,
        obs_s,
        theo,
        bend,
        source_vec,
        baseline_vec,
        freq_x,
        freq_s,
        np.asarray(time_seconds),
    ]
    if sigma_x is not None:
        arrays_for_n.append(sigma_x)

    if sigma_s is not None:
        arrays_for_n.append(sigma_s)

    if flag is not None:
        arrays_for_n.append(np.asarray(flag))

    n_common = core._align_common_length(arrays_for_n)
    if n_common < 3:
        raise RuntimeError(f"not enough aligned observations: {n_common}")

    obs_x = obs_x[:n_common]
    obs_s = obs_s[:n_common]
    theo = theo[:n_common]
    bend = bend[:n_common]
    source_vec = source_vec[:n_common]
    baseline_vec = baseline_vec[:n_common]
    freq_x = freq_x[:n_common]
    freq_s = freq_s[:n_common]
    time_seconds = np.asarray(time_seconds[:n_common], dtype=np.float64)
    if sigma_x is not None:
        sigma_x = sigma_x[:n_common]

    if sigma_s is not None:
        sigma_s = sigma_s[:n_common]

    if flag is not None:
        flag = flag[:n_common]

    obs_if, iono_valid = core._compute_iono_free_group_delay(
        tau_x=obs_x,
        tau_s=obs_s,
        freq_x_mhz=freq_x,
        freq_s_mhz=freq_s,
    )
    sigma_if: Optional[np.ndarray] = None
    if sigma_x is not None and sigma_s is not None:
        ax = np.square(freq_x)
        ass = np.square(freq_s)
        den = ax - ass
        sigma_if = np.full(n_common, np.nan, dtype=np.float64)
        ok = np.isfinite(ax) & np.isfinite(ass) & (np.abs(den) > 1.0e-9)
        ok &= np.isfinite(sigma_x) & np.isfinite(sigma_s) & (sigma_x > 0.0) & (sigma_s > 0.0)
        if np.any(ok):
            cx = ax[ok] / den[ok]
            cs = ass[ok] / den[ok]
            sigma_if[ok] = np.sqrt(np.square(cx * sigma_x[ok]) + np.square(cs * sigma_s[ok]))

    tau_base = theo - bend
    obs_minus_base = obs_if - tau_base
    mask = np.isfinite(obs_minus_base) & np.isfinite(bend) & np.asarray(iono_valid, dtype=bool)
    mask &= np.abs(bend) >= float(threshold_s)
    if sigma_if is not None:
        mask &= np.isfinite(sigma_if) & (sigma_if > 0.0)

    if flag is not None and not bool(disable_flag_filter):
        if np.issubdtype(np.asarray(flag).dtype, np.number):
            mask &= np.asarray(flag == 0)
        else:
            txt = np.asarray([str(v).strip() for v in flag], dtype=object)
            mask &= np.asarray([(v == "" or v == "0") for v in txt], dtype=bool)

    return {
        "input_root": str(input_root),
        "n_common": int(n_common),
        "obs_minus_base": np.asarray(obs_minus_base, dtype=np.float64),
        "template": np.asarray(bend, dtype=np.float64),
        "sigma": None if sigma_if is None else np.asarray(sigma_if, dtype=np.float64),
        "mask_base": np.asarray(mask, dtype=bool),
        "source_vec": np.asarray(source_vec, dtype=object),
        "baseline_vec": np.asarray(baseline_vec, dtype=object),
        "time_seconds": np.asarray(time_seconds, dtype=np.float64),
        "index": index,
        "nuisance_context": {
            "baseline_variable": {"name": baseline_name, "file": str(baseline_file)},
            "source_variable": {"name": source_name, "file": str(source_file)},
            "time_variable": {
                "ymdhm_name": ymdhm_name,
                "ymdhm_file": str(ymdhm_file),
                "second_name": sec_name,
                "second_file": str(sec_file),
            },
        },
    }


# Function: Execute one weighted fit on the provided boolean mask.

def _fit_with_mask(
    prepared: Dict[str, object],
    mask: np.ndarray,
    nuisance_mode: str,
) -> Optional[Dict[str, object]]:
    mask_use = np.asarray(mask, dtype=bool)
    if int(np.sum(mask_use)) < 3:
        return None

    obs_minus_base = np.asarray(prepared["obs_minus_base"], dtype=np.float64)
    template = np.asarray(prepared["template"], dtype=np.float64)
    sigma = prepared.get("sigma")
    index = prepared["index"]  # type: ignore[assignment]
    n_common = int(prepared["n_common"])
    keep_idx = np.where(mask_use)[0]
    x = template[keep_idx]
    y = obs_minus_base[keep_idx]
    sigma_vec = None if sigma is None else np.asarray(sigma, dtype=np.float64)[keep_idx]
    if sigma_vec is None:
        w = np.ones_like(y, dtype=np.float64)
    else:
        w = 1.0 / np.square(sigma_vec)

    z, nuisance_info = core._build_nuisance_matrix(
        index=index,  # type: ignore[arg-type]
        n_common=n_common,
        keep_idx=keep_idx,
        nuisance_mode=str(nuisance_mode),
    )
    try:
        fit = core._weighted_linear_fit(x=x, y=y, w=w, z=z)
    except Exception:
        return None

    beta_est = float(fit["slope"])
    beta_sigma = float(fit["slope_sigma"])
    return {
        "n_points": int(keep_idx.size),
        "beta_est": beta_est,
        "beta_sigma": beta_sigma,
        "delta_beta": float(beta_est - 1.0),
        "chi2": float(fit["chi2"]),
        "dof": int(core._safe_float(fit.get("dof"), 0.0)),
        "weighted_rmse_s": float(fit["weighted_rmse"]),
        "n_params": int(core._safe_float(fit.get("n_params"), 0.0)),
        "nuisance_info": nuisance_info,
    }


# Function: Score one drop-group refit impact against the all-source baseline fit.

def _drop_impact_row(
    label: str,
    group_type: str,
    n_removed: int,
    fit_all: Dict[str, object],
    fit_drop: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if fit_drop is None:
        return None

    beta_all = float(fit_all["beta_est"])
    sig_all = float(fit_all["beta_sigma"])
    beta_drop = float(fit_drop["beta_est"])
    sig_drop = float(fit_drop["beta_sigma"])
    impact = float(beta_all - beta_drop)
    sigma_comb = float(math.sqrt(max(0.0, (sig_all * sig_all) + (sig_drop * sig_drop))))
    abs_z = float(abs(impact) / sigma_comb) if sigma_comb > 0.0 else math.nan
    return {
        "group_type": group_type,
        "group_label": label,
        "n_removed": int(n_removed),
        "beta_drop": beta_drop,
        "beta_drop_sigma": sig_drop,
        "impact_beta_all_minus_drop": impact,
        "sigma_combined": sigma_comb,
        "abs_z_impact": abs_z,
    }


# Function: Build quartile labels for time-band decomposition.

def _time_quartile_masks(time_seconds: np.ndarray, base_mask: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    t = np.asarray(time_seconds, dtype=np.float64)
    m = np.asarray(base_mask, dtype=bool)
    tv = t[m]
    if tv.size < 8:
        return []

    q = np.quantile(tv, [0.25, 0.5, 0.75])
    q1, q2, q3 = [float(v) for v in q]
    masks: List[Tuple[str, np.ndarray]] = []
    masks.append(("Q1", m & (t <= q1)))
    masks.append(("Q2", m & (t > q1) & (t <= q2)))
    masks.append(("Q3", m & (t > q2) & (t <= q3)))
    masks.append(("Q4", m & (t > q3)))
    return masks


# Function: Plot top source/baseline/time impacts with vector PDF output.

def _plot_components(
    pdf_path: Path,
    png_path: Path,
    source_rows: List[Dict[str, object]],
    baseline_rows: List[Dict[str, object]],
    time_rows: List[Dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.2), sharey=False)
    groups = [("Source", source_rows), ("Baseline", baseline_rows), ("Time quartile", time_rows)]
    for ax, (title, rows) in zip(axes, groups):
        if not rows:
            ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel("|impact beta|")
            ax.grid(True, alpha=0.25)
            continue

        rows_sorted = sorted(rows, key=lambda r: abs(float(r["impact_beta_all_minus_drop"])), reverse=True)[:8]
        labels = [str(r["group_label"]) for r in rows_sorted]
        vals = [abs(float(r["impact_beta_all_minus_drop"])) for r in rows_sorted]
        y = np.arange(len(rows_sorted), dtype=np.float64)
        ax.barh(y, vals, color="tab:blue", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("|impact beta|")
        ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("VLBI beta source-filter decomposition (drop-one refit impact)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Copy generated artifacts to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        if path.exists():
            shutil.copy2(path, dst / path.name)


# Function: Main entrypoint for source-filter decomposition audit.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Decompose VLBI source-filter beta shift by source/baseline/time groups.")
    ap.add_argument("--session", type=str, default="17MAY01XA", help="Session label used in output filenames.")
    ap.add_argument(
        "--input-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb" / "17MAY01XA" / "extracted",
        help="Extracted vgosDb root directory.",
    )
    ap.add_argument(
        "--source-include",
        type=str,
        default="0229+131,0235+164",
        help="Comma-separated source allowlist for the selected-source branch.",
    )
    ap.add_argument(
        "--nuisance-mode",
        type=str,
        default="baseline_intercept_linear",
        choices=["none", "baseline_intercept", "baseline_intercept_linear"],
        help="Nuisance mode used for all refits.",
    )
    ap.add_argument(
        "--observable-series",
        type=str,
        default="full",
        choices=["full", "fringe"],
        help="Observable series passed to reconstruction.",
    )
    ap.add_argument(
        "--min-template-abs",
        type=float,
        default=1.0e-14,
        help="Absolute threshold on |Cal-BendSun template| [s].",
    )
    ap.add_argument(
        "--top-sources",
        type=int,
        default=12,
        help="Maximum number of source groups for drop-one decomposition.",
    )
    ap.add_argument(
        "--top-baselines",
        type=int,
        default=12,
        help="Maximum number of baseline groups for drop-one decomposition.",
    )
    args = ap.parse_args()

    input_root = args.input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    session = str(args.session).strip()
    session_slug = _slugify(session)
    source_allow = _parse_source_allowlist(str(args.source_include))
    if not source_allow:
        raise ValueError("source-include must contain at least one source for decomposition.")

    index = core._scan_netcdf_variables(input_root)
    if not index:
        raise RuntimeError(f"no netCDF files found under: {input_root}")

    prepared = _prepare_vectors(
        index=index,
        input_root=input_root,
        band_index=0,
        observable_series=str(args.observable_series),
        threshold_s=float(args.min_template_abs),
        disable_flag_filter=False,
    )
    base_mask = np.asarray(prepared["mask_base"], dtype=bool)
    source_vec = np.asarray(prepared["source_vec"], dtype=object)
    baseline_vec = np.asarray(prepared["baseline_vec"], dtype=object)
    time_seconds = np.asarray(prepared["time_seconds"], dtype=np.float64)
    allow_set = set(source_allow)
    selected_mask = base_mask & np.asarray([str(s) in allow_set for s in source_vec], dtype=bool)
    added_mask = base_mask & (~selected_mask)
    if int(np.sum(selected_mask)) < 3:
        raise RuntimeError("selected source mask has too few points after base filtering.")

    fit_all = _fit_with_mask(prepared=prepared, mask=base_mask, nuisance_mode=str(args.nuisance_mode))
    fit_selected = _fit_with_mask(prepared=prepared, mask=selected_mask, nuisance_mode=str(args.nuisance_mode))
    if fit_all is None or fit_selected is None:
        raise RuntimeError("failed to estimate all/selected baseline fits.")

    beta_all = float(fit_all["beta_est"])
    beta_sel = float(fit_selected["beta_est"])
    sig_all = float(fit_all["beta_sigma"])
    sig_sel = float(fit_selected["beta_sigma"])
    delta_beta = float(beta_all - beta_sel)
    sigma_comb_delta = float(math.sqrt(max(0.0, (sig_all * sig_all) + (sig_sel * sig_sel))))
    abs_z_delta = float(abs(delta_beta) / sigma_comb_delta) if sigma_comb_delta > 0.0 else math.nan

    source_rows: List[Dict[str, object]] = []
    source_values = np.asarray([str(s) for s in source_vec], dtype=object)
    unique_sources, source_counts = np.unique(source_values[added_mask], return_counts=True)
    order_sources = np.argsort(-source_counts.astype(np.int64))
    top_sources = int(max(1, args.top_sources))
    for idx in order_sources[:top_sources]:
        src = str(unique_sources[idx])
        rm_mask = base_mask & (source_values == src)
        n_removed = int(np.sum(rm_mask))
        if n_removed < 3:
            continue

        fit_drop = _fit_with_mask(prepared=prepared, mask=base_mask & (~rm_mask), nuisance_mode=str(args.nuisance_mode))
        row = _drop_impact_row(
            label=src,
            group_type="source",
            n_removed=n_removed,
            fit_all=fit_all,
            fit_drop=fit_drop,
        )
        if row is not None:
            source_rows.append(row)

    baseline_rows: List[Dict[str, object]] = []
    baseline_values = np.asarray([str(b) for b in baseline_vec], dtype=object)
    unique_baselines, baseline_counts = np.unique(baseline_values[added_mask], return_counts=True)
    order_baselines = np.argsort(-baseline_counts.astype(np.int64))
    top_baselines = int(max(1, args.top_baselines))
    for idx in order_baselines[:top_baselines]:
        bl = str(unique_baselines[idx])
        rm_mask = base_mask & (baseline_values == bl)
        n_removed = int(np.sum(rm_mask))
        if n_removed < 3:
            continue

        fit_drop = _fit_with_mask(prepared=prepared, mask=base_mask & (~rm_mask), nuisance_mode=str(args.nuisance_mode))
        row = _drop_impact_row(
            label=bl,
            group_type="baseline",
            n_removed=n_removed,
            fit_all=fit_all,
            fit_drop=fit_drop,
        )
        if row is not None:
            baseline_rows.append(row)

    time_rows: List[Dict[str, object]] = []
    for q_label, q_mask in _time_quartile_masks(time_seconds=time_seconds, base_mask=base_mask):
        rm_mask = np.asarray(q_mask, dtype=bool)
        n_removed = int(np.sum(rm_mask))
        if n_removed < 3:
            continue

        fit_drop = _fit_with_mask(prepared=prepared, mask=base_mask & (~rm_mask), nuisance_mode=str(args.nuisance_mode))
        row = _drop_impact_row(
            label=q_label,
            group_type="time_quartile",
            n_removed=n_removed,
            fit_all=fit_all,
            fit_drop=fit_drop,
        )
        if row is not None:
            time_rows.append(row)

    all_components = source_rows + baseline_rows + time_rows
    max_abs_z_component = max([float(r["abs_z_impact"]) for r in all_components if math.isfinite(float(r["abs_z_impact"]))], default=math.nan)
    max_abs_impact_component = max([abs(float(r["impact_beta_all_minus_drop"])) for r in all_components], default=math.nan)
    status = "watch"
    if math.isfinite(abs_z_delta):
        status = "pass" if abs_z_delta <= 2.0 else ("watch" if abs_z_delta <= 3.0 else "reject")

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"vlbi_{session_slug}_beta_source_filter_decomposition_summary.csv"
    components_csv = out_dir / f"vlbi_{session_slug}_beta_source_filter_decomposition_components.csv"
    metrics_json = out_dir / f"vlbi_{session_slug}_beta_source_filter_decomposition_metrics.json"
    plot_pdf = out_dir / f"vlbi_{session_slug}_beta_source_filter_decomposition.pdf"
    plot_png = out_dir / f"vlbi_{session_slug}_beta_source_filter_decomposition.png"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["session", session])
        w.writerow(["nuisance_mode", str(args.nuisance_mode)])
        w.writerow(["n_points_all", int(np.sum(base_mask))])
        w.writerow(["n_points_selected", int(np.sum(selected_mask))])
        w.writerow(["n_points_added_by_filter", int(np.sum(added_mask))])
        w.writerow(["beta_all", f"{beta_all:.16e}"])
        w.writerow(["beta_all_sigma", f"{sig_all:.16e}"])
        w.writerow(["beta_selected", f"{beta_sel:.16e}"])
        w.writerow(["beta_selected_sigma", f"{sig_sel:.16e}"])
        w.writerow(["delta_beta_all_minus_selected", f"{delta_beta:.16e}"])
        w.writerow(["sigma_combined_delta_beta", f"{sigma_comb_delta:.16e}"])
        w.writerow(["abs_z_delta_beta", f"{abs_z_delta:.16e}"])
        w.writerow(["status", status])
        w.writerow(["max_abs_z_component", f"{max_abs_z_component:.16e}" if math.isfinite(max_abs_z_component) else "nan"])
        w.writerow(["max_abs_impact_component", f"{max_abs_impact_component:.16e}" if math.isfinite(max_abs_impact_component) else "nan"])

    with components_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "group_type",
                "group_label",
                "n_removed",
                "beta_drop",
                "beta_drop_sigma",
                "impact_beta_all_minus_drop",
                "sigma_combined",
                "abs_z_impact",
            ]
        )
        for row in sorted(all_components, key=lambda r: (str(r["group_type"]), -abs(float(r["impact_beta_all_minus_drop"])))):
            w.writerow(
                [
                    str(row["group_type"]),
                    str(row["group_label"]),
                    int(row["n_removed"]),
                    f"{float(row['beta_drop']):.16e}",
                    f"{float(row['beta_drop_sigma']):.16e}",
                    f"{float(row['impact_beta_all_minus_drop']):.16e}",
                    f"{float(row['sigma_combined']):.16e}",
                    f"{float(row['abs_z_impact']):.16e}",
                ]
            )

    _plot_components(
        pdf_path=plot_pdf,
        png_path=plot_png,
        source_rows=source_rows,
        baseline_rows=baseline_rows,
        time_rows=time_rows,
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "status": status,
        "method": {
            "description": "drop-one decomposition on all-source mask using same direct-fit nuisance mode",
            "nuisance_mode": str(args.nuisance_mode),
            "observable_series": str(args.observable_series),
            "source_filter_selected": source_allow,
        },
        "counts": {
            "n_common_raw": int(prepared["n_common"]),
            "n_points_all": int(np.sum(base_mask)),
            "n_points_selected": int(np.sum(selected_mask)),
            "n_points_added_by_filter": int(np.sum(added_mask)),
        },
        "baseline_fit": {
            "all": fit_all,
            "selected": fit_selected,
            "delta_beta_all_minus_selected": delta_beta,
            "sigma_combined_delta_beta": sigma_comb_delta,
            "abs_z_delta_beta": abs_z_delta,
        },
        "decomposition_summary": {
            "max_abs_z_component": max_abs_z_component,
            "max_abs_impact_component": max_abs_impact_component,
            "watch_clear_gate": "pass when abs_z_delta_beta <= 2.0 (same as source-filter sensitivity gate)",
        },
        "components": {
            "source": source_rows,
            "baseline": baseline_rows,
            "time_quartile": time_rows,
        },
        "nuisance_context": prepared["nuisance_context"],
        "outputs": {
            "summary_csv": str(summary_csv),
            "components_csv": str(components_csv),
            "metrics_json": str(metrics_json),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [summary_csv, components_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", summary_csv)
    print("Wrote:", components_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when this file is invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
