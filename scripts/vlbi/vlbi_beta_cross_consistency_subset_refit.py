#!/usr/bin/env python3
"""
vlbi_beta_cross_consistency_subset_refit.py

Roadmap Step 8.7.47.19:
- Re-aggregate VLBI beta estimates on same-period / same-sensitivity subsets.
- Quantify cross-channel |z(beta_vlbi-beta_llr)| decomposition.
- Freeze whether VLBI beta is eligible as a cross-channel comparator.

Inputs:
- output/public/vlbi/vlbi_allsky_beta_consistency_summary.csv
- output/public/llr/llr_kappa_llr_metrics.json

Outputs (default: output/private/vlbi and synced to output/public/vlbi):
- vlbi_beta_cross_consistency_subset_refit.csv
- vlbi_beta_cross_consistency_subset_refit_metrics.json
- vlbi_beta_cross_consistency_subset_refit.pdf
- vlbi_beta_cross_consistency_subset_refit.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[2]
_SESSION_YEAR_RE = re.compile(r"^(\d{2})")


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。
def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_status_from_abs_z` の入出力契約と処理意図を定義する。

def _status_from_abs_z(abs_z: Optional[float]) -> str:
    # 条件分岐: `abs_z is None or not np.isfinite(abs_z)` を満たす経路を評価する。
    if abs_z is None or not np.isfinite(abs_z):
        return "reject"

    # 条件分岐: `abs_z <= 2.0` を満たす経路を評価する。

    if abs_z <= 2.0:
        return "pass"

    # 条件分岐: `abs_z <= 3.0` を満たす経路を評価する。

    if abs_z <= 3.0:
        return "watch"

    return "reject"


# 関数: `_consistency_status` の入出力契約と処理意図を定義する。

def _consistency_status(chi2_dof: Optional[float]) -> str:
    # 条件分岐: `chi2_dof is None or not np.isfinite(chi2_dof)` を満たす経路を評価する。
    if chi2_dof is None or not np.isfinite(chi2_dof):
        return "reject"

    # 条件分岐: `chi2_dof <= 2.0` を満たす経路を評価する。

    if chi2_dof <= 2.0:
        return "pass"

    # 条件分岐: `chi2_dof <= 5.0` を満たす経路を評価する。

    if chi2_dof <= 5.0:
        return "watch"

    return "reject"


# 関数: `_combine_status` の入出力契約と処理意図を定義する。

def _combine_status(statuses: Sequence[str]) -> str:
    norm = [str(v or "").strip().lower() for v in statuses if str(v or "").strip()]
    # 条件分岐: `not norm` を満たす経路を評価する。
    if not norm:
        return "reject"

    # 条件分岐: `any(v == "reject" for v in norm)` を満たす経路を評価する。

    if any(v == "reject" for v in norm):
        return "reject"

    # 条件分岐: `all(v == "pass" for v in norm)` を満たす経路を評価する。

    if all(v == "pass" for v in norm):
        return "pass"

    return "watch"


# 関数: `_parse_float_list` の入出力契約と処理意図を定義する。

def _parse_float_list(csv_like: str) -> List[float]:
    out: List[float] = []
    for tok in str(csv_like).split(","):
        t = tok.strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            continue

        try:
            out.append(float(t))
        except ValueError:
            continue

    return out


# 関数: `_parse_int_list` の入出力契約と処理意図を定義する。

def _parse_int_list(csv_like: str) -> List[int]:
    out: List[int] = []
    for tok in str(csv_like).split(","):
        t = tok.strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            continue

        try:
            out.append(int(float(t)))
        except ValueError:
            continue

    return out


# 関数: `_parse_windows` の入出力契約と処理意図を定義する。

def _parse_windows(spec: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for tok in str(spec).split(","):
        t = tok.strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            continue

        # 条件分岐: `":" not in t` を満たす経路を評価する。

        if ":" not in t:
            continue

        name, rng = t.split(":", 1)
        name = name.strip()
        rng = rng.strip()
        # 条件分岐: `"-" not in rng` を満たす経路を評価する。
        if "-" not in rng:
            continue

        a, b = rng.split("-", 1)
        try:
            y0 = int(a.strip())
            y1 = int(b.strip())
        except ValueError:
            continue

        # 条件分岐: `y1 < y0` を満たす経路を評価する。

        if y1 < y0:
            y0, y1 = y1, y0

        out.append((name, y0, y1))

    return out


# 関数: `_session_year` の入出力契約と処理意図を定義する。

def _session_year(session: str) -> Optional[int]:
    m = _SESSION_YEAR_RE.match(str(session or ""))
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    yy = int(m.group(1))
    # 条件分岐: `yy >= 70` を満たす経路を評価する。
    if yy >= 70:
        return 1900 + yy

    return 2000 + yy


# 関数: `_weighted_mean_and_chi2` の入出力契約と処理意図を定義する。

def _weighted_mean_and_chi2(beta: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(beta) & np.isfinite(sigma) & (sigma > 0)
    b = beta[ok]
    s = sigma[ok]
    # 条件分岐: `len(b) < 2` を満たす経路を評価する。
    if len(b) < 2:
        return {
            "n_valid": int(len(b)),
            "beta_weighted_mean": float("nan"),
            "beta_weighted_sigma": float("nan"),
            "chi2": float("nan"),
            "dof": float("nan"),
            "chi2_dof": float("nan"),
        }

    w = 1.0 / (s * s)
    mu = float(np.sum(w * b) / np.sum(w))
    sig = float(math.sqrt(1.0 / np.sum(w)))
    chi2 = float(np.sum(((b - mu) / s) ** 2))
    dof = float(len(b) - 1)
    return {
        "n_valid": int(len(b)),
        "beta_weighted_mean": mu,
        "beta_weighted_sigma": sig,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": float(chi2 / max(dof, 1.0)),
    }


# 関数: `_load_llr_beta` の入出力契約と処理意図を定義する。

def _load_llr_beta(path: Path) -> Tuple[float, float, str]:
    j = json.loads(path.read_text(encoding="utf-8"))
    fit = j.get("fit") if isinstance(j.get("fit"), dict) else {}
    beta_mapping = fit.get("beta_mapping") if isinstance(fit.get("beta_mapping"), dict) else {}
    beta_est = float(beta_mapping.get("beta_est", fit.get("selected_kappa_est", float("nan"))))
    beta_sigma = float(beta_mapping.get("beta_sigma", fit.get("selected_kappa_sigma", float("nan"))))
    source = str(beta_mapping.get("source", "fit_selected"))
    # 条件分岐: `not np.isfinite(beta_est) or not np.isfinite(beta_sigma) or beta_sigma <= 0` を満たす経路を評価する。
    if not np.isfinite(beta_est) or not np.isfinite(beta_sigma) or beta_sigma <= 0:
        raise RuntimeError(f"invalid llr beta from {path}")

    return beta_est, beta_sigma, source


# 関数: `_evaluate_scenarios` の入出力契約と処理意図を定義する。

def _evaluate_scenarios(
    df: pd.DataFrame,
    llr_beta: float,
    llr_sigma: float,
    thresholds: Sequence[float],
    windows: Sequence[Tuple[str, int, int]],
    min_sessions_list: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        for win_name, y0, y1 in windows:
            for min_sessions in min_sessions_list:
                sub = df[
                    (df["fit_ok"].astype(bool))
                    & (pd.to_numeric(df["max_abs_bendsun_ns"], errors="coerce") >= float(threshold))
                    & (pd.to_numeric(df["session_year"], errors="coerce") >= int(y0))
                    & (pd.to_numeric(df["session_year"], errors="coerce") <= int(y1))
                ].copy()
                n_sessions = int(len(sub))
                scenario_id = f"thr{threshold:g}|{win_name}|min{int(min_sessions)}"
                # 条件分岐: `n_sessions < int(min_sessions)` を満たす経路を評価する。
                if n_sessions < int(min_sessions):
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "threshold_ns": float(threshold),
                            "year_window": win_name,
                            "year_start": int(y0),
                            "year_end": int(y1),
                            "min_sessions": int(min_sessions),
                            "n_sessions": n_sessions,
                            "fit_ok": False,
                            "beta_weighted_mean": float("nan"),
                            "beta_weighted_sigma": float("nan"),
                            "chi2": float("nan"),
                            "dof": float("nan"),
                            "chi2_dof": float("nan"),
                            "subset_consistency_status": "reject",
                            "abs_z_vs_llr": float("nan"),
                            "cross_consistency_status": "reject",
                            "scenario_status": "reject",
                            "sessions": "",
                            "reason": "insufficient_sessions",
                        }
                    )
                    continue

                stats = _weighted_mean_and_chi2(
                    beta=pd.to_numeric(sub["beta_est"], errors="coerce").to_numpy(dtype=float),
                    sigma=pd.to_numeric(sub["beta_sigma"], errors="coerce").to_numpy(dtype=float),
                )
                n_valid = int(stats.get("n_valid", 0))
                # 条件分岐: `n_valid < int(min_sessions)` を満たす経路を評価する。
                if n_valid < int(min_sessions):
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "threshold_ns": float(threshold),
                            "year_window": win_name,
                            "year_start": int(y0),
                            "year_end": int(y1),
                            "min_sessions": int(min_sessions),
                            "n_sessions": n_sessions,
                            "fit_ok": False,
                            "beta_weighted_mean": float("nan"),
                            "beta_weighted_sigma": float("nan"),
                            "chi2": float("nan"),
                            "dof": float("nan"),
                            "chi2_dof": float("nan"),
                            "subset_consistency_status": "reject",
                            "abs_z_vs_llr": float("nan"),
                            "cross_consistency_status": "reject",
                            "scenario_status": "reject",
                            "sessions": "",
                            "reason": "insufficient_valid",
                        }
                    )
                    continue

                beta_v = float(stats["beta_weighted_mean"])
                sigma_v = float(stats["beta_weighted_sigma"])
                denom = math.sqrt(max((sigma_v * sigma_v) + (llr_sigma * llr_sigma), 1e-30))
                abs_z = abs((beta_v - llr_beta) / denom) if denom > 0 else float("nan")
                subset_status = _consistency_status(float(stats["chi2_dof"]))
                cross_status = _status_from_abs_z(abs_z)
                scenario_status = _combine_status([subset_status, cross_status])
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "threshold_ns": float(threshold),
                        "year_window": win_name,
                        "year_start": int(y0),
                        "year_end": int(y1),
                        "min_sessions": int(min_sessions),
                        "n_sessions": n_sessions,
                        "fit_ok": True,
                        "beta_weighted_mean": beta_v,
                        "beta_weighted_sigma": sigma_v,
                        "chi2": float(stats["chi2"]),
                        "dof": float(stats["dof"]),
                        "chi2_dof": float(stats["chi2_dof"]),
                        "subset_consistency_status": subset_status,
                        "abs_z_vs_llr": float(abs_z),
                        "cross_consistency_status": cross_status,
                        "scenario_status": scenario_status,
                        "sessions": ",".join(sorted(sub["session"].astype(str).tolist())),
                        "reason": "",
                    }
                )

    out = pd.DataFrame(rows).sort_values(
        ["threshold_ns", "year_start", "year_end", "min_sessions", "n_sessions"],
        ascending=[True, True, True, True, False],
    )
    return out.reset_index(drop=True)


# 関数: `_pick_reference_rows` の入出力契約と処理意図を定義する。

def _pick_reference_rows(df: pd.DataFrame) -> Dict[str, Any]:
    fit_ok = df[df["fit_ok"].astype(bool)].copy()
    baseline = df[
        (pd.to_numeric(df["threshold_ns"], errors="coerce") == 10.0)
        & (df["year_window"].astype(str) == "all")
        & (pd.to_numeric(df["min_sessions"], errors="coerce") == 3)
    ].copy()
    baseline_row = baseline.iloc[0].to_dict() if len(baseline) > 0 else None
    best_any_row = None
    best_operational_row = None
    # 条件分岐: `len(fit_ok) > 0` を満たす経路を評価する。
    if len(fit_ok) > 0:
        fit_ok = fit_ok[np.isfinite(pd.to_numeric(fit_ok["abs_z_vs_llr"], errors="coerce"))].copy()
        # 条件分岐: `len(fit_ok) > 0` を満たす経路を評価する。
        if len(fit_ok) > 0:
            fit_ok = fit_ok.sort_values(["abs_z_vs_llr", "chi2_dof", "threshold_ns", "year_start"])
            best_any_row = fit_ok.iloc[0].to_dict()
            op = fit_ok[pd.to_numeric(fit_ok["min_sessions"], errors="coerce") >= 3].copy()
            # 条件分岐: `len(op) > 0` を満たす経路を評価する。
            if len(op) > 0:
                best_operational_row = op.iloc[0].to_dict()

    return {
        "baseline": baseline_row,
        "best_any": best_any_row,
        "best_operational": best_operational_row,
    }


# 関数: `_eligibility_from_best_operational` の入出力契約と処理意図を定義する。

def _eligibility_from_best_operational(best_operational: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # 条件分岐: `not isinstance(best_operational, dict)` を満たす経路を評価する。
    if not isinstance(best_operational, dict):
        return {
            "eligible": False,
            "status": "reject",
            "reason": "no_operational_scenario",
        }

    subset_status = str(best_operational.get("subset_consistency_status", "reject"))
    n_sessions = int(best_operational.get("n_sessions", 0))
    # 条件分岐: `n_sessions < 3` を満たす経路を評価する。
    if n_sessions < 3:
        return {
            "eligible": False,
            "status": "reject",
            "reason": "operational_session_count_lt_3",
        }

    # 条件分岐: `subset_status == "reject"` を満たす経路を評価する。

    if subset_status == "reject":
        return {
            "eligible": False,
            "status": "reject",
            "reason": "subset_consistency_reject",
        }

    return {
        "eligible": True,
        "status": "pass" if subset_status == "pass" else "watch",
        "reason": "operational_consistency_non_reject",
    }


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.2), height_ratios=[1.2, 1.0])
    ax0 = axes[0]
    q = df[df["fit_ok"].astype(bool)].copy()
    q = q[np.isfinite(pd.to_numeric(q["abs_z_vs_llr"], errors="coerce"))].copy()
    # 条件分岐: `q.empty` を満たす経路を評価する。
    if q.empty:
        ax0.text(0.5, 0.5, "no valid scenarios", transform=ax0.transAxes, ha="center", va="center")
        ax0.set_axis_off()
    else:
        q = q.sort_values(["abs_z_vs_llr"], ascending=[False]).head(20).copy()
        labels = q["scenario_id"].astype(str).tolist()
        x = np.arange(len(labels), dtype=float)
        y = pd.to_numeric(q["abs_z_vs_llr"], errors="coerce").to_numpy(dtype=float)
        ax0.bar(x, y, color="#1f77b4", alpha=0.82)
        ax0.axhline(2.0, color="#999999", linestyle="--", linewidth=1.0)
        ax0.axhline(3.0, color="#999999", linestyle="--", linewidth=1.0)
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, rotation=70, ha="right", fontsize=8.0)
        ax0.set_ylabel("|z(beta_vlbi-beta_llr)|")
        ax0.set_title("Cross-channel z by VLBI subset scenario")
        ax0.grid(axis="y", alpha=0.2)

    ax1 = axes[1]
    q2 = df[df["fit_ok"].astype(bool)].copy()
    q2 = q2[np.isfinite(pd.to_numeric(q2["chi2_dof"], errors="coerce"))].copy()
    # 条件分岐: `q2.empty` を満たす経路を評価する。
    if q2.empty:
        ax1.text(0.5, 0.5, "no valid scenarios", transform=ax1.transAxes, ha="center", va="center")
        ax1.set_axis_off()
    else:
        q2 = q2.sort_values(["chi2_dof"], ascending=[False]).head(20).copy()
        labels = q2["scenario_id"].astype(str).tolist()
        x = np.arange(len(labels), dtype=float)
        y = pd.to_numeric(q2["chi2_dof"], errors="coerce").to_numpy(dtype=float)
        ax1.bar(x, y, color="#ff7f0e", alpha=0.82)
        ax1.axhline(2.0, color="#999999", linestyle="--", linewidth=1.0)
        ax1.axhline(5.0, color="#999999", linestyle="--", linewidth=1.0)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=70, ha="right", fontsize=8.0)
        ax1.set_ylabel("chi2/dof")
        ax1.set_title("VLBI subset self-consistency by scenario")
        ax1.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_sync_outputs_to_public` の入出力契約と処理意図を定義する。

def _sync_outputs_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[str]:
    public_root.mkdir(parents=True, exist_ok=True)
    synced: List[str] = []
    for p in paths:
        try:
            rel = p.resolve().relative_to(private_root.resolve())
        except Exception:
            rel = Path(p.name)

        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        synced.append(_safe_rel(dst, _ROOT))

    return synced


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="VLBI subset re-aggregation for beta cross-consistency decomposition.")
    ap.add_argument(
        "--vlbi-summary-csv",
        type=str,
        default=str(_ROOT / "output" / "public" / "vlbi" / "vlbi_allsky_beta_consistency_summary.csv"),
    )
    ap.add_argument(
        "--llr-metrics-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "llr" / "llr_kappa_llr_metrics.json"),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "vlbi"),
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "vlbi"),
    )
    ap.add_argument("--thresholds", type=str, default="10,12,15,20")
    ap.add_argument("--min-sessions", type=str, default="2,3")
    ap.add_argument(
        "--year-windows",
        type=str,
        default="all:2000-2099,2017_2018:2017-2018,le2020:2000-2020,ge2020:2020-2099,ge2021:2021-2099,2021_2022:2021-2022",
    )
    args = ap.parse_args()

    vlbi_summary_csv = Path(str(args.vlbi_summary_csv))
    llr_metrics_json = Path(str(args.llr_metrics_json))
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    # 条件分岐: `not vlbi_summary_csv.is_absolute()` を満たす経路を評価する。
    if not vlbi_summary_csv.is_absolute():
        vlbi_summary_csv = (_ROOT / vlbi_summary_csv).resolve()

    # 条件分岐: `not llr_metrics_json.is_absolute()` を満たす経路を評価する。

    if not llr_metrics_json.is_absolute():
        llr_metrics_json = (_ROOT / llr_metrics_json).resolve()

    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。

    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()

    # 条件分岐: `not public_dir.is_absolute()` を満たす経路を評価する。

    if not public_dir.is_absolute():
        public_dir = (_ROOT / public_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _parse_float_list(str(args.thresholds))
    min_sessions_list = _parse_int_list(str(args.min_sessions))
    windows = _parse_windows(str(args.year_windows))
    # 条件分岐: `not thresholds` を満たす経路を評価する。
    if not thresholds:
        raise RuntimeError("empty thresholds")

    # 条件分岐: `not min_sessions_list` を満たす経路を評価する。

    if not min_sessions_list:
        raise RuntimeError("empty min-sessions")

    # 条件分岐: `not windows` を満たす経路を評価する。

    if not windows:
        raise RuntimeError("empty year-windows")

    df = pd.read_csv(vlbi_summary_csv)
    # 条件分岐: `"session" not in df.columns` を満たす経路を評価する。
    if "session" not in df.columns:
        raise RuntimeError(f"missing 'session' column in {vlbi_summary_csv}")

    df["session_year"] = df["session"].astype(str).map(_session_year)
    llr_beta, llr_sigma, llr_source = _load_llr_beta(llr_metrics_json)

    subset_df = _evaluate_scenarios(
        df=df,
        llr_beta=llr_beta,
        llr_sigma=llr_sigma,
        thresholds=thresholds,
        windows=windows,
        min_sessions_list=min_sessions_list,
    )
    refs = _pick_reference_rows(subset_df)
    baseline = refs.get("baseline")
    best_any = refs.get("best_any")
    best_operational = refs.get("best_operational")
    eligibility = _eligibility_from_best_operational(best_operational if isinstance(best_operational, dict) else None)

    delta_best_any = float("nan")
    delta_best_op = float("nan")
    base_abs_z = float("nan")
    # 条件分岐: `isinstance(baseline, dict) and np.isfinite(float(baseline.get("abs_z_vs_llr",...` を満たす経路を評価する。
    if isinstance(baseline, dict) and np.isfinite(float(baseline.get("abs_z_vs_llr", float("nan")))):
        base_abs_z = float(baseline.get("abs_z_vs_llr"))
        # 条件分岐: `isinstance(best_any, dict) and np.isfinite(float(best_any.get("abs_z_vs_llr",...` を満たす経路を評価する。
        if isinstance(best_any, dict) and np.isfinite(float(best_any.get("abs_z_vs_llr", float("nan")))):
            delta_best_any = base_abs_z - float(best_any.get("abs_z_vs_llr"))

        # 条件分岐: `isinstance(best_operational, dict) and np.isfinite(float(best_operational.get...` を満たす経路を評価する。

        if isinstance(best_operational, dict) and np.isfinite(float(best_operational.get("abs_z_vs_llr", float("nan")))):
            delta_best_op = base_abs_z - float(best_operational.get("abs_z_vs_llr"))

    csv_path = out_dir / "vlbi_beta_cross_consistency_subset_refit.csv"
    metrics_path = out_dir / "vlbi_beta_cross_consistency_subset_refit_metrics.json"
    plot_pdf = out_dir / "vlbi_beta_cross_consistency_subset_refit.pdf"
    plot_png = out_dir / "vlbi_beta_cross_consistency_subset_refit.png"
    subset_df.to_csv(csv_path, index=False)
    _write_plot(subset_df, out_pdf=plot_pdf, out_png=plot_png)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.19"},
        "input": {
            "vlbi_summary_csv": _safe_rel(vlbi_summary_csv, _ROOT),
            "llr_metrics_json": _safe_rel(llr_metrics_json, _ROOT),
            "n_vlbi_rows": int(len(df)),
            "thresholds": [float(v) for v in thresholds],
            "min_sessions": [int(v) for v in min_sessions_list],
            "year_windows": [{"name": n, "start": int(a), "end": int(b)} for n, a, b in windows],
        },
        "llr_reference": {
            "beta_est": float(llr_beta),
            "beta_sigma": float(llr_sigma),
            "source": llr_source,
        },
        "baseline_scenario": baseline,
        "best_any_scenario": best_any,
        "best_operational_scenario": best_operational,
        "improvement_vs_baseline": {
            "baseline_abs_z": base_abs_z,
            "delta_abs_z_best_any": delta_best_any,
            "delta_abs_z_best_operational": delta_best_op,
        },
        "vlbi_beta_comparator_eligibility": eligibility,
        "outputs": {
            "subset_csv": _safe_rel(csv_path, _ROOT),
            "plot_pdf": _safe_rel(plot_pdf, _ROOT),
            "plot_png": _safe_rel(plot_png, _ROOT),
        },
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    produced = [csv_path, metrics_path, plot_pdf, plot_png]
    synced = _sync_outputs_to_public(produced, private_root=out_dir, public_root=public_dir)
    print(f"[ok] wrote: {csv_path}")
    print(f"[ok] wrote: {metrics_path}")
    print(f"[ok] wrote: {plot_pdf}")
    print(f"[ok] wrote: {plot_png}")
    print(f"[ok] synced_to_public: {len(synced)} files")
    print(
        f"[summary] baseline_abs_z={base_abs_z} "
        f"best_any_delta={delta_best_any} "
        f"best_operational_delta={delta_best_op} "
        f"eligible={eligibility.get('eligible')} "
        f"eligibility_status={eligibility.get('status')} "
        f"reason={eligibility.get('reason')}"
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
