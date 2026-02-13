#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ks_two_sample_dcrit(alpha: float, n: int, m: int) -> Optional[float]:
    if n <= 0 or m <= 0:
        return None
    c = None
    if abs(alpha - 0.10) < 1e-12:
        c = 1.22
    elif abs(alpha - 0.05) < 1e-12:
        c = 1.36
    elif abs(alpha - 0.01) < 1e-12:
        c = 1.63
    elif abs(alpha - 0.001) < 1e-12:
        c = 1.95
    if c is None:
        return None
    return float(c * math.sqrt((n + m) / (n * m)))


def _autocorr(x: np.ndarray, lag: int) -> Optional[float]:
    if lag <= 0:
        return 1.0
    if x.size <= lag + 2:
        return None
    a = x[:-lag]
    b = x[lag:]
    if a.size < 3 or b.size < 3:
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0.0 or sb <= 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _effective_n_from_rhos(n: int, rhos: Dict[str, Any]) -> Optional[float]:
    if n <= 0:
        return None
    vals = []
    for k, v in rhos.items():
        if not k.startswith("lag="):
            continue
        if not isinstance(v, (int, float)):
            continue
        vals.append(float(v))
    if not vals:
        return None
    denom = 1.0 + 2.0 * sum(vals)
    if denom <= 0:
        return None
    return float(n / denom)


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()

    default_wielgus = root / "output" / "private" / "eht" / "wielgus2022_m3_observed_metrics.json"
    default_out = root / "output" / "private" / "eht" / "wielgus2022_m3_drw_independence_sim_metrics.json"
    default_png = root / "output" / "private" / "eht" / "wielgus2022_m3_drw_independence_sim.png"

    ap = argparse.ArgumentParser(description="Simulate DRW (OU) process and estimate independence of 3h mi3 windows; derive effective n and KS D_crit impact.")
    ap.add_argument("--wielgus-m3-metrics", type=str, default=str(default_wielgus))
    ap.add_argument("--tau-row", type=str, default="FULL_HI", choices=["A1_all_HI", "FULL_HI", "2005-2019"])
    ap.add_argument("--deltaT-hours", type=float, default=3.0)
    ap.add_argument("--dt-seconds", type=float, default=60.0)
    ap.add_argument("--num-windows", type=int, default=2000)
    ap.add_argument("--burnin-windows", type=int, default=200)
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=0.15, help="Lognormal amplitude: F = mu * exp(frac * X), X~OU(var=1).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default=str(default_out))
    ap.add_argument("--out-png", type=str, default=str(default_png))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    w_path = Path(args.wielgus_m3_metrics)
    out_path = Path(args.out)
    out_png = Path(args.out_png)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "inputs": {"wielgus_m3_metrics_json": str(w_path)},
        "params": {
            "tau_row": str(args.tau_row),
            "deltaT_hours": float(args.deltaT_hours),
            "dt_seconds": float(args.dt_seconds),
            "num_windows": int(args.num_windows),
            "burnin_windows": int(args.burnin_windows),
            "max_lag": int(args.max_lag),
            "mu": float(args.mu),
            "frac": float(args.frac),
            "seed": int(args.seed),
        },
        "derived": {},
        "outputs": {"json": str(out_path), "png": str(out_png)},
    }

    if not w_path.exists():
        payload["ok"] = False
        payload["reason"] = "missing_inputs"
        payload["missing"] = [str(w_path)]
        _write_json(out_path, payload)
        print(f"[warn] missing input: {w_path}")
        return 2

    wielgus = _read_json(w_path)
    gp = ((wielgus.get("extracted") or {}).get("gpresults_tau_hours") or {}).get("rows") or {}
    row = gp.get(args.tau_row) if isinstance(gp, dict) else None
    tau_h = None
    if isinstance(row, dict) and isinstance(row.get("tau_h"), (int, float)):
        tau_h = float(row["tau_h"])
    if not (isinstance(tau_h, float) and tau_h > 0):
        payload["ok"] = False
        payload["reason"] = "tau_not_found"
        payload["tau_row"] = str(args.tau_row)
        _write_json(out_path, payload)
        print(f"[warn] tau not found for row={args.tau_row}")
        return 2

    deltaT_h = float(args.deltaT_hours)
    dt_s = float(args.dt_seconds)
    if deltaT_h <= 0 or dt_s <= 0:
        payload["ok"] = False
        payload["reason"] = "invalid_params"
        _write_json(out_path, payload)
        return 2

    steps_per_window = int(round((deltaT_h * 3600.0) / dt_s))
    if steps_per_window < 5:
        payload["ok"] = False
        payload["reason"] = "deltaT_too_short_for_dt"
        payload["steps_per_window"] = steps_per_window
        _write_json(out_path, payload)
        return 2

    total_windows = int(args.num_windows) + int(args.burnin_windows)
    if total_windows < 50:
        payload["ok"] = False
        payload["reason"] = "too_few_windows"
        _write_json(out_path, payload)
        return 2

    rng = np.random.default_rng(int(args.seed))

    # OU recursion with stationary variance 1:
    # X_{t+dt} = a X_t + sqrt(1-a^2) * N(0,1), a=exp(-dt/tau)
    tau_s = tau_h * 3600.0
    a = math.exp(-dt_s / tau_s)
    sig = math.sqrt(max(0.0, 1.0 - a * a))
    n_steps = total_windows * steps_per_window
    x = np.empty(n_steps, dtype=float)
    x[0] = rng.normal()
    eps = rng.normal(size=n_steps - 1)
    for i in range(1, n_steps):
        x[i] = a * x[i - 1] + sig * eps[i - 1]

    mu = float(args.mu)
    frac = float(args.frac)
    if not math.isfinite(mu) or mu <= 0.0 or not math.isfinite(frac) or frac <= 0.0:
        payload["ok"] = False
        payload["reason"] = "invalid_mu_or_frac"
        _write_json(out_path, payload)
        return 2

    flux = mu * np.exp(frac * x)
    flux = flux.reshape(total_windows, steps_per_window)
    w_mean = np.mean(flux, axis=1)
    w_std = np.std(flux, axis=1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = w_std / w_mean
    mi = mi[int(args.burnin_windows) :]
    mi = mi[np.isfinite(mi)]

    if mi.size < 200:
        payload["ok"] = False
        payload["reason"] = "insufficient_finite_windows"
        payload["mi_windows_n"] = int(mi.size)
        _write_json(out_path, payload)
        return 2

    rhos: Dict[str, Any] = {}
    for lag in range(1, int(args.max_lag) + 1):
        r = _autocorr(mi, lag)
        rhos[f"lag={lag}"] = r

    eff_n_42 = _effective_n_from_rhos(42, rhos)
    eff_n_7 = _effective_n_from_rhos(7, rhos)

    dcrit = {}
    for n_obs_name, n_obs in [
        ("n_obs=42", 42),
        ("n_obs=eff42", int(round(eff_n_42)) if isinstance(eff_n_42, float) else None),
        ("n_obs=30", 30),
    ]:
        if not isinstance(n_obs, int) or n_obs <= 0:
            continue
        dcrit[n_obs_name] = {f"n_model={m}": _ks_two_sample_dcrit(0.01, n_obs, m) for m in (9, 18, 28)}

    payload["derived"] = {
        "tau_h": tau_h,
        "steps_per_window": steps_per_window,
        "mi3_windows_n": int(mi.size),
        "mi3_summary": {
            "min": float(np.min(mi)),
            "max": float(np.max(mi)),
            "mean": float(np.mean(mi)),
            "median": float(np.median(mi)),
        },
        "mi3_autocorr_by_lag": rhos,
        "effective_n": {"for_n=42": eff_n_42, "for_n=7": eff_n_7, "note": "n_eff = n / (1 + 2 * sum(rho_lag)) using lags up to max_lag"},
        "ks_dcrit_alpha_0p01_examples": dcrit,
    }

    if not bool(args.no_plot):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            xs = []
            ys = []
            for k, v in rhos.items():
                if not k.startswith("lag="):
                    continue
                try:
                    lag = int(k.split("=", 1)[1])
                except Exception:
                    continue
                if not isinstance(v, (int, float)):
                    continue
                xs.append(lag)
                ys.append(float(v))
            xs, ys = zip(*sorted(zip(xs, ys))) if xs else ([], [])
            ax.axhline(0.0, color="#999999", linewidth=1)
            ax.plot(xs, ys, marker="o")
            ax.set_title("DRW (OU) simulation: autocorr of mi3 (3h windows)")
            ax.set_xlabel("lag (windows)")
            ax.set_ylabel("corr(mi3_t, mi3_{t+lag})")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
        except Exception:
            pass

    _write_json(out_path, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "wielgus2022_m3_drw_independence_sim",
                "outputs": [
                    str(out_path.relative_to(root)).replace("\\", "/"),
                    str(out_png.relative_to(root)).replace("\\", "/"),
                ],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "tau_h": float(tau_h),
                    "effective_n_42": payload.get("derived", {}).get("effective_n", {}).get("for_n=42"),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_path}")
    if out_png.exists():
        print(f"[ok] png : {out_png}")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
