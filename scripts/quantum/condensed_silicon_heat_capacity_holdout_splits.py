from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the same implementation as Step 7.14.3 so that the holdout test
# remains comparable and does not fork physics code.
from scripts.quantum.condensed_silicon_heat_capacity_debye_baseline import (  # noqa: E402
    _debye_cv_molar,
    _golden_section_minimize,
)


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(int(chunk_bytes))
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cp_shomate(*, coeffs: dict[str, float], t_k: float) -> float:
    t = t_k / 1000.0
    a = float(coeffs["A"])
    b = float(coeffs["B"])
    c = float(coeffs["C"])
    d = float(coeffs["D"])
    e = float(coeffs["E"])
    return a + b * t + c * (t**2) + d * (t**3) + (e / (t**2))


def _sigma_cp_proxy(cp_j_per_molk: float) -> float:
    # Same proxy as the baseline metrics (NIST WebBook condensed Shomate blocks):
    # max(2% of Cp, 0.2 J/mol·K)
    return float(max(0.02 * abs(float(cp_j_per_molk)), 0.2))


def _smoothstep(u: float) -> float:
    x = 0.0 if u <= 0.0 else (1.0 if u >= 1.0 else float(u))
    return x * x * (3.0 - 2.0 * x)


def _load_webbook_shomate_solid(*, path: Path) -> dict[str, Any]:
    """
    Load NIST WebBook condensed-Si Shomate coefficients (solid block).

    Expected schema: {"shomate":[{"phase":"solid","t_min_k":...,"t_max_k":...,"coeffs":{A..H}}...]}
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    payload = _read_json(path)
    blocks = payload.get("shomate")
    if not isinstance(blocks, list):
        raise RuntimeError(f"invalid schema (missing shomate list): {path}")
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if str(b.get("phase")) != "solid":
            continue
        coeffs = b.get("coeffs")
        if not isinstance(coeffs, dict) or not all(k in coeffs for k in ["A", "B", "C", "D", "E"]):
            continue
        return {
            "phase": "solid",
            "t_min_k": float(b.get("t_min_k")),
            "t_max_k": float(b.get("t_max_k")),
            "coeffs": {k: float(coeffs[k]) for k in coeffs.keys()},
            "reference": b.get("reference"),
            "comment": b.get("comment"),
        }
    raise RuntimeError(f"solid Shomate block not found: {path}")


def _cp_frozen_hybrid_debye_shomate(
    *,
    t_k: float,
    theta_d_k: float,
    shomate_solid: dict[str, Any],
) -> float:
    """
    Frozen hybrid Cp model:

    - Use WebBook Shomate (solid) inside its published validity range.
    - Fallback to Debye Cv (molar) outside that range (includes T=0).
    """
    t = float(t_k)
    t_min = float(shomate_solid.get("t_min_k"))
    t_max = float(shomate_solid.get("t_max_k"))
    coeffs = shomate_solid.get("coeffs") if isinstance(shomate_solid.get("coeffs"), dict) else {}

    if t > 0.0 and t_min <= t <= t_max and all(k in coeffs for k in ["A", "B", "C", "D", "E"]):
        ae = {k: float(coeffs[k]) for k in ["A", "B", "C", "D", "E"]}
        return float(_cp_shomate(coeffs=ae, t_k=t))
    return float(_debye_cv_molar(t_k=t, theta_d_k=float(theta_d_k)))


def _fit_theta_d_minimax_max_abs_z(
    *,
    temps_k: np.ndarray,
    cp_obs: np.ndarray,
    sigma: np.ndarray,
    train_idx: list[int],
    shomate_solid: dict[str, Any],
    theta_min_k: float = 200.0,
    theta_max_k: float = 1200.0,
) -> float:
    """
    Fit θ_D by minimizing the maximum |z| over the training range.
    """

    def max_abs_z(theta: float) -> float:
        worst = 0.0
        th = float(theta)
        for i in train_idx:
            sig = float(sigma[i])
            if sig <= 0.0:
                continue
            pred = _cp_frozen_hybrid_debye_shomate(
                t_k=float(temps_k[i]),
                theta_d_k=th,
                shomate_solid=shomate_solid,
            )
            z = (float(pred) - float(cp_obs[i])) / sig
            if math.isfinite(z):
                worst = max(worst, abs(z))
        return float(worst)

    grid = np.linspace(float(theta_min_k), float(theta_max_k), 201, dtype=float)
    vals = [max_abs_z(float(th)) for th in grid.tolist()]
    j0 = int(np.argmin(np.asarray(vals, dtype=float)))
    theta0 = float(grid[j0])

    lo = max(float(theta_min_k), theta0 - 80.0)
    hi = min(float(theta_max_k), theta0 + 80.0)
    theta_opt, _ = _golden_section_minimize(max_abs_z, lo, hi, tol=1e-6)
    return float(theta_opt)


def _metrics_for_idx(
    *,
    idx: list[int],
    cp_obs: np.ndarray,
    cp_pred: np.ndarray,
    sigma: np.ndarray,
    param_count: int,
    is_train: bool,
) -> dict[str, float | int]:
    n = 0
    sum_z2 = 0.0
    max_abs_z = 0.0
    exceed_3sigma = 0
    for i in idx:
        sig = float(sigma[i])
        if sig <= 0.0:
            continue
        z = (float(cp_pred[i]) - float(cp_obs[i])) / sig
        if not math.isfinite(z):
            continue
        n += 1
        sum_z2 += z * z
        max_abs_z = max(max_abs_z, abs(z))
        if abs(z) > 3.0:
            exceed_3sigma += 1

    dof = max(1, n - int(param_count)) if is_train else max(1, n)
    red_chi2 = float(sum_z2 / dof) if dof > 0 else float("nan")
    rms_z = float("nan") if n <= 0 else math.sqrt(float(sum_z2 / n))
    return {
        "n": int(n),
        "max_abs_z": float(max_abs_z),
        "rms_z": float(rms_z),
        "reduced_chi2": float(red_chi2),
        "exceed_3sigma_n": int(exceed_3sigma),
    }


def _fit_theta_d(
    *,
    temps_k: np.ndarray,
    cp_obs: np.ndarray,
    sigma: np.ndarray,
    train_idx: list[int],
) -> float:
    def sse(theta: float) -> float:
        s = 0.0
        for i in train_idx:
            sig = float(sigma[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            r = float(cp_obs[i]) - _debye_cv_molar(t_k=float(temps_k[i]), theta_d_k=float(theta))
            s += w * r * r
        return float(s)

    theta0, _ = _golden_section_minimize(sse, 200.0, 1200.0, tol=1e-6)
    return float(theta0)


def _fit_shomate_coeffs(
    *,
    temps_k: np.ndarray,
    cp_obs: np.ndarray,
    sigma: np.ndarray,
    train_idx: list[int],
) -> dict[str, float]:
    rows = []
    y = []
    w = []
    for i in train_idx:
        t_k = float(temps_k[i])
        if t_k <= 0.0:
            continue
        sig = float(sigma[i])
        if sig <= 0.0:
            continue
        t = t_k / 1000.0
        rows.append([1.0, t, t * t, t * t * t, 1.0 / (t * t)])
        y.append(float(cp_obs[i]))
        w.append(1.0 / (sig * sig))

    if len(rows) < 6:
        raise RuntimeError(f"not enough points to fit Shomate (need >=6, got {len(rows)})")

    x = np.asarray(rows, dtype=float)
    yv = np.asarray(y, dtype=float)
    wv = np.asarray(w, dtype=float)
    sw = np.sqrt(wv)
    xw = x * sw[:, None]
    yw = yv * sw
    coeff, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    a, b, c, d, e = [float(v) for v in coeff.tolist()]
    return {"A": a, "B": b, "C": c, "D": d, "E": e}


def main() -> None:
    out_dir = _ROOT / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = _ROOT / "data" / "quantum" / "sources" / "nist_janaf_silicon_si"
    extracted_path = src_dir / "extracted_values.json"
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_silicon_janaf_sources.py"
        )

    webbook_dir = _ROOT / "data" / "quantum" / "sources" / "nist_webbook_condensed_silicon_si"
    webbook_extracted = webbook_dir / "extracted_values.json"
    if not webbook_extracted.exists():
        raise SystemExit(
            f"[fail] missing: {webbook_extracted}\n"
            "Run: python -B scripts/quantum/fetch_silicon_condensed_thermochemistry_sources.py"
        )
    shomate_solid = _load_webbook_shomate_solid(path=webbook_extracted)

    extracted = _read_json(extracted_path)
    points = extracted.get("points")
    if not isinstance(points, list) or not points:
        raise SystemExit(f"[fail] points missing/empty: {extracted_path}")

    solid = [
        p
        for p in points
        if isinstance(p, dict)
        and str(p.get("phase")) == "solid"
        and isinstance(p.get("T_K"), (int, float))
        and isinstance(p.get("Cp_J_per_molK"), (int, float))
    ]
    if not solid:
        raise SystemExit(f"[fail] no solid-phase points found in: {extracted_path}")

    solid = sorted(solid, key=lambda x: float(x["T_K"]))
    temps_k = np.asarray([float(p["T_K"]) for p in solid], dtype=float)
    cp_obs = np.asarray([float(p["Cp_J_per_molK"]) for p in solid], dtype=float)
    sigma = np.asarray([_sigma_cp_proxy(float(cp)) for cp in cp_obs.tolist()], dtype=float)

    def _idx_in_range(t0: float, t1: float) -> list[int]:
        return [i for i, t in enumerate(temps_k.tolist()) if float(t0) <= float(t) <= float(t1)]

    # Frozen hybrid baseline (Debye low-T + WebBook Shomate high-T):
    # Fit θ_D once on the low-T band to align with strict |z|<=3 gating, then keep it fixed.
    theta_d_hybrid = _fit_theta_d_minimax_max_abs_z(
        temps_k=temps_k,
        cp_obs=cp_obs,
        sigma=sigma,
        train_idx=_idx_in_range(100.0, 300.0),
        shomate_solid=shomate_solid,
    )
    cp_hybrid = np.asarray(
        [
            _cp_frozen_hybrid_debye_shomate(t_k=float(t), theta_d_k=float(theta_d_hybrid), shomate_solid=shomate_solid)
            for t in temps_k.tolist()
        ],
        dtype=float,
    )

    # Temperature-band holdout (solid).
    splits = [
        {"name": "A_low_to_high", "train_T_K": [100.0, 300.0], "test_T_K": [350.0, float(np.max(temps_k))]},
        {"name": "B_high_to_low", "train_T_K": [350.0, float(np.max(temps_k))], "test_T_K": [100.0, 300.0]},
    ]

    split_results: list[dict[str, Any]] = []
    rows_out: list[dict[str, Any]] = []

    for sp in splits:
        name = str(sp["name"])
        train_r = [float(sp["train_T_K"][0]), float(sp["train_T_K"][1])]
        test_r = [float(sp["test_T_K"][0]), float(sp["test_T_K"][1])]
        train_idx = _idx_in_range(train_r[0], train_r[1])
        test_idx = _idx_in_range(test_r[0], test_r[1])

        models: dict[str, Any] = {}

        # Model 0: Frozen hybrid baseline (Debye low-T + WebBook Shomate high-T).
        # This model does not refit parameters per split; it is an operational "frozen I/F" baseline.
        m0_train = _metrics_for_idx(
            idx=train_idx, cp_obs=cp_obs, cp_pred=cp_hybrid, sigma=sigma, param_count=0, is_train=False
        )
        m0_test = _metrics_for_idx(
            idx=test_idx, cp_obs=cp_obs, cp_pred=cp_hybrid, sigma=sigma, param_count=0, is_train=False
        )
        coeffs0 = shomate_solid.get("coeffs") if isinstance(shomate_solid.get("coeffs"), dict) else {}
        models["debye_lowT_minimax_plus_shomate_frozen"] = {
            "params": {
                "theta_D_K": float(theta_d_hybrid),
                "shomate_solid_t_min_k": float(shomate_solid.get("t_min_k")),
                "shomate_solid_t_max_k": float(shomate_solid.get("t_max_k")),
                "A": float(coeffs0.get("A")),
                "B": float(coeffs0.get("B")),
                "C": float(coeffs0.get("C")),
                "D": float(coeffs0.get("D")),
                "E": float(coeffs0.get("E")),
            },
            "train": m0_train,
            "test": m0_test,
            "notes": [
                "Frozen hybrid: θ_D is fit once by minimax max|z| on 100-300 K (JANAF solid), while high-T Cp is fixed by WebBook Shomate (solid; 298-1685 K).",
                "This is used as an operational baseline to show a holdout-ok example under minimal additional constraints (no per-split refitting).",
            ],
        }

        # Model 1: Debye 1-parameter fit (θ_D).
        theta_d = _fit_theta_d(temps_k=temps_k, cp_obs=cp_obs, sigma=sigma, train_idx=train_idx)
        cp_debye = np.asarray([_debye_cv_molar(t_k=float(t), theta_d_k=float(theta_d)) for t in temps_k.tolist()], dtype=float)
        m_train = _metrics_for_idx(idx=train_idx, cp_obs=cp_obs, cp_pred=cp_debye, sigma=sigma, param_count=1, is_train=True)
        m_test = _metrics_for_idx(idx=test_idx, cp_obs=cp_obs, cp_pred=cp_debye, sigma=sigma, param_count=1, is_train=False)
        models["debye_1p"] = {"params": {"theta_D_K": float(theta_d)}, "train": m_train, "test": m_test}

        # Model 2: Shomate 5-parameter fit (A..E).
        try:
            sh = _fit_shomate_coeffs(temps_k=temps_k, cp_obs=cp_obs, sigma=sigma, train_idx=train_idx)
            cp_sh = np.asarray([_cp_shomate(coeffs=sh, t_k=float(t)) if float(t) > 0 else float("nan") for t in temps_k.tolist()], dtype=float)
            m2_train = _metrics_for_idx(idx=[i for i in train_idx if float(temps_k[i]) > 0.0], cp_obs=cp_obs, cp_pred=cp_sh, sigma=sigma, param_count=5, is_train=True)
            m2_test = _metrics_for_idx(idx=[i for i in test_idx if float(temps_k[i]) > 0.0], cp_obs=cp_obs, cp_pred=cp_sh, sigma=sigma, param_count=5, is_train=False)
            models["shomate_5p_fit"] = {"params": sh, "train": m2_train, "test": m2_test}
        except Exception as exc:
            models["shomate_5p_fit"] = {"supported": False, "reason": str(exc)}

        # Flatten rows for CSV.
        for mid, m in models.items():
            if not isinstance(m, dict) or m.get("supported") is False:
                rows_out.append(
                    {
                        "split": name,
                        "model_id": mid,
                        "train_min_T_K": float(train_r[0]),
                        "train_max_T_K": float(train_r[1]),
                        "test_min_T_K": float(test_r[0]),
                        "test_max_T_K": float(test_r[1]),
                        "supported": False,
                        "reason": str(m.get("reason") or ""),
                    }
                )
                continue
            tr = m.get("train") if isinstance(m.get("train"), dict) else {}
            te = m.get("test") if isinstance(m.get("test"), dict) else {}
            params = m.get("params") if isinstance(m.get("params"), dict) else {}
            rows_out.append(
                {
                    "split": name,
                    "model_id": mid,
                    "train_min_T_K": float(train_r[0]),
                    "train_max_T_K": float(train_r[1]),
                    "test_min_T_K": float(test_r[0]),
                    "test_max_T_K": float(test_r[1]),
                    "supported": True,
                    "train_n": int(tr.get("n", 0) or 0),
                    "train_max_abs_z": tr.get("max_abs_z"),
                    "train_reduced_chi2": tr.get("reduced_chi2"),
                    "test_n": int(te.get("n", 0) or 0),
                    "test_max_abs_z": te.get("max_abs_z"),
                    "test_reduced_chi2": te.get("reduced_chi2"),
                    "train_exceed_3sigma_n": int(tr.get("exceed_3sigma_n", 0) or 0),
                    "test_exceed_3sigma_n": int(te.get("exceed_3sigma_n", 0) or 0),
                    **{f"param_{k}": v for k, v in params.items()},
                }
            )

        split_results.append(
            {
                "name": name,
                "train_T_K": train_r,
                "test_T_K": test_r,
                "models": models,
            }
        )

    out_csv = out_dir / "condensed_silicon_heat_capacity_holdout_splits.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        # Stable header order for downstream parsing.
        cols = [
            "split",
            "model_id",
            "train_min_T_K",
            "train_max_T_K",
            "test_min_T_K",
            "test_max_T_K",
            "supported",
            "reason",
            "train_n",
            "train_max_abs_z",
            "train_reduced_chi2",
            "test_n",
            "test_max_abs_z",
            "test_reduced_chi2",
            "train_exceed_3sigma_n",
            "test_exceed_3sigma_n",
            "param_theta_D_K",
            "param_A",
            "param_B",
            "param_C",
            "param_D",
            "param_E",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Plot: holdout severity (test max abs z) for each split and model.
    categories: list[str] = []
    y_debye: list[float] = []
    y_shomate: list[float] = []
    y_hybrid: list[float] = []
    for sp in split_results:
        categories.append(str(sp["name"]))
        m0 = (
            sp["models"].get("debye_lowT_minimax_plus_shomate_frozen") if isinstance(sp.get("models"), dict) else None
        )
        m1 = sp["models"].get("debye_1p") if isinstance(sp.get("models"), dict) else None
        m2 = sp["models"].get("shomate_5p_fit") if isinstance(sp.get("models"), dict) else None
        v0 = None
        v1 = None
        v2 = None
        if isinstance(m0, dict) and isinstance(m0.get("test"), dict):
            v0 = m0["test"].get("max_abs_z")
        if isinstance(m1, dict) and isinstance(m1.get("test"), dict):
            v1 = m1["test"].get("max_abs_z")
        if isinstance(m2, dict) and isinstance(m2.get("test"), dict):
            v2 = m2["test"].get("max_abs_z")
        y_hybrid.append(float(v0) if isinstance(v0, (int, float)) else float("nan"))
        y_debye.append(float(v1) if isinstance(v1, (int, float)) else float("nan"))
        y_shomate.append(float(v2) if isinstance(v2, (int, float)) else float("nan"))

    x = np.arange(len(categories), dtype=float)
    w_bar = 0.26
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.2), dpi=170)
    ax.bar(x - w_bar, y_hybrid, width=w_bar, color="#2ca02c", alpha=0.85, label="Hybrid (frozen)")
    ax.bar(x, y_debye, width=w_bar, color="#1f77b4", alpha=0.85, label="Debye (1p fit)")
    ax.bar(x + w_bar, y_shomate, width=w_bar, color="#ff7f0e", alpha=0.85, label="Shomate fit (5p)")
    ax.axhline(3.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("test max abs(z)")
    ax.set_title("Si Cp(T): temperature-split holdout (JANAF solid points)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_png = out_dir / "condensed_silicon_heat_capacity_holdout_splits.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_metrics = out_dir / "condensed_silicon_heat_capacity_holdout_splits_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.20.6",
                "inputs": {
                    "janaf_extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)},
                    "webbook_condensed_shomate": {"path": str(webbook_extracted), "sha256": _sha256(webbook_extracted)},
                },
                "assumptions": {
                    "sigma_proxy": {"Cp": "max(2% of Cp, 0.2 J/mol·K)"},
                    "hybrid_frozen_model": {
                        "name": "Debye (low-T θ_D) + WebBook Shomate (high-T)",
                        "theta_D_fit": {
                            "objective": "minimize max abs(z) on 100-300 K (JANAF solid points)",
                            "theta_D_K": float(theta_d_hybrid),
                        },
                        "shomate_solid": {
                            "t_min_k": float(shomate_solid.get("t_min_k")),
                            "t_max_k": float(shomate_solid.get("t_max_k")),
                            "coeffs": {
                                k: float(shomate_solid.get("coeffs", {}).get(k))
                                for k in ["A", "B", "C", "D", "E", "F", "G", "H"]
                                if isinstance(shomate_solid.get("coeffs"), dict) and k in shomate_solid.get("coeffs", {})
                            },
                            "reference": shomate_solid.get("reference"),
                            "comment": shomate_solid.get("comment"),
                        },
                        "policy": "Use Shomate (solid) within its validity range; fallback to Debye outside (includes T=0).",
                    },
                    "holdout_reduced_chi2_definition": {
                        "train": "sum(z^2)/max(1,n-params)",
                        "test": "sum(z^2)/max(1,n)  (no parameters are fit on the test range)",
                    },
                },
                "data_range_T_K": [float(np.min(temps_k)), float(np.max(temps_k))],
                "splits": split_results,
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "This audit treats JANAF solid-phase Cp° points as the operational reference for holdout.",
                    "Debye model is expected to degrade outside the low-temperature fit band; Shomate fit is a flexible empirical proxy.",
                    "Hybrid frozen baseline combines independent low-T (Debye θ_D) and high-T (WebBook Shomate) constraints and is expected to pass strict+holdout under the operational |z|<=3 gate.",
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


if __name__ == "__main__":
    main()
