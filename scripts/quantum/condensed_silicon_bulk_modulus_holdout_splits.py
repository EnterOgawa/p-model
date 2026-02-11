from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_ROOT = _repo_root()


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


def _load_ioffe_elastic_constants(root: Path) -> dict[str, Any]:
    src = root / "data" / "quantum" / "sources" / "ioffe_silicon_mechanical_properties" / "extracted_values.json"
    if not src.exists():
        raise SystemExit(
            f"[fail] missing: {src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_elastic_constants_sources.py"
        )
    obj = _read_json(src)
    return {"path": src, "sha256": _sha256(src), "data": obj}


def _bulk_modulus_GPa_from_1e11_dyn_cm2(x: float) -> float:
    # 1 dyn/cm^2 = 0.1 Pa => 1e11 dyn/cm^2 = 1e10 Pa = 10 GPa.
    return 10.0 * float(x)


def _cij_linear(*, t_k: float, intercept: float, slope: float, t_min: float, t_max: float) -> float:
    t = float(t_k)
    if t < float(t_min):
        t = float(t_min)
    if t > float(t_max):
        t = float(t_max)
    return float(intercept) + float(slope) * t


def _sigma_b_proxy_gpa(b_gpa: float) -> float:
    # Ioffe page does not provide a strict uncertainty; use a conservative operational proxy.
    return float(max(0.005 * float(b_gpa), 0.5))


def _metrics_for_idx(
    *,
    idx: np.ndarray,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    param_count: int,
    is_train: bool,
) -> dict[str, float | int]:
    ii = np.asarray(idx, dtype=np.int64).reshape(-1)
    if ii.size == 0:
        return {"n": 0, "max_abs_z": float("nan"), "rms_z": float("nan"), "reduced_chi2": float("nan"), "exceed_3sigma_n": 0}

    z = (y_pred[ii] - y_obs[ii]) / np.maximum(1e-30, sigma[ii])
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"n": 0, "max_abs_z": float("nan"), "rms_z": float("nan"), "reduced_chi2": float("nan"), "exceed_3sigma_n": 0}

    n = int(z.size)
    sum_z2 = float(np.sum(z * z))
    max_abs_z = float(np.max(np.abs(z)))
    rms_z = float(math.sqrt(sum_z2 / max(1, n)))
    exceed = int(np.sum(np.abs(z) > 3.0))
    dof = max(1, n - int(param_count)) if is_train else max(1, n)
    red_chi2 = float(sum_z2 / dof)
    return {
        "n": n,
        "max_abs_z": max_abs_z,
        "rms_z": rms_z,
        "reduced_chi2": red_chi2,
        "exceed_3sigma_n": exceed,
    }


def _fit_const(*, y: np.ndarray, sigma: np.ndarray, idx: np.ndarray) -> float:
    ii = np.asarray(idx, dtype=np.int64).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)[ii]
    ss = np.asarray(sigma, dtype=float).reshape(-1)[ii]
    w = 1.0 / np.maximum(1e-30, ss * ss)
    return float(np.sum(w * yy) / np.sum(w))


def _fit_linear(*, x: np.ndarray, y: np.ndarray, sigma: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    ii = np.asarray(idx, dtype=np.int64).reshape(-1)
    xx = np.asarray(x, dtype=float).reshape(-1)[ii]
    yy = np.asarray(y, dtype=float).reshape(-1)[ii]
    ss = np.asarray(sigma, dtype=float).reshape(-1)[ii]
    w = 1.0 / np.maximum(1e-30, ss * ss)

    # Weighted least squares for y = a + b x.
    X = np.column_stack([np.ones_like(xx), xx]).astype(float)
    WX = X * w.reshape(-1, 1)
    A = X.T @ WX
    b = X.T @ (w * yy)
    try:
        beta = np.linalg.solve(A, b)
    except Exception:
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    a = float(beta[0])
    slope = float(beta[1])
    return a, slope


def _idx_in_range(temps_k: np.ndarray, *, t0: float, t1: float) -> np.ndarray:
    t = np.asarray(temps_k, dtype=float).reshape(-1)
    return np.where((t >= float(t0)) & (t <= float(t1)))[0].astype(np.int64)


def _ref_curve_from_ioffe(*, t_k: np.ndarray, obj: dict[str, Any]) -> np.ndarray:
    vals = obj.get("values", {})
    if not isinstance(vals, dict):
        raise SystemExit("[fail] invalid extracted_values.json: missing 'values' dict")

    b_ref_1e11 = float(vals.get("bulk_modulus_from_C11_C12_1e11_dyn_cm2"))

    temp_dep = obj.get("temperature_dependence_linear", {})
    if not isinstance(temp_dep, dict):
        raise SystemExit("[fail] invalid extracted_values.json: missing 'temperature_dependence_linear' dict")

    tr = temp_dep.get("T_range_K", {})
    if not isinstance(tr, dict):
        raise SystemExit("[fail] invalid extracted_values.json: missing 'T_range_K' dict")
    t_lin_min = float(tr.get("min"))
    t_lin_max = float(tr.get("max"))

    c11 = temp_dep.get("C11", {})
    c12 = temp_dep.get("C12", {})
    if not isinstance(c11, dict) or not isinstance(c12, dict):
        raise SystemExit("[fail] invalid extracted_values.json: missing C11/C12 linear dicts")

    c11_a = float(c11.get("intercept_1e11_dyn_cm2"))
    c11_b = float(c11.get("slope_1e11_dyn_cm2_per_K"))
    c12_a = float(c12.get("intercept_1e11_dyn_cm2"))
    c12_b = float(c12.get("slope_1e11_dyn_cm2_per_K"))

    def b_lin_1e11(tk: float) -> float:
        c11_t = _cij_linear(t_k=float(tk), intercept=c11_a, slope=c11_b, t_min=t_lin_min, t_max=t_lin_max)
        c12_t = _cij_linear(t_k=float(tk), intercept=c12_a, slope=c12_b, t_min=t_lin_min, t_max=t_lin_max)
        return float((c11_t + 2.0 * c12_t) / 3.0)

    # Keep the same "switch" policy as the baseline: below t_lin_min hold B constant at B_ref.
    t_switch = float(t_lin_min)
    b_offset_1e11 = float(b_ref_1e11 - float(b_lin_1e11(t_switch)))

    out: list[float] = []
    for t in np.asarray(t_k, dtype=float).reshape(-1).tolist():
        if float(t) < float(t_switch):
            b_1e11 = float(b_ref_1e11)
        else:
            b_1e11 = float(b_lin_1e11(float(t))) + float(b_offset_1e11)
        out.append(_bulk_modulus_GPa_from_1e11_dyn_cm2(float(b_1e11)))
    return np.asarray(out, dtype=float)


def main() -> None:
    out_dir = _ROOT / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    ioffe = _load_ioffe_elastic_constants(_ROOT)
    obj = ioffe["data"]

    temps_k = np.arange(0.0, 600.0 + 1.0, 1.0, dtype=float)
    b_ref_gpa = _ref_curve_from_ioffe(t_k=temps_k, obj=obj)
    sigma_gpa = np.asarray([_sigma_b_proxy_gpa(float(b)) for b in b_ref_gpa.tolist()], dtype=float)

    splits = [
        {"name": "A_low_to_high", "train_T_K": [0.0, 300.0], "test_T_K": [400.0, 600.0]},
        {"name": "B_high_to_low", "train_T_K": [400.0, 600.0], "test_T_K": [0.0, 300.0]},
    ]

    split_results: list[dict[str, Any]] = []
    rows_out: list[dict[str, Any]] = []

    for sp in splits:
        name = str(sp["name"])
        train_r = [float(sp["train_T_K"][0]), float(sp["train_T_K"][1])]
        test_r = [float(sp["test_T_K"][0]), float(sp["test_T_K"][1])]
        train_idx = _idx_in_range(temps_k, t0=train_r[0], t1=train_r[1])
        test_idx = _idx_in_range(temps_k, t0=test_r[0], t1=test_r[1])

        models: dict[str, Any] = {}

        # Model 0: Frozen reference curve (operational I/F baseline; no per-split refit).
        m0_train = _metrics_for_idx(idx=train_idx, y_obs=b_ref_gpa, y_pred=b_ref_gpa, sigma=sigma_gpa, param_count=0, is_train=False)
        m0_test = _metrics_for_idx(idx=test_idx, y_obs=b_ref_gpa, y_pred=b_ref_gpa, sigma=sigma_gpa, param_count=0, is_train=False)
        models["ioffe_piecewise_frozen"] = {
            "params": {},
            "train": m0_train,
            "test": m0_test,
            "notes": ["Operational baseline: use the Ioffe-derived B(T) curve as a frozen reference (no refit per split)."],
        }

        # Model 1: Constant B fit on the train band.
        b0 = _fit_const(y=b_ref_gpa, sigma=sigma_gpa, idx=train_idx)
        b_const = np.full_like(b_ref_gpa, float(b0), dtype=float)
        m1_train = _metrics_for_idx(idx=train_idx, y_obs=b_ref_gpa, y_pred=b_const, sigma=sigma_gpa, param_count=1, is_train=True)
        m1_test = _metrics_for_idx(idx=test_idx, y_obs=b_ref_gpa, y_pred=b_const, sigma=sigma_gpa, param_count=1, is_train=False)
        models["const_1p_fit"] = {"params": {"B0_GPa": float(b0)}, "train": m1_train, "test": m1_test}

        # Model 2: Linear B(T)=a+bT fit on the train band.
        a0, b0s = _fit_linear(x=temps_k, y=b_ref_gpa, sigma=sigma_gpa, idx=train_idx)
        b_lin = a0 + b0s * temps_k
        m2_train = _metrics_for_idx(idx=train_idx, y_obs=b_ref_gpa, y_pred=b_lin, sigma=sigma_gpa, param_count=2, is_train=True)
        m2_test = _metrics_for_idx(idx=test_idx, y_obs=b_ref_gpa, y_pred=b_lin, sigma=sigma_gpa, param_count=2, is_train=False)
        models["linear_2p_fit"] = {"params": {"a_GPa": float(a0), "b_GPa_per_K": float(b0s)}, "train": m2_train, "test": m2_test}

        for mid, m in models.items():
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
                    "reason": "",
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

        split_results.append({"name": name, "train_T_K": train_r, "test_T_K": test_r, "models": models})

    out_csv = out_dir / "condensed_silicon_bulk_modulus_holdout_splits.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
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
            "param_B0_GPa",
            "param_a_GPa",
            "param_b_GPa_per_K",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Plot: holdout severity (test max abs z) for each split and model.
    categories: list[str] = []
    y_frozen: list[float] = []
    y_const: list[float] = []
    y_linear: list[float] = []
    for sp in split_results:
        categories.append(str(sp["name"]))
        mm = sp.get("models") if isinstance(sp.get("models"), dict) else {}
        m0 = mm.get("ioffe_piecewise_frozen") if isinstance(mm, dict) else None
        m1 = mm.get("const_1p_fit") if isinstance(mm, dict) else None
        m2 = mm.get("linear_2p_fit") if isinstance(mm, dict) else None
        v0 = (m0.get("test") or {}).get("max_abs_z") if isinstance(m0, dict) else None
        v1 = (m1.get("test") or {}).get("max_abs_z") if isinstance(m1, dict) else None
        v2 = (m2.get("test") or {}).get("max_abs_z") if isinstance(m2, dict) else None
        y_frozen.append(float(v0) if isinstance(v0, (int, float)) else float("nan"))
        y_const.append(float(v1) if isinstance(v1, (int, float)) else float("nan"))
        y_linear.append(float(v2) if isinstance(v2, (int, float)) else float("nan"))

    x = np.arange(len(categories), dtype=float)
    w_bar = 0.26
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.2), dpi=170)
    ax.bar(x - w_bar, y_frozen, width=w_bar, color="#2ca02c", alpha=0.85, label="Ioffe reference (frozen)")
    ax.bar(x, y_const, width=w_bar, color="#1f77b4", alpha=0.85, label="Const fit (1p)")
    ax.bar(x + w_bar, y_linear, width=w_bar, color="#ff7f0e", alpha=0.85, label="Linear fit (2p)")
    ax.axhline(3.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("test max abs(z)")
    ax.set_title("Si bulk modulus B(T): temperature-split holdout (Ioffe elastic constants reference)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_png = out_dir / "condensed_silicon_bulk_modulus_holdout_splits.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_metrics = out_dir / "condensed_silicon_bulk_modulus_holdout_splits_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.20.6",
                "inputs": {"ioffe_extracted_values": {"path": str(ioffe["path"]), "sha256": str(ioffe["sha256"])}},
                "assumptions": {
                    "sigma_proxy": {"B_GPa": "max(0.5% of B, 0.5 GPa)"},
                    "holdout_reduced_chi2_definition": {
                        "train": "sum(z^2)/max(1,n-params)",
                        "test": "sum(z^2)/max(1,n)  (no parameters are fit on the test range)",
                    },
                },
                "data_range_T_K": [float(np.min(temps_k)), float(np.max(temps_k))],
                "splits": split_results,
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "This audit treats the Ioffe elastic-constants-derived B(T) as an operational reference curve.",
                    "Const/linear fits are used only to quantify temperature-band sensitivity under a simple model class; this is not a first-principles B(T) prediction claim.",
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

