from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "output" / "public" / "quantum" / "bell"


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_stable_seed` の入出力契約と処理意図を定義する。

def _stable_seed(*parts: str) -> int:
    payload = "::".join([str(p) for p in parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


# 関数: `_shuffle_setting_bits_codes` の入出力契約と処理意図を定義する。

def _shuffle_setting_bits_codes(c: np.ndarray, *, encoding: str, rng: np.random.Generator) -> np.ndarray:
    """
    For time-tag datasets where codes encode both {setting,outcome}, shuffle *only* the setting labels
    while keeping the detector/outcome bit intact.
    """
    cc = np.asarray(c, dtype=np.uint16).reshape(-1)
    # 条件分岐: `cc.size == 0` を満たす経路を評価する。
    if cc.size == 0:
        return cc.copy()

    # 条件分岐: `encoding == "bit0-setting"` を満たす経路を評価する。

    if encoding == "bit0-setting":
        # bit0: setting, bit1: detector (outcome)
        setting = (cc & 1).astype(np.uint16, copy=False)
        perm = rng.permutation(int(setting.size))
        setting2 = setting[perm]
        return (cc & np.uint16(2)) | setting2

    # 条件分岐: `encoding == "bit0-detector"` を満たす経路を評価する。

    if encoding == "bit0-detector":
        # bit0: detector (outcome), bit1: setting
        detector = (cc & 1).astype(np.uint16, copy=False)
        setting = ((cc >> 1) & 1).astype(np.uint16, copy=False)
        perm = rng.permutation(int(setting.size))
        setting2 = setting[perm]
        return detector | (setting2 << np.uint16(1))

    raise ValueError(f"unknown encoding: {encoding}")


# 関数: `_circular_time_shift_sorted_pair` の入出力契約と処理意図を定義する。

def _circular_time_shift_sorted_pair(
    t_s: np.ndarray, x: np.ndarray, *, shift_s: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Circular time-shift t_s by shift_s (wrap inside [t0,t1]) and re-sort together with x.
    Intended for time-tag null tests (break true pairing while keeping marginals).
    """
    t0 = np.asarray(t_s, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x).reshape(-1)
    # 条件分岐: `t0.size != x0.size` を満たす経路を評価する。
    if t0.size != x0.size:
        raise ValueError("t_s and x must have same length")

    # 条件分岐: `t0.size == 0` を満たす経路を評価する。

    if t0.size == 0:
        return t0.copy(), x0.copy(), {"supported": True, "shift_s": float(shift_s), "span_s": 0.0}

    t_min = float(np.min(t0))
    t_max = float(np.max(t0))
    span = float(t_max - t_min)
    # 条件分岐: `not math.isfinite(span) or span <= 0.0` を満たす経路を評価する。
    if not math.isfinite(span) or span <= 0.0:
        return t0.copy(), x0.copy(), {"supported": False, "reason": "invalid time span"}

    u = (t0 - t_min + float(shift_s)) % span
    t2 = u + t_min
    order = np.argsort(t2, kind="mergesort")
    return (
        t2[order].astype(np.float64, copy=False),
        x0[order].copy(),
        {"supported": True, "shift_s": float(shift_s), "span_s": float(span), "wrap": True},
    )


# 関数: `_snap_to_grid_ge` の入出力契約と処理意図を定義する。

def _snap_to_grid_ge(*, x: float | None, grid: Iterable[float]) -> float | None:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return None

    try:
        x0 = float(x)
    except Exception:
        return None

    # 条件分岐: `not math.isfinite(x0)` を満たす経路を評価する。

    if not math.isfinite(x0):
        return None

    vals = sorted({float(v) for v in _finite(grid) if math.isfinite(float(v))})
    # 条件分岐: `not vals` を満たす経路を評価する。
    if not vals:
        return None

    for v in vals:
        # 条件分岐: `v >= x0` を満たす経路を評価する。
        if v >= x0:
            return float(v)

    return float(vals[-1])


# 関数: `_safe_div_array` の入出力契約と処理意図を定義する。

def _safe_div_array(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num0 = np.asarray(num, dtype=float)
    den0 = np.asarray(den, dtype=float)
    out = np.full_like(num0, float("nan"), dtype=float)
    np.divide(num0, den0, out=out, where=den0 != 0.0)
    return out


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_read_json_or_none` の入出力契約と処理意図を定義する。

def _read_json_or_none(path: Path) -> Any | None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    try:
        return _read_json(path)
    except Exception:
        return None


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(int(chunk_bytes)), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_relpath_from_root` の入出力契約と処理意図を定義する。

def _relpath_from_root(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except Exception:
        return str(path)

    return rel.as_posix()


# 関数: `_load_script_module` の入出力契約と処理意図を定義する。

def _load_script_module(*, rel_path: str, name: str) -> Any:
    path = ROOT / rel_path
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    # 条件分岐: `spec is None or spec.loader is None` を満たす経路を評価する。
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 関数: `_ks_distance` の入出力契約と処理意図を定義する。

def _ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    # 条件分岐: `x.size == 0 or y.size == 0` を満たす経路を評価する。
    if x.size == 0 or y.size == 0:
        return float("nan")

    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.sort(np.asarray(y, dtype=float))
    n = xs.size
    m = ys.size
    i = 0
    j = 0
    d = 0.0
    while i < n and j < m:
        # 条件分岐: `xs[i] <= ys[j]` を満たす経路を評価する。
        if xs[i] <= ys[j]:
            i += 1
        else:
            j += 1

        d = max(d, abs(i / n - j / m))

    d = max(d, abs(1.0 - j / m), abs(i / n - 1.0))
    return float(d)


# 関数: `_delay_signature_delta_median` の入出力契約と処理意図を定義する。

def _delay_signature_delta_median(
    x0: np.ndarray | list[float],
    x1: np.ndarray | list[float],
    *,
    epsilon: float = 0.1,
) -> dict[str, Any]:
    """
    Returns a robust delay signature as a delta-median (ns) and an approximate z-score:
      z = |Δmedian| / σ(Δmedian)
    where σ(median) is estimated from mid-quantile width around the median:
      σ(median) ≈ (q(0.5+ε) - q(0.5-ε)) / (4 ε √n)
    """
    eps = float(epsilon)
    # 条件分岐: `not math.isfinite(eps) or eps <= 0.0 or eps >= 0.5` を満たす経路を評価する。
    if not math.isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        eps = 0.1

    # 関数: `_clean` の入出力契約と処理意図を定義する。

    def _clean(a: np.ndarray | list[float]) -> np.ndarray:
        x = np.asarray(a, dtype=float).reshape(-1)
        return x[np.isfinite(x)]

    x0v = _clean(x0)
    x1v = _clean(x1)
    n0 = int(x0v.size)
    n1 = int(x1v.size)

    # 関数: `_median_and_sigma` の入出力契約と処理意図を定義する。
    def _median_and_sigma(x: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
        n = int(x.size)
        # 条件分岐: `n <= 1` を満たす経路を評価する。
        if n <= 1:
            return None, None, None, None

        q_lo, med, q_hi = np.quantile(x, [0.5 - eps, 0.5, 0.5 + eps]).astype(float).tolist()
        width = float(q_hi) - float(q_lo)
        # 条件分岐: `not math.isfinite(width) or width <= 0.0` を満たす経路を評価する。
        if not math.isfinite(width) or width <= 0.0:
            return float(med), None, float(q_lo), float(q_hi)

        sigma = width / (4.0 * eps * math.sqrt(float(n)))
        return float(med), float(sigma), float(q_lo), float(q_hi)

    m0, s0, q0lo, q0hi = _median_and_sigma(x0v)
    m1, s1, q1lo, q1hi = _median_and_sigma(x1v)

    delta = None
    # 条件分岐: `m0 is not None and m1 is not None and math.isfinite(float(m0)) and math.isfin...` を満たす経路を評価する。
    if m0 is not None and m1 is not None and math.isfinite(float(m0)) and math.isfinite(float(m1)):
        delta = float(m0) - float(m1)

    sigma_delta = None
    # 条件分岐: `s0 is not None and s1 is not None and s0 > 0.0 and s1 > 0.0` を満たす経路を評価する。
    if s0 is not None and s1 is not None and s0 > 0.0 and s1 > 0.0:
        sigma_delta = math.sqrt(float(s0) ** 2 + float(s1) ** 2)

    z = None
    # 条件分岐: `delta is not None and sigma_delta is not None and sigma_delta > 0.0` を満たす経路を評価する。
    if delta is not None and sigma_delta is not None and sigma_delta > 0.0:
        z = abs(float(delta)) / float(sigma_delta)

    return {
        "unit": "ns",
        "epsilon": float(eps),
        "n0": int(n0),
        "n1": int(n1),
        "setting0_median_ns": m0,
        "setting1_median_ns": m1,
        "delta_median_0_minus_1_ns": delta,
        "q0_midwidth_ns": {"setting0": [q0lo, q0hi], "setting1": [q1lo, q1hi]},
        "sigma_median_ns": {"setting0": s0, "setting1": s1},
        "sigma_delta_median_ns": sigma_delta,
        "z_delta_median": z,
    }


# 関数: `_finite` の入出力契約と処理意図を定義する。

def _finite(values: Iterable[float | None]) -> list[float]:
    out: list[float] = []
    for v in values:
        # 条件分岐: `v is None` を満たす経路を評価する。
        if v is None:
            continue

        try:
            x = float(v)
        except Exception:
            continue

        # 条件分岐: `math.isfinite(x)` を満たす経路を評価する。

        if math.isfinite(x):
            out.append(x)

    return out


# 関数: `_min_max` の入出力契約と処理意図を定義する。

def _min_max(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    v = _finite(values)
    # 条件分岐: `not v` を満たす経路を評価する。
    if not v:
        return None, None

    return float(min(v)), float(max(v))


# 関数: `_dataset_display_name` の入出力契約と処理意図を定義する。

def _dataset_display_name(dataset_id: str) -> str:
    ds = str(dataset_id or "")
    # 条件分岐: `ds.startswith("weihs1998_")` を満たす経路を評価する。
    if ds.startswith("weihs1998_"):
        return "Weihs 1998"

    # 条件分岐: `ds.startswith("nist_")` を満たす経路を評価する。

    if ds.startswith("nist_"):
        return "NIST"

    # 条件分岐: `ds.startswith("kwiat2013_")` を満たす経路を評価する。

    if ds.startswith("kwiat2013_"):
        return "Kwiat/Christensen 2013"

    # 条件分岐: `ds == "delft_hensen2015"` を満たす経路を評価する。

    if ds == "delft_hensen2015":
        return "Delft (Hensen 2015)"

    # 条件分岐: `ds.startswith("delft_hensen2016_")` を満たす経路を評価する。

    if ds.startswith("delft_hensen2016_"):
        return "Delft (Hensen 2016)"

    return ds


# 関数: `_dataset_year` の入出力契約と処理意図を定義する。

def _dataset_year(dataset_id: str) -> int | None:
    ds = str(dataset_id or "")
    # 条件分岐: `ds.startswith("weihs1998_")` を満たす経路を評価する。
    if ds.startswith("weihs1998_"):
        return 1998

    # 条件分岐: `ds.startswith("kwiat2013_")` を満たす経路を評価する。

    if ds.startswith("kwiat2013_"):
        return 2013

    # 条件分岐: `ds == "delft_hensen2015"` を満たす経路を評価する。

    if ds == "delft_hensen2015":
        return 2015

    # 条件分岐: `ds.startswith("delft_hensen2016_")` を満たす経路を評価する。

    if ds.startswith("delft_hensen2016_"):
        return 2016

    # 条件分岐: `ds.startswith("nist_")` を満たす経路を評価する。

    if ds.startswith("nist_"):
        return 2015

    return None


# 関数: `_dataset_selection_class` の入出力契約と処理意図を定義する。

def _dataset_selection_class(dataset_id: str) -> str:
    ds = str(dataset_id or "")
    # 条件分岐: `ds.startswith("delft_hensen")` を満たす経路を評価する。
    if ds.startswith("delft_hensen"):
        return "event_ready_offset"

    return "time_tag_window_or_offset"


# 関数: `_dataset_statistic_family` の入出力契約と処理意図を定義する。

def _dataset_statistic_family(dataset_id: str) -> str:
    ds = str(dataset_id or "")
    # 条件分岐: `ds.startswith(("nist_", "kwiat2013_"))` を満たす経路を評価する。
    if ds.startswith(("nist_", "kwiat2013_")):
        return "CH"

    return "CHSH"


# 関数: `_linear_trend_metrics` の入出力契約と処理意図を定義する。

def _linear_trend_metrics(points: Iterable[tuple[float | None, float | None]]) -> dict[str, Any]:
    xs_v: list[float] = []
    ys_v: list[float] = []
    for x, y in points:
        try:
            xv = float(x) if x is not None else float("nan")
            yv = float(y) if y is not None else float("nan")
        except Exception:
            continue

        # 条件分岐: `math.isfinite(xv) and math.isfinite(yv)` を満たす経路を評価する。

        if math.isfinite(xv) and math.isfinite(yv):
            xs_v.append(xv)
            ys_v.append(yv)

    # 条件分岐: `len(xs_v) < 2` を満たす経路を評価する。

    if len(xs_v) < 2:
        return {"supported": False, "reason": "need >=2 finite points"}

    xs = np.asarray(xs_v, dtype=float)
    ys = np.asarray(ys_v, dtype=float)
    x_ref = float(np.mean(xs))
    xc = xs - x_ref
    try:
        slope, intercept = np.polyfit(xc, ys, 1).astype(float).tolist()
    except Exception as exc:
        return {"supported": False, "reason": str(exc)}

    y_fit = slope * xc + intercept
    residuals = ys - y_fit
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    r = None
    sx = float(np.std(xs, ddof=1))
    sy = float(np.std(ys, ddof=1))
    # 条件分岐: `sx > 0.0 and sy > 0.0` を満たす経路を評価する。
    if sx > 0.0 and sy > 0.0:
        corr = float(np.corrcoef(xs, ys)[0, 1])
        # 条件分岐: `math.isfinite(corr)` を満たす経路を評価する。
        if math.isfinite(corr):
            r = corr

    year_min = float(np.min(xs))
    year_max = float(np.max(xs))
    span_years = float(year_max - year_min)
    delta_span = float(slope * span_years)
    return {
        "supported": True,
        "n_points": int(xs.size),
        "year_min": year_min,
        "year_max": year_max,
        "year_span": span_years,
        "year_ref": x_ref,
        "slope_per_year": float(slope),
        "intercept_at_year_ref": float(intercept),
        "delta_over_span": delta_span,
        "pearson_r": r,
        "rmse": rmse,
        "median_abs_residual": float(np.median(np.abs(residuals))),
    }


# 関数: `_group_metric_summary` の入出力契約と処理意図を定義する。

def _group_metric_summary(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    value_key: str,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        group = str(row.get(group_key) or "unknown")
        value = row.get(value_key)
        try:
            xv = float(value) if value is not None else float("nan")
        except Exception:
            continue

        # 条件分岐: `not math.isfinite(xv)` を満たす経路を評価する。

        if not math.isfinite(xv):
            continue

        buckets.setdefault(group, []).append(xv)

    out: list[dict[str, Any]] = []
    for group in sorted(buckets.keys()):
        arr = np.asarray(buckets[group], dtype=float)
        out.append(
            {
                "group": group,
                "n": int(arr.size),
                "median": float(np.median(arr)),
                "p16": float(np.quantile(arr, 0.16)),
                "p84": float(np.quantile(arr, 0.84)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )

    return out


# 関数: `_delay_signature_z_max` の入出力契約と処理意図を定義する。

def _delay_signature_z_max(delay_signature: Any) -> float | None:
    # 条件分岐: `not isinstance(delay_signature, dict)` を満たす経路を評価する。
    if not isinstance(delay_signature, dict):
        return None

    zs: list[float] = []
    for who in ("Alice", "Bob"):
        d = delay_signature.get(who)
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        z = d.get("z_delta_median")
        try:
            zv = float(z)
        except Exception:
            continue

        # 条件分岐: `math.isfinite(zv)` を満たす経路を評価する。

        if math.isfinite(zv):
            zs.append(abs(zv))

    return float(max(zs)) if zs else None


# 関数: `_nanstd` の入出力契約と処理意図を定義する。

def _nanstd(values: Iterable[float | None]) -> float:
    v = np.asarray(_finite(values), dtype=float)
    # 条件分岐: `v.size <= 1` を満たす経路を評価する。
    if v.size <= 1:
        return float("nan")

    return float(np.std(v, ddof=1))


# 関数: `_nan_cov` の入出力契約と処理意図を定義する。

def _nan_cov(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    # 条件分岐: `X.ndim != 2` を満たす経路を評価する。
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    n_points = int(X.shape[1])
    cov = np.full((n_points, n_points), float("nan"), dtype=float)
    for i in range(n_points):
        xi = X[:, i]
        for j in range(n_points):
            xj = X[:, j]
            m = np.isfinite(xi) & np.isfinite(xj)
            # 条件分岐: `int(np.sum(m)) <= 1` を満たす経路を評価する。
            if int(np.sum(m)) <= 1:
                continue

            cov[i, j] = float(np.cov(xi[m], xj[m], ddof=1)[0, 1])

    return cov


# 関数: `_matrix_to_json` の入出力契約と処理意図を定義する。

def _matrix_to_json(cov: np.ndarray) -> list[list[float | None]]:
    m = np.asarray(cov, dtype=float)
    out: list[list[float | None]] = []
    for i in range(int(m.shape[0])):
        row: list[float | None] = []
        for j in range(int(m.shape[1])):
            v = float(m[i, j])
            row.append(v if math.isfinite(v) else None)

        out.append(row)

    return out


# 関数: `_matrix_from_json` の入出力契約と処理意図を定義する。

def _matrix_from_json(cov_json: Any) -> np.ndarray:
    # 条件分岐: `not isinstance(cov_json, list)` を満たす経路を評価する。
    if not isinstance(cov_json, list):
        return np.zeros((0, 0), dtype=float)

    rows: list[list[float]] = []
    n_cols = None
    for row in cov_json:
        # 条件分岐: `not isinstance(row, list)` を満たす経路を評価する。
        if not isinstance(row, list):
            continue

        # 条件分岐: `n_cols is None` を満たす経路を評価する。

        if n_cols is None:
            n_cols = len(row)

        # 条件分岐: `n_cols is None or len(row) != n_cols` を満たす経路を評価する。

        if n_cols is None or len(row) != n_cols:
            continue

        vals: list[float] = []
        for value in row:
            try:
                x = float(value)
            except Exception:
                x = float("nan")

            vals.append(x)

        rows.append(vals)

    # 条件分岐: `not rows` を満たす経路を評価する。

    if not rows:
        return np.zeros((0, 0), dtype=float)

    return np.asarray(rows, dtype=float)


# 関数: `_diag_sigma_from_cov` の入出力契約と処理意図を定義する。

def _diag_sigma_from_cov(cov: np.ndarray) -> list[float | None]:
    m = np.asarray(cov, dtype=float)
    # 条件分岐: `m.ndim != 2 or m.shape[0] == 0` を満たす経路を評価する。
    if m.ndim != 2 or m.shape[0] == 0:
        return []

    out: list[float | None] = []
    for v in np.diag(m).tolist():
        try:
            x = float(v)
        except Exception:
            out.append(None)
            continue

        # 条件分岐: `not math.isfinite(x) or x < 0.0` を満たす経路を評価する。

        if not math.isfinite(x) or x < 0.0:
            out.append(None)
            continue

        out.append(float(math.sqrt(x)))

    return out


# 関数: `_cov_eigen_summary` の入出力契約と処理意図を定義する。

def _cov_eigen_summary(cov: np.ndarray) -> dict[str, Any]:
    m = np.asarray(cov, dtype=float)
    # 条件分岐: `m.ndim != 2 or m.shape[0] == 0 or m.shape[0] != m.shape[1]` を満たす経路を評価する。
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[0] != m.shape[1]:
        return {"supported": False, "reason": "covariance matrix is empty or not square"}

    m2 = np.nan_to_num((m + m.T) * 0.5, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        eig_vals, _ = np.linalg.eigh(m2)
    except Exception as exc:
        return {"supported": False, "reason": str(exc)}

    order = np.argsort(eig_vals)[::-1]
    eig_sorted = np.asarray(eig_vals[order], dtype=float)
    pos = eig_sorted[eig_sorted > 0.0]
    total_pos = float(np.sum(pos)) if pos.size else 0.0
    explained = [
        float(v / total_pos) if total_pos > 0.0 and math.isfinite(float(v)) and float(v) > 0.0 else 0.0
        for v in eig_sorted.tolist()
    ]
    abs_nonzero = [abs(float(v)) for v in eig_sorted.tolist() if math.isfinite(float(v)) and abs(float(v)) > 1e-15]
    cond = float(abs(float(eig_sorted[0])) / min(abs_nonzero)) if abs_nonzero else None
    return {
        "supported": True,
        "eigenvalues_desc": [float(v) for v in eig_sorted.tolist()],
        "explained_ratio_positive": explained,
        "rank_eps_1e-10": int(np.linalg.matrix_rank(m2, tol=1e-10)),
        "condition_number_abs": cond,
    }


# 関数: `_corrcoef_rows` の入出力契約と処理意図を定義する。

def _corrcoef_rows(X: np.ndarray) -> np.ndarray:
    a = np.asarray(X, dtype=float)
    # 条件分岐: `a.ndim != 2` を満たす経路を評価する。
    if a.ndim != 2:
        raise ValueError("X must be 2D")

    # 条件分岐: `a.shape[0] == 0` を満たす経路を評価する。

    if a.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)

    n_rows = int(a.shape[0])
    out = np.full((n_rows, n_rows), float("nan"), dtype=float)
    for i in range(n_rows):
        xi = np.asarray(a[i], dtype=float)
        mi = np.isfinite(xi)
        for j in range(n_rows):
            xj = np.asarray(a[j], dtype=float)
            mj = np.isfinite(xj)
            mask = mi & mj
            # 条件分岐: `int(np.sum(mask)) <= 1` を満たす経路を評価する。
            if int(np.sum(mask)) <= 1:
                continue

            x0 = xi[mask]
            x1 = xj[mask]
            s0 = float(np.std(x0, ddof=1))
            s1 = float(np.std(x1, ddof=1))
            # 条件分岐: `s0 <= 0.0 or s1 <= 0.0` を満たす経路を評価する。
            if s0 <= 0.0 or s1 <= 0.0:
                continue

            corr = float(np.cov(x0, x1, ddof=1)[0, 1] / (s0 * s1))
            # 条件分岐: `math.isfinite(corr)` を満たす経路を評価する。
            if math.isfinite(corr):
                out[i, j] = corr

    return out


# 関数: `_bootstrap_corrcoef_rows` の入出力契約と処理意図を定義する。

def _bootstrap_corrcoef_rows(
    X: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(X, dtype=float)
    # 条件分岐: `a.ndim != 2` を満たす経路を評価する。
    if a.ndim != 2:
        raise ValueError("X must be 2D")

    n_rows, n_cols = int(a.shape[0]), int(a.shape[1])
    # 条件分岐: `n_rows == 0 or n_cols <= 1 or n_boot <= 1` を満たす経路を評価する。
    if n_rows == 0 or n_cols <= 1 or n_boot <= 1:
        zero = np.zeros((n_rows, n_rows), dtype=float)
        return zero, zero, zero

    rng = np.random.default_rng(int(seed))
    boot = np.full((int(n_boot), n_rows, n_rows), float("nan"), dtype=float)
    for idx in range(int(n_boot)):
        pick = rng.integers(0, n_cols, size=n_cols, endpoint=False)
        boot[idx] = _corrcoef_rows(a[:, pick])

    mean = np.nanmean(boot, axis=0)
    sigma = np.nanstd(boot, axis=0, ddof=1)
    flat = boot.reshape(int(n_boot), n_rows * n_rows)
    cov_flat = _nan_cov(flat)
    return mean, sigma, cov_flat


# 関数: `_jackknife_corrcoef_rows` の入出力契約と処理意図を定義する。

def _jackknife_corrcoef_rows(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(X, dtype=float)
    # 条件分岐: `a.ndim != 2` を満たす経路を評価する。
    if a.ndim != 2:
        raise ValueError("X must be 2D")

    n_rows, n_cols = int(a.shape[0]), int(a.shape[1])
    # 条件分岐: `n_rows == 0 or n_cols <= 2` を満たす経路を評価する。
    if n_rows == 0 or n_cols <= 2:
        zero = np.zeros((n_rows, n_rows), dtype=float)
        return zero, zero

    jk = np.full((n_cols, n_rows, n_rows), float("nan"), dtype=float)
    for idx in range(n_cols):
        keep = np.ones(n_cols, dtype=bool)
        keep[idx] = False
        jk[idx] = _corrcoef_rows(a[:, keep])

    mean = np.nanmean(jk, axis=0)
    diff = jk - mean[None, :, :]
    var = (float(n_cols - 1) / float(n_cols)) * np.nansum(diff * diff, axis=0)
    sigma = np.sqrt(np.maximum(var, 0.0))
    return mean, sigma


# 関数: `_extract_sweep_profile` の入出力契約と処理意図を定義する。

def _extract_sweep_profile(
    *,
    dataset_id: str,
    cov_obj: dict[str, Any],
    cov_boot_obj: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = []
    for source_name, obj in (("bootstrap", cov_boot_obj), ("analytic", cov_obj)):
        for sweep_key in ("window_sweep", "offset_sweep"):
            sweep = obj.get(sweep_key) if isinstance(obj, dict) else None
            # 条件分岐: `not isinstance(sweep, dict)` を満たす経路を評価する。
            if not isinstance(sweep, dict):
                continue

            # 条件分岐: `not bool(sweep.get("supported"))` を満たす経路を評価する。

            if not bool(sweep.get("supported")):
                continue

            values = sweep.get("values")
            means = sweep.get("mean")
            # 条件分岐: `not isinstance(values, list) or not isinstance(means, list) or len(values) !=...` を満たす経路を評価する。
            if not isinstance(values, list) or not isinstance(means, list) or len(values) != len(means) or len(values) < 3:
                continue

            pts: list[tuple[float, float]] = []
            for raw_x, raw_y in zip(values, means):
                try:
                    x = float(raw_x)
                    y = float(raw_y)
                except Exception:
                    continue

                # 条件分岐: `math.isfinite(x) and math.isfinite(y)` を満たす経路を評価する。

                if math.isfinite(x) and math.isfinite(y):
                    pts.append((x, y))

            # 条件分岐: `len(pts) < 3` を満たす経路を評価する。

            if len(pts) < 3:
                continue

            pts.sort(key=lambda t: t[0])
            xs = np.asarray([p[0] for p in pts], dtype=float)
            ys = np.asarray([p[1] for p in pts], dtype=float)
            x_min = float(np.min(xs))
            x_max = float(np.max(xs))
            # 条件分岐: `x_max > x_min` を満たす経路を評価する。
            if x_max > x_min:
                x_norm = (xs - x_min) / (x_max - x_min)
            else:
                x_norm = np.zeros_like(xs)

            candidates.append(
                {
                    "source": source_name,
                    "sweep_key": sweep_key,
                    "param_name": str(sweep.get("param_name") or ""),
                    "x": xs,
                    "x_norm": x_norm,
                    "y": ys,
                    "n_points": int(xs.size),
                    "x_min": x_min,
                    "x_max": x_max,
                }
            )

    # 条件分岐: `not candidates` を満たす経路を評価する。

    if not candidates:
        return None

    # Prefer bootstrap-window, then bootstrap-offset, then analytic.

    score_map = {("bootstrap", "window_sweep"): 0, ("bootstrap", "offset_sweep"): 1, ("analytic", "window_sweep"): 2}
    best = sorted(candidates, key=lambda c: score_map.get((c["source"], c["sweep_key"]), 3))[0]
    return {
        "dataset_id": str(dataset_id),
        "source": str(best["source"]),
        "sweep_key": str(best["sweep_key"]),
        "param_name": str(best["param_name"]),
        "n_points": int(best["n_points"]),
        "param_min": float(best["x_min"]),
        "param_max": float(best["x_max"]),
        "x_norm": [float(v) for v in np.asarray(best["x_norm"], dtype=float).tolist()],
        "y": [float(v) for v in np.asarray(best["y"], dtype=float).tolist()],
    }


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None

    # 条件分岐: `not math.isfinite(out)` を満たす経路を評価する。

    if not math.isfinite(out):
        return None

    return out


# 関数: `_row_nearest` の入出力契約と処理意図を定義する。

def _row_nearest(*, rows: list[dict[str, Any]], x_key: str, target: float) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        x = _safe_float(row.get(x_key))
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            continue

        d = abs(float(x) - float(target))
        # 条件分岐: `best is None or d < best[0]` を満たす経路を評価する。
        if best is None or d < best[0]:
            best = (d, row)

    return best[1] if best is not None else None


# 関数: `_extract_xy` の入出力契約と処理意図を定義する。

def _extract_xy(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
    x_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pts: list[tuple[float, float]] = []
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        x0 = _safe_float(row.get(x_key))
        # 条件分岐: `x0 is None` を満たす経路を評価する。
        if x0 is None:
            continue

        y0 = None
        for key in y_keys:
            y0 = _safe_float(row.get(key))
            # 条件分岐: `y0 is not None` を満たす経路を評価する。
            if y0 is not None:
                break

        # 条件分岐: `y0 is None` を満たす経路を評価する。

        if y0 is None:
            continue

        pts.append((float(x0) * float(x_scale), float(y0)))

    # 条件分岐: `len(pts) < 2` を満たす経路を評価する。

    if len(pts) < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    pts.sort(key=lambda t: t[0])
    xs = np.asarray([p[0] for p in pts], dtype=float)
    ys = np.asarray([p[1] for p in pts], dtype=float)
    return xs, ys


# 関数: `_local_slope_at` の入出力契約と処理意図を定義する。

def _local_slope_at(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
    target: float,
    x_scale: float = 1.0,
) -> float | None:
    xs, ys = _extract_xy(rows=rows, x_key=x_key, y_keys=y_keys, x_scale=x_scale)
    # 条件分岐: `xs.size < 2` を満たす経路を評価する。
    if xs.size < 2:
        return None

    idx = int(np.argmin(np.abs(xs - float(target))))
    # 条件分岐: `idx <= 0` を満たす経路を評価する。
    if idx <= 0:
        i0, i1 = 0, 1
    # 条件分岐: 前段条件が不成立で、`idx >= int(xs.size) - 1` を追加評価する。
    elif idx >= int(xs.size) - 1:
        i0, i1 = int(xs.size) - 2, int(xs.size) - 1
    else:
        i0, i1 = idx - 1, idx + 1

    dx = float(xs[i1] - xs[i0])
    # 条件分岐: `not math.isfinite(dx) or abs(dx) <= 0.0` を満たす経路を評価する。
    if not math.isfinite(dx) or abs(dx) <= 0.0:
        return None

    dy = float(ys[i1] - ys[i0])
    return float(dy / dx)


# 関数: `_median_step` の入出力契約と処理意図を定義する。

def _median_step(*, values: list[float]) -> float | None:
    # 条件分岐: `len(values) < 2` を満たす経路を評価する。
    if len(values) < 2:
        return None

    arr = np.sort(np.asarray(values, dtype=float))
    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    # 条件分岐: `diffs.size == 0` を満たす経路を評価する。
    if diffs.size == 0:
        return None

    return float(np.median(diffs))


# 関数: `_relative_spread` の入出力契約と処理意図を定義する。

def _relative_spread(values: list[float]) -> float | None:
    arr = np.asarray(_finite(values), dtype=float)
    # 条件分岐: `arr.size <= 1` を満たす経路を評価する。
    if arr.size <= 1:
        return None

    med = float(np.median(arr))
    scale = max(abs(med), 1e-12)
    return float(np.std(arr, ddof=1) / scale)


# 関数: `_flatten_counts_from_row` の入出力契約と処理意図を定義する。

def _flatten_counts_from_row(row: dict[str, Any]) -> list[float]:
    out: list[float] = []
    for key in (
        "n_by_setting",
        "coinc_by_setting_pair",
        "n_by_setting_subtracted",
        "coinc_by_setting_pair_subtracted",
        "coinc_by_setting_pair_subtracted_clipped",
    ):
        value = row.get(key)
        # 条件分岐: `not isinstance(value, list)` を満たす経路を評価する。
        if not isinstance(value, list):
            continue

        for elem in value:
            # 条件分岐: `isinstance(elem, list)` を満たす経路を評価する。
            if isinstance(elem, list):
                out.extend(_finite(elem))
            else:
                x = _safe_float(elem)
                # 条件分岐: `x is not None` を満たす経路を評価する。
                if x is not None:
                    out.append(float(x))

        # 条件分岐: `out` を満たす経路を評価する。

        if out:
            return out

    return out


# 関数: `_max_abs_delay_median_ns` の入出力契約と処理意図を定義する。

def _max_abs_delay_median_ns(delay_signature: Any) -> tuple[float | None, float | None]:
    # 条件分岐: `not isinstance(delay_signature, dict)` を満たす経路を評価する。
    if not isinstance(delay_signature, dict):
        return None, None

    vals: list[float] = []
    for who in ("Alice", "Bob"):
        d = delay_signature.get(who)
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        v = _safe_float(d.get("delta_median_0_minus_1_ns"))
        # 条件分岐: `v is not None` を満たす経路を評価する。
        if v is not None:
            vals.append(abs(float(v)))

    z = _delay_signature_z_max(delay_signature)
    return (float(max(vals)) if vals else None), z


# 関数: `_reshape_2x2_counts` の入出力契約と処理意図を定義する。

def _reshape_2x2_counts(raw: Any) -> np.ndarray | None:
    # 条件分岐: `not isinstance(raw, list) or len(raw) != 2` を満たす経路を評価する。
    if not isinstance(raw, list) or len(raw) != 2:
        return None

    rows: list[list[float]] = []
    for row in raw:
        # 条件分岐: `not isinstance(row, list) or len(row) != 2` を満たす経路を評価する。
        if not isinstance(row, list) or len(row) != 2:
            return None

        vals = []
        for value in row:
            x = _safe_float(value)
            # 条件分岐: `x is None` を満たす経路を評価する。
            if x is None:
                return None

            vals.append(float(x))

        rows.append(vals)

    arr = np.asarray(rows, dtype=float)
    # 条件分岐: `arr.shape != (2, 2)` を満たす経路を評価する。
    if arr.shape != (2, 2):
        return None

    return arr


# 関数: `_setting_balance_from_pair_trials` の入出力契約と処理意図を定義する。

def _setting_balance_from_pair_trials(pair_trials: np.ndarray) -> dict[str, Any] | None:
    arr = np.asarray(pair_trials, dtype=float)
    # 条件分岐: `arr.shape != (2, 2)` を満たす経路を評価する。
    if arr.shape != (2, 2):
        return None

    total = float(np.sum(arr))
    # 条件分岐: `not math.isfinite(total) or total <= 0.0` を満たす経路を評価する。
    if not math.isfinite(total) or total <= 0.0:
        return None

    probs = arr / total
    expected = 0.25
    pair_bias = float(np.max(np.abs(probs - expected)))
    pair_balance = 1.0 - pair_bias / expected if expected > 0.0 else None
    return {
        "pair_trials_total": int(round(total)),
        "pair_trials": [[int(round(v)) for v in row] for row in arr.tolist()],
        "pair_probabilities": [[float(v) for v in row] for row in probs.tolist()],
        "pair_max_abs_bias_from_uniform": pair_bias,
        "pair_balance_score": float(pair_balance) if pair_balance is not None else None,
    }


# 関数: `_setting_balance_from_marginals` の入出力契約と処理意図を定義する。

def _setting_balance_from_marginals(
    *,
    alice_counts: list[int] | tuple[int, int],
    bob_counts: list[int] | tuple[int, int],
) -> dict[str, Any] | None:
    # 条件分岐: `len(alice_counts) != 2 or len(bob_counts) != 2` を満たす経路を評価する。
    if len(alice_counts) != 2 or len(bob_counts) != 2:
        return None

    a0 = _safe_float(alice_counts[0])
    a1 = _safe_float(alice_counts[1])
    b0 = _safe_float(bob_counts[0])
    b1 = _safe_float(bob_counts[1])
    # 条件分岐: `None in (a0, a1, b0, b1)` を満たす経路を評価する。
    if None in (a0, a1, b0, b1):
        return None

    a0 = float(a0)
    a1 = float(a1)
    b0 = float(b0)
    b1 = float(b1)
    a_tot = a0 + a1
    b_tot = b0 + b1
    # 条件分岐: `a_tot <= 0.0 or b_tot <= 0.0` を満たす経路を評価する。
    if a_tot <= 0.0 or b_tot <= 0.0:
        return None

    p_a1 = a1 / a_tot
    p_b1 = b1 / b_tot
    a_bias = abs(p_a1 - 0.5)
    b_bias = abs(p_b1 - 0.5)
    max_bias = max(a_bias, b_bias)
    balance = 1.0 - max_bias / 0.5
    return {
        "alice_setting_counts": [int(round(a0)), int(round(a1))],
        "bob_setting_counts": [int(round(b0)), int(round(b1))],
        "alice_p1": float(p_a1),
        "bob_p1": float(p_b1),
        "alice_abs_bias_from_half": float(a_bias),
        "bob_abs_bias_from_half": float(b_bias),
        "max_abs_bias": float(max_bias),
        "marginal_balance_score": float(balance),
    }


# 関数: `_load_detection_efficiency_proxy` の入出力契約と処理意図を定義する。

def _load_detection_efficiency_proxy(*, ds_dir: Path) -> dict[str, Any]:
    trial_path = ds_dir / "trial_based_counts.json"
    # 条件分岐: `not trial_path.exists()` を満たす経路を評価する。
    if not trial_path.exists():
        return {
            "supported": False,
            "reason": "trial_based_counts.json not found (no direct denominator for detection efficiency)",
        }

    try:
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"supported": False, "reason": str(exc)}

    counts = trial.get("counts") if isinstance(trial.get("counts"), dict) else {}
    pair_trials = _reshape_2x2_counts(counts.get("trials_by_setting_pair"))
    pair_coinc = _reshape_2x2_counts(counts.get("coinc_by_setting_pair"))
    # 条件分岐: `pair_trials is None` を満たす経路を評価する。
    if pair_trials is None:
        return {"supported": False, "reason": "trials_by_setting_pair missing or invalid"}

    n_trials_total = float(np.sum(pair_trials))
    # 条件分岐: `not math.isfinite(n_trials_total) or n_trials_total <= 0.0` を満たす経路を評価する。
    if not math.isfinite(n_trials_total) or n_trials_total <= 0.0:
        return {"supported": False, "reason": "non-positive trial total"}

    n_coinc_total = float(np.sum(pair_coinc)) if pair_coinc is not None else float("nan")

    a_trials = _finite((counts.get("alice_trials_by_setting") or []))
    b_trials = _finite((counts.get("bob_trials_by_setting") or []))
    a_clicks = _finite((counts.get("alice_clicks_by_setting") or []))
    b_clicks = _finite((counts.get("bob_clicks_by_setting") or []))
    # 条件分岐: `len(a_trials) != 2 or len(b_trials) != 2 or len(a_clicks) != 2 or len(b_click...` を満たす経路を評価する。
    if len(a_trials) != 2 or len(b_trials) != 2 or len(a_clicks) != 2 or len(b_clicks) != 2:
        return {"supported": False, "reason": "alice/bob trial-click marginals missing or invalid"}

    a_trials_tot = float(sum(a_trials))
    b_trials_tot = float(sum(b_trials))
    a_clicks_tot = float(sum(a_clicks))
    b_clicks_tot = float(sum(b_clicks))
    # 条件分岐: `a_trials_tot <= 0.0 or b_trials_tot <= 0.0` を満たす経路を評価する。
    if a_trials_tot <= 0.0 or b_trials_tot <= 0.0:
        return {"supported": False, "reason": "invalid alice/bob trial totals"}

    eta_a = a_clicks_tot / a_trials_tot
    eta_b = b_clicks_tot / b_trials_tot
    eta_min = min(eta_a, eta_b)
    eta_pair = (n_coinc_total / n_trials_total) if (math.isfinite(n_coinc_total) and n_trials_total > 0.0) else None
    return {
        "supported": True,
        "method": "trial_based_counts",
        "trials_total": int(round(n_trials_total)),
        "coinc_total": int(round(n_coinc_total)) if math.isfinite(n_coinc_total) else None,
        "eta_alice": float(eta_a),
        "eta_bob": float(eta_b),
        "eta_min": float(eta_min),
        "eta_pair": float(eta_pair) if eta_pair is not None else None,
    }


# 関数: `_load_freedom_choice_proxy` の入出力契約と処理意図を定義する。

def _load_freedom_choice_proxy(*, dataset_id: str, ds_dir: Path) -> dict[str, Any]:
    trial_path = ds_dir / "trial_based_counts.json"
    # 条件分岐: `trial_path.exists()` を満たす経路を評価する。
    if trial_path.exists():
        try:
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"supported": False, "reason": str(exc)}

        counts = trial.get("counts") if isinstance(trial.get("counts"), dict) else {}
        pair_trials = _reshape_2x2_counts(counts.get("trials_by_setting_pair"))
        marg = _setting_balance_from_marginals(
            alice_counts=[
                int(v)
                for v in _finite((counts.get("alice_trials_by_setting") or []))[:2]
            ],
            bob_counts=[
                int(v)
                for v in _finite((counts.get("bob_trials_by_setting") or []))[:2]
            ],
        )
        # 条件分岐: `pair_trials is not None` を満たす経路を評価する。
        if pair_trials is not None:
            pair = _setting_balance_from_pair_trials(pair_trials)
            # 条件分岐: `pair is not None` を満たす経路を評価する。
            if pair is not None:
                out = {
                    "supported": True,
                    "method": "trial_pair_uniformity",
                    "max_abs_bias": float(pair["pair_max_abs_bias_from_uniform"]),
                }
                out.update(pair)
                # 条件分岐: `isinstance(marg, dict)` を満たす経路を評価する。
                if isinstance(marg, dict):
                    out["marginal"] = marg

                return out

        # 条件分岐: `isinstance(marg, dict)` を満たす経路を評価する。

        if isinstance(marg, dict):
            out = {"supported": True, "method": "trial_marginal_balance", "max_abs_bias": float(marg["max_abs_bias"])}
            out.update(marg)
            return out

    norm_path = ds_dir / "normalized_events.json"
    # 条件分岐: `norm_path.exists()` を満たす経路を評価する。
    if norm_path.exists():
        try:
            norm = json.loads(norm_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"supported": False, "reason": str(exc)}

        preview = norm.get("preview") if isinstance(norm.get("preview"), dict) else {}
        a_counts = preview.get("a_setting_counts")
        b_counts = preview.get("b_setting_counts")
        # 条件分岐: `isinstance(a_counts, list) and isinstance(b_counts, list) and len(a_counts) =...` を満たす経路を評価する。
        if isinstance(a_counts, list) and isinstance(b_counts, list) and len(a_counts) == 2 and len(b_counts) == 2:
            marg = _setting_balance_from_marginals(
                alice_counts=[int(v) for v in _finite(a_counts)[:2]],
                bob_counts=[int(v) for v in _finite(b_counts)[:2]],
            )
            # 条件分岐: `isinstance(marg, dict)` を満たす経路を評価する。
            if isinstance(marg, dict):
                out = {"supported": True, "method": "event_ready_marginal_preview", "max_abs_bias": float(marg["max_abs_bias"])}
                out.update(marg)
                return out

    # Weihs only has independent event streams; estimate marginal setting balance from code parity (bit0).

    npz_path = ds_dir / "normalized_events.npz"
    # 条件分岐: `str(dataset_id).startswith("weihs1998_") and npz_path.exists()` を満たす経路を評価する。
    if str(dataset_id).startswith("weihs1998_") and npz_path.exists():
        try:
            data = np.load(npz_path)
            a_c = np.asarray(data["a_c"], dtype=np.int64).reshape(-1)
            b_c = np.asarray(data["b_c"], dtype=np.int64).reshape(-1)
            a_counts = [int(np.sum((a_c & 1) == 0)), int(np.sum((a_c & 1) == 1))]
            b_counts = [int(np.sum((b_c & 1) == 0)), int(np.sum((b_c & 1) == 1))]
            marg = _setting_balance_from_marginals(alice_counts=a_counts, bob_counts=b_counts)
            # 条件分岐: `isinstance(marg, dict)` を満たす経路を評価する。
            if isinstance(marg, dict):
                out = {"supported": True, "method": "time_tag_setting_bit_parity", "max_abs_bias": float(marg["max_abs_bias"])}
                out.update(marg)
                return out
        except Exception as exc:
            return {"supported": False, "reason": str(exc)}

    return {"supported": False, "reason": "no setting-balance source found"}


# 関数: `_write_selection_loophole_quantification` の入出力契約と処理意図を定義する。

def _write_selection_loophole_quantification(
    *,
    longterm: dict[str, Any],
) -> dict[str, Any]:
    out_json = OUT_BASE / "selection_loophole_quantification.json"
    out_csv = OUT_BASE / "selection_loophole_quantification.csv"
    out_png = OUT_BASE / "selection_loophole_quantification.png"

    datasets_lt = longterm.get("datasets") if isinstance(longterm.get("datasets"), list) else []
    thresholds_lt = longterm.get("thresholds") if isinstance(longterm.get("thresholds"), dict) else {}
    fair_th = float(thresholds_lt.get("selection_origin_ratio_min", 1.0))
    locality_th = float(thresholds_lt.get("delay_signature_z_min", 3.0))
    detection_th = float(2.0 / 3.0)
    freedom_bias_th = float(0.05)

    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for item in datasets_lt:
        # 条件分岐: `not isinstance(item, dict)` を満たす経路を評価する。
        if not isinstance(item, dict):
            continue

        ds_id = str(item.get("dataset_id") or "")
        display = str(item.get("display_name") or _dataset_display_name(ds_id))
        ds_dir = OUT_BASE / ds_id

        # fair sampling
        ratio = _safe_float(item.get("ratio"))
        fair_supported = ratio is not None
        fair_pass = bool(ratio <= fair_th) if fair_supported else None

        # detection efficiency proxy
        det = _load_detection_efficiency_proxy(ds_dir=ds_dir)
        eta_min = _safe_float(det.get("eta_min"))
        det_supported = bool(det.get("supported")) and eta_min is not None
        det_pass = bool(eta_min >= detection_th) if det_supported else None

        # locality proxy from delay setting-dependence z
        delay_z = _safe_float(item.get("delay_z_max"))
        locality_supported = delay_z is not None
        locality_pass = bool(delay_z <= locality_th) if locality_supported else None

        # freedom-of-choice proxy from setting-balance
        foc = _load_freedom_choice_proxy(dataset_id=ds_id, ds_dir=ds_dir)
        foc_bias = _safe_float(foc.get("max_abs_bias"))
        foc_supported = bool(foc.get("supported")) and foc_bias is not None
        foc_pass = bool(foc_bias <= freedom_bias_th) if foc_supported else None

        supported_flags = [x for x in [fair_supported, det_supported, locality_supported, foc_supported] if bool(x)]
        pass_flags = [x for x in [fair_pass, det_pass, locality_pass, foc_pass] if isinstance(x, bool)]
        overall_supported = bool(supported_flags)
        overall_pass = bool(pass_flags) and all(pass_flags) if overall_supported else None

        row = {
            "dataset_id": ds_id,
            "display_name": display,
            "statistic_family": item.get("statistic_family"),
            "fair_sampling": {
                "supported": fair_supported,
                "metric_name": "selection_ratio",
                "value": ratio,
                "threshold": fair_th,
                "pass": fair_pass,
                "direction": "lower_or_equal_is_closure",
                "source": "falsification_pack.ratio",
            },
            "detection": {
                "supported": det_supported,
                "metric_name": "eta_min",
                "value": eta_min,
                "threshold": detection_th,
                "pass": det_pass,
                "direction": "higher_or_equal_is_closure",
                "source": str(det.get("method") or det.get("reason") or "n/a"),
                "details": det,
            },
            "locality": {
                "supported": locality_supported,
                "metric_name": "delay_setting_dependence_z",
                "value": delay_z,
                "threshold": locality_th,
                "pass": locality_pass,
                "direction": "lower_or_equal_is_closure",
                "source": "falsification_pack.delay_signature",
            },
            "freedom_of_choice": {
                "supported": foc_supported,
                "metric_name": "setting_bias_max_abs",
                "value": foc_bias,
                "threshold": freedom_bias_th,
                "pass": foc_pass,
                "direction": "lower_or_equal_is_closure",
                "source": str(foc.get("method") or foc.get("reason") or "n/a"),
                "details": foc,
            },
            "overall": {
                "supported": overall_supported,
                "all_pass": overall_pass,
                "supported_item_n": int(sum(1 for x in [fair_supported, det_supported, locality_supported, foc_supported] if bool(x))),
            },
        }
        rows.append(row)

        csv_rows.append(
            {
                "dataset_id": ds_id,
                "display_name": display,
                "fair_sampling_ratio": ratio,
                "fair_sampling_threshold": fair_th,
                "fair_sampling_pass": fair_pass,
                "detection_eta_min": eta_min,
                "detection_threshold_eta_min": detection_th,
                "detection_pass": det_pass,
                "locality_delay_z": delay_z,
                "locality_threshold_delay_z": locality_th,
                "locality_pass": locality_pass,
                "freedom_bias_max_abs": foc_bias,
                "freedom_threshold_bias": freedom_bias_th,
                "freedom_pass": foc_pass,
                "overall_supported": overall_supported,
                "overall_all_pass": overall_pass,
            }
        )

    # 関数: `_summary_for` の入出力契約と処理意図を定義する。

    def _summary_for(name: str) -> dict[str, Any]:
        vals: list[float] = []
        pass_n = 0
        support_n = 0
        for r in rows:
            obj = r.get(name) if isinstance(r.get(name), dict) else {}
            # 条件分岐: `bool(obj.get("supported"))` を満たす経路を評価する。
            if bool(obj.get("supported")):
                support_n += 1
                v = _safe_float(obj.get("value"))
                # 条件分岐: `v is not None` を満たす経路を評価する。
                if v is not None:
                    vals.append(float(v))

                # 条件分岐: `isinstance(obj.get("pass"), bool) and bool(obj.get("pass"))` を満たす経路を評価する。

                if isinstance(obj.get("pass"), bool) and bool(obj.get("pass")):
                    pass_n += 1

        return {
            "n_supported": int(support_n),
            "n_pass": int(pass_n),
            "n_fail": int(max(support_n - pass_n, 0)),
            "median_value": float(np.median(np.asarray(vals, dtype=float))) if vals else None,
            "max_value": float(np.max(np.asarray(vals, dtype=float))) if vals else None,
            "min_value": float(np.min(np.asarray(vals, dtype=float))) if vals else None,
        }

    payload = {
        "generated_utc": _utc_now(),
        "phase": {
            "phase": 7,
            "step": "7.16.10",
            "name": "Bell: selection loophole full quantification (fair/detection/locality/freedom-of-choice)",
        },
        "main_script": {"path": "scripts/quantum/bell_primary_products.py", "repro": "python -B scripts/quantum/bell_primary_products.py"},
        "definitions": {
            "fair_sampling": "selection_ratio = Δ(stat) / σ_stat(median) from sweep summary; closure proxy if <= threshold.",
            "detection": "eta_min = min(eta_alice, eta_bob) from trial-based counts; closure proxy if >= 2/3 (operational).",
            "locality": "delay setting-dependence z = |Δmedian|/σ(Δmedian); closure proxy if <= threshold.",
            "freedom_of_choice": "setting bias max abs from uniform setting distribution; closure proxy if <= threshold.",
            "note": "All thresholds are operational for this repository; they are not universal loophole-closure criteria.",
        },
        "thresholds": {
            "fair_sampling_ratio_max": fair_th,
            "detection_eta_min_min": detection_th,
            "locality_delay_z_max": locality_th,
            "freedom_setting_bias_max_abs": freedom_bias_th,
        },
        "datasets": rows,
        "summary": {
            "fair_sampling": _summary_for("fair_sampling"),
            "detection": _summary_for("detection"),
            "locality": _summary_for("locality"),
            "freedom_of_choice": _summary_for("freedom_of_choice"),
            "overall": {
                "n_datasets": int(len(rows)),
                "n_all_supported": int(sum(1 for r in rows if bool((r.get("overall") or {}).get("supported")))),
                "n_all_pass": int(sum(1 for r in rows if bool((r.get("overall") or {}).get("all_pass")))),
            },
        },
        "outputs": {
            "json": "output/public/quantum/bell/selection_loophole_quantification.json",
            "csv": "output/public/quantum/bell/selection_loophole_quantification.csv",
            "png": "output/public/quantum/bell/selection_loophole_quantification.png",
        },
    }
    _write_json(out_json, payload)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_id",
                "display_name",
                "fair_sampling_ratio",
                "fair_sampling_threshold",
                "fair_sampling_pass",
                "detection_eta_min",
                "detection_threshold_eta_min",
                "detection_pass",
                "locality_delay_z",
                "locality_threshold_delay_z",
                "locality_pass",
                "freedom_bias_max_abs",
                "freedom_threshold_bias",
                "freedom_pass",
                "overall_supported",
                "overall_all_pass",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        payload["plot_written"] = False
        _write_json(out_json, payload)
        return payload

    labels = [str(r.get("display_name") or r.get("dataset_id") or "") for r in rows]
    x = np.arange(len(labels), dtype=float)

    # 関数: `_panel_data` の入出力契約と処理意図を定義する。
    def _panel_data(key: str) -> tuple[list[float], list[str], list[bool]]:
        values: list[float] = []
        colors: list[str] = []
        is_na: list[bool] = []
        for r in rows:
            obj = r.get(key) if isinstance(r.get(key), dict) else {}
            v = _safe_float(obj.get("value"))
            p = obj.get("pass")
            # 条件分岐: `v is None` を満たす経路を評価する。
            if v is None:
                values.append(0.0)
                colors.append("0.85")
                is_na.append(True)
            else:
                values.append(float(v))
                # 条件分岐: `isinstance(p, bool)` を満たす経路を評価する。
                if isinstance(p, bool):
                    colors.append("#2ca02c" if p else "#d62728")
                else:
                    colors.append("0.6")

                is_na.append(False)

        return values, colors, is_na

    fair_vals, fair_colors, fair_na = _panel_data("fair_sampling")
    det_vals, det_colors, det_na = _panel_data("detection")
    loc_vals, loc_colors, loc_na = _panel_data("locality")
    foc_vals, foc_colors, foc_na = _panel_data("freedom_of_choice")

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.4), dpi=170)
    panels = [
        (axes[0, 0], fair_vals, fair_colors, fair_na, fair_th, "Fair sampling (selection ratio)", "ratio"),
        (axes[0, 1], det_vals, det_colors, det_na, detection_th, "Detection (eta_min)", "eta_min"),
        (axes[1, 0], loc_vals, loc_colors, loc_na, locality_th, "Locality proxy (delay z)", "z"),
        (axes[1, 1], foc_vals, foc_colors, foc_na, freedom_bias_th, "Freedom-of-choice proxy (setting bias)", "abs bias"),
    ]
    for ax, vals, colors, na_mask, threshold, title, ylabel in panels:
        ax.bar(x, vals, color=colors, alpha=0.88)
        ax.axhline(float(threshold), color="0.2", ls="--", lw=1.0)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3, ls=":")
        for i, is_na in enumerate(na_mask):
            # 条件分岐: `is_na` を満たす経路を評価する。
            if is_na:
                ax.text(float(i), float(threshold) * 0.08, "n/a", ha="center", va="bottom", fontsize=8, color="0.35")

    fig.suptitle("Bell loophole quantification (operational proxies)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    payload["plot_written"] = True
    _write_json(out_json, payload)
    return payload


# 関数: `_corrcoef_columns` の入出力契約と処理意図を定義する。

def _corrcoef_columns(X: np.ndarray) -> np.ndarray:
    a = np.asarray(X, dtype=float)
    # 条件分岐: `a.ndim != 2` を満たす経路を評価する。
    if a.ndim != 2:
        raise ValueError("X must be 2D")

    n_rows, n_cols = int(a.shape[0]), int(a.shape[1])
    out = np.full((n_cols, n_cols), float("nan"), dtype=float)
    # 条件分岐: `n_rows <= 1 or n_cols == 0` を満たす経路を評価する。
    if n_rows <= 1 or n_cols == 0:
        return out

    for i in range(n_cols):
        xi = a[:, i]
        for j in range(n_cols):
            xj = a[:, j]
            m = np.isfinite(xi) & np.isfinite(xj)
            # 条件分岐: `int(np.sum(m)) <= 1` を満たす経路を評価する。
            if int(np.sum(m)) <= 1:
                continue

            s0 = float(np.std(xi[m], ddof=1))
            s1 = float(np.std(xj[m], ddof=1))
            # 条件分岐: `s0 <= 0.0 or s1 <= 0.0` を満たす経路を評価する。
            if s0 <= 0.0 or s1 <= 0.0:
                continue

            out[i, j] = float(np.cov(xi[m], xj[m], ddof=1)[0, 1] / (s0 * s1))

    return out


# 関数: `_write_systematics_decomposition_15items` の入出力契約と処理意図を定義する。

def _write_systematics_decomposition_15items(
    *,
    results: list[dict[str, Any]],
    cov_index: list[dict[str, Any]],
    falsification_pack: dict[str, Any],
) -> dict[str, Any]:
    item_specs: list[tuple[str, str]] = [
        ("coincidence_window", "coincidence window"),
        ("offset", "offset"),
        ("threshold", "threshold"),
        ("detector_efficiency_drift", "detector efficiency drift"),
        ("dark_count", "dark count"),
        ("accidental_correction", "accidental correction"),
        ("event_ready_definition", "event-ready definition"),
        ("trial_definition", "trial definition"),
        ("setting_switch_timing", "setting switch timing"),
        ("polarization_correction", "polarization correction"),
        ("transmission_loss", "transmission loss"),
        ("clock_sync", "clock synchronization"),
        ("dead_time", "dead time"),
        ("electronic_noise", "electronic noise"),
        ("environmental_variation", "environmental variation"),
    ]
    item_ids = [k for k, _ in item_specs]
    item_label = {k: v for k, v in item_specs}

    fc_map: dict[str, dict[str, Any]] = {}
    for d in falsification_pack.get("datasets", []) if isinstance(falsification_pack.get("datasets"), list) else []:
        # 条件分岐: `isinstance(d, dict)` を満たす経路を評価する。
        if isinstance(d, dict):
            fc_map[str(d.get("dataset_id") or "")] = d

    out_json = OUT_BASE / "systematics_decomposition_15items.json"
    out_csv = OUT_BASE / "systematics_decomposition_15items.csv"
    out_png = OUT_BASE / "systematics_decomposition_15items.png"

    per_dataset: list[dict[str, Any]] = []
    matrix_sigma: list[list[float]] = []
    csv_rows: list[dict[str, Any]] = []

    # 関数: `_make_item` の入出力契約と処理意図を定義する。
    def _make_item(value: float, source: str, method: str) -> dict[str, Any]:
        return {"delta_stat": float(max(value, 0.0)), "source": str(source), "method": str(method)}

    for r in results:
        ds = str(r.get("dataset_id") or "")
        stat_kind = str(r.get("statistic") or "")
        ds_dir = OUT_BASE / ds
        fc = fc_map.get(ds, {})

        w_obj = None
        o_obj = None
        nw_obj = None
        try:
            wp = ds_dir / "window_sweep_metrics.json"
            # 条件分岐: `wp.exists()` を満たす経路を評価する。
            if wp.exists():
                w_obj = json.loads(wp.read_text(encoding="utf-8"))
        except Exception:
            w_obj = None

        try:
            op = ds_dir / "offset_sweep_metrics.json"
            # 条件分岐: `op.exists()` を満たす経路を評価する。
            if op.exists():
                o_obj = json.loads(op.read_text(encoding="utf-8"))
        except Exception:
            o_obj = None

        try:
            npth = ds_dir / "natural_window_frozen.json"
            # 条件分岐: `npth.exists()` を満たす経路を評価する。
            if npth.exists():
                nw_obj = json.loads(npth.read_text(encoding="utf-8"))
        except Exception:
            nw_obj = None

        w_rows = list(w_obj.get("rows") or []) if isinstance(w_obj, dict) else []
        o_rows = list(o_obj.get("rows") or []) if isinstance(o_obj, dict) else []
        y_keys_window = ["S_fixed_abs", "S_fixed", "J_prob"]
        y_keys_offset = ["S", "S_combined", "J_prob"]

        base_stat = None
        base_sigma = None
        frozen_window_ns = None
        frozen_offset_ns = None
        # 条件分岐: `isinstance(nw_obj, dict)` を満たす経路を評価する。
        if isinstance(nw_obj, dict):
            b = nw_obj.get("baseline") if isinstance(nw_obj.get("baseline"), dict) else {}
            base_stat = _safe_float(b.get("statistic"))
            base_sigma = _safe_float(b.get("sigma_boot"))
            nwin = nw_obj.get("natural_window") if isinstance(nw_obj.get("natural_window"), dict) else {}
            frozen_window_ns = _safe_float(nwin.get("frozen_window_ns"))
            foff_ps = _safe_float(nwin.get("frozen_start_offset_ps"))
            frozen_offset_ns = (float(foff_ps) * 1e-3) if foff_ps is not None else None

        # 条件分岐: `base_sigma is None` を満たす経路を評価する。

        if base_sigma is None:
            base_sigma = _safe_float(fc.get("sigma_stat_med"))

        # 条件分岐: `base_sigma is None and isinstance(r.get("baseline"), dict)` を満たす経路を評価する。

        if base_sigma is None and isinstance(r.get("baseline"), dict):
            rb = r.get("baseline") if isinstance(r.get("baseline"), dict) else {}
            base_sigma = (
                _safe_float(rb.get("S_err"))
                or _safe_float(rb.get("S_combined_err"))
                or _safe_float(rb.get("J_trial_sigma_boot"))
            )

        # 条件分岐: `base_stat is None and isinstance(r.get("baseline"), dict)` を満たす経路を評価する。

        if base_stat is None and isinstance(r.get("baseline"), dict):
            rb = r.get("baseline") if isinstance(r.get("baseline"), dict) else {}
            base_stat = _safe_float(rb.get("S")) or _safe_float(rb.get("S_combined")) or _safe_float(rb.get("J_trial"))

        # 条件分岐: `frozen_window_ns is None and isinstance(r.get("baseline"), dict)` を満たす経路を評価する。

        if frozen_window_ns is None and isinstance(r.get("baseline"), dict):
            rb = r.get("baseline") if isinstance(r.get("baseline"), dict) else {}
            frozen_window_ns = _safe_float(rb.get("ref_window_ns"))

        # 条件分岐: `frozen_offset_ns is None and isinstance(r.get("baseline"), dict)` を満たす経路を評価する。

        if frozen_offset_ns is None and isinstance(r.get("baseline"), dict):
            rb = r.get("baseline") if isinstance(r.get("baseline"), dict) else {}
            frozen_offset_ns = _safe_float(rb.get("offset_ns"))

        # 条件分岐: `base_sigma is None` を満たす経路を評価する。

        if base_sigma is None:
            ws = []
            for row in w_rows:
                # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
                if not isinstance(row, dict):
                    continue

                ws.append(_safe_float(row.get("S_fixed_sigma_boot")))
                ws.append(_safe_float(row.get("J_sigma_boot")))

            wsf = _finite(ws)
            # 条件分岐: `wsf` を満たす経路を評価する。
            if wsf:
                base_sigma = float(np.median(np.asarray(wsf, dtype=float)))

        # 条件分岐: `base_sigma is None` を満たす経路を評価する。

        if base_sigma is None:
            os = []
            for row in o_rows:
                # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
                if not isinstance(row, dict):
                    continue

                os.append(_safe_float(row.get("S_err")))
                os.append(_safe_float(row.get("S_combined_err")))

            osf = _finite(os)
            # 条件分岐: `osf` を満たす経路を評価する。
            if osf:
                base_sigma = float(np.median(np.asarray(osf, dtype=float)))

        # 条件分岐: `base_sigma is None` を満たす経路を評価する。

        if base_sigma is None:
            base_sigma = 0.0

        stat_unit = "delta_abs_S" if stat_kind.startswith("CHSH") else "delta_J_prob"
        items: dict[str, dict[str, Any]] = {}

        # direct or near-direct ingredients
        delta_window = _safe_float(fc.get("delta_window"))
        # 条件分岐: `delta_window is None` を満たす経路を評価する。
        if delta_window is None:
            delta_window = _safe_float(r.get("window_delta"))

        delta_offset = _safe_float(fc.get("delta_offset"))
        # 条件分岐: `delta_offset is None` を満たす経路を評価する。
        if delta_offset is None:
            delta_offset = _safe_float(r.get("offset_delta"))

        # 条件分岐: `delta_window is not None` を満たす経路を評価する。

        if delta_window is not None:
            items["coincidence_window"] = _make_item(abs(delta_window), "direct", "delta_window from sweep summary")
        else:
            items["coincidence_window"] = _make_item(base_sigma * 0.50, "proxy_default", "fallback 0.5*sigma_stat")

        # 条件分岐: `delta_offset is not None` を満たす経路を評価する。

        if delta_offset is not None:
            items["offset"] = _make_item(abs(delta_offset), "direct", "delta_offset from sweep summary")
        else:
            items["offset"] = _make_item(base_sigma * 0.30, "proxy_default", "fallback 0.3*sigma_stat")

        frozen_w_row = None
        # 条件分岐: `frozen_window_ns is not None and w_rows` を満たす経路を評価する。
        if frozen_window_ns is not None and w_rows:
            frozen_w_row = _row_nearest(rows=w_rows, x_key="window_ns", target=float(frozen_window_ns))
        # 条件分岐: 前段条件が不成立で、`w_rows` を追加評価する。
        elif w_rows:
            frozen_w_row = w_rows[0] if isinstance(w_rows[0], dict) else None

        frozen_o_row = None
        # 条件分岐: `frozen_offset_ns is not None and o_rows` を満たす経路を評価する。
        if frozen_offset_ns is not None and o_rows:
            frozen_o_row = _row_nearest(rows=o_rows, x_key="start_offset_ps", target=float(frozen_offset_ns) * 1000.0)
        # 条件分岐: 前段条件が不成立で、`o_rows` を追加評価する。
        elif o_rows:
            frozen_o_row = o_rows[0] if isinstance(o_rows[0], dict) else None

        # accidental correction (direct when available)

        acc_val = None
        # 条件分岐: `isinstance(frozen_w_row, dict)` を満たす経路を評価する。
        if isinstance(frozen_w_row, dict):
            raw = _safe_float(frozen_w_row.get("S_fixed_abs"))
            sub = _safe_float(frozen_w_row.get("S_fixed_accidental_subtracted_abs"))
            # 条件分岐: `raw is None` を満たす経路を評価する。
            if raw is None:
                raw = _safe_float(frozen_w_row.get("J_prob"))

            # 条件分岐: `sub is None` を満たす経路を評価する。

            if sub is None:
                sub = _safe_float(frozen_w_row.get("J_prob_subtracted_clipped")) or _safe_float(
                    frozen_w_row.get("J_prob_subtracted")
                )

            # 条件分岐: `raw is not None and sub is not None` を満たす経路を評価する。

            if raw is not None and sub is not None:
                acc_val = abs(float(raw) - float(sub))

        # 条件分岐: `acc_val is not None` を満たす経路を評価する。

        if acc_val is not None:
            items["accidental_correction"] = _make_item(acc_val, "direct", "raw vs accidental-subtracted at frozen point")
        else:
            items["accidental_correction"] = _make_item(base_sigma * 0.20, "proxy_default", "fallback 0.2*sigma_stat")

        # trial / event-ready definitions

        trial_val = None
        # 条件分岐: `isinstance(w_obj, dict)` を満たす経路を評価する。
        if isinstance(w_obj, dict):
            tb = w_obj.get("trial_based") if isinstance(w_obj.get("trial_based"), dict) else {}
            trial_val = _safe_float(tb.get("J_prob"))

        # 条件分岐: `trial_val is not None and base_stat is not None` を満たす経路を評価する。

        if trial_val is not None and base_stat is not None:
            items["trial_definition"] = _make_item(
                abs(float(base_stat) - float(trial_val)),
                "direct",
                "difference between frozen statistic and trial-based statistic",
            )
        else:
            items["trial_definition"] = _make_item(base_sigma * 0.15, "proxy_default", "fallback 0.15*sigma_stat")

        event_ready_val = None
        # 条件分岐: `isinstance(o_obj, dict)` を満たす経路を評価する。
        if isinstance(o_obj, dict):
            ob = o_obj.get("baseline") if isinstance(o_obj.get("baseline"), dict) else {}
            event_ready_val = _safe_float(ob.get("S")) or _safe_float(ob.get("S_combined")) or _safe_float(ob.get("J_prob"))

        # 条件分岐: `event_ready_val is not None and base_stat is not None` を満たす経路を評価する。

        if event_ready_val is not None and base_stat is not None:
            items["event_ready_definition"] = _make_item(
                abs(float(base_stat) - float(event_ready_val)),
                "direct",
                "difference between frozen statistic and event-ready baseline",
            )
        else:
            items["event_ready_definition"] = _make_item(base_sigma * 0.20, "proxy_default", "fallback 0.2*sigma_stat")

        # slope-based local sensitivities

        slope_w = None
        # 条件分岐: `frozen_window_ns is not None and w_rows` を満たす経路を評価する。
        if frozen_window_ns is not None and w_rows:
            slope_w = _local_slope_at(
                rows=w_rows,
                x_key="window_ns",
                y_keys=y_keys_window,
                target=float(frozen_window_ns),
                x_scale=1.0,
            )

        w_values = _finite([_safe_float(row.get("window_ns")) for row in w_rows if isinstance(row, dict)])
        w_step = _median_step(values=[float(v) for v in w_values]) if w_values else None
        # 条件分岐: `slope_w is not None and w_step is not None` を満たす経路を評価する。
        if slope_w is not None and w_step is not None:
            items["threshold"] = _make_item(abs(float(slope_w) * float(w_step)), "proxy_indicator", "local window slope * median step")
        else:
            items["threshold"] = _make_item(base_sigma * 0.25, "proxy_default", "fallback 0.25*sigma_stat")

        slope_o = None
        target_off_ns = frozen_offset_ns if frozen_offset_ns is not None else 0.0
        # 条件分岐: `o_rows` を満たす経路を評価する。
        if o_rows:
            slope_o = _local_slope_at(
                rows=o_rows,
                x_key="start_offset_ps",
                y_keys=y_keys_offset,
                target=float(target_off_ns),
                x_scale=1e-3,
            )

        delay_abs_ns, delay_z = _max_abs_delay_median_ns(fc.get("delay_signature"))
        # 条件分岐: `delay_abs_ns is not None and slope_o is not None` を満たす経路を評価する。
        if delay_abs_ns is not None and slope_o is not None:
            items["setting_switch_timing"] = _make_item(
                abs(float(slope_o) * float(delay_abs_ns)),
                "proxy_indicator",
                "local offset slope * |delta_median_ns|",
            )
        # 条件分岐: 前段条件が不成立で、`delay_z is not None` を追加評価する。
        elif delay_z is not None:
            items["setting_switch_timing"] = _make_item(
                base_sigma * min(float(delay_z) / 3.0, 3.0),
                "proxy_indicator",
                "scaled by delay z-signature",
            )
        else:
            items["setting_switch_timing"] = _make_item(base_sigma * 0.20, "proxy_default", "fallback 0.2*sigma_stat")

        # count-based indicators at frozen point

        counts = _flatten_counts_from_row(frozen_w_row) if isinstance(frozen_w_row, dict) else []
        # 条件分岐: `not counts and isinstance(frozen_o_row, dict)` を満たす経路を評価する。
        if not counts and isinstance(frozen_o_row, dict):
            counts = _flatten_counts_from_row(frozen_o_row)

        counts_arr = np.asarray(_finite(counts), dtype=float) if counts else np.asarray([], dtype=float)
        counts_cv = None
        counts_imb = None
        # 条件分岐: `counts_arr.size > 0` を満たす経路を評価する。
        if counts_arr.size > 0:
            m = float(np.mean(counts_arr))
            # 条件分岐: `m > 0` を満たす経路を評価する。
            if m > 0:
                counts_cv = float(np.std(counts_arr, ddof=1) / m) if counts_arr.size > 1 else 0.0
                counts_imb = float((np.max(counts_arr) - np.min(counts_arr)) / m)

        # 条件分岐: `counts_cv is not None` を満たす経路を評価する。

        if counts_cv is not None:
            items["detector_efficiency_drift"] = _make_item(
                base_sigma * max(counts_cv, 0.0),
                "proxy_indicator",
                "sigma_stat * coefficient of variation of setting counts",
            )
        else:
            items["detector_efficiency_drift"] = _make_item(base_sigma * 0.35, "proxy_default", "fallback 0.35*sigma_stat")

        acc_rate = None
        # 条件分岐: `isinstance(frozen_w_row, dict)` を満たす経路を評価する。
        if isinstance(frozen_w_row, dict):
            p_acc = _safe_float(frozen_w_row.get("pairs_total_accidental"))
            p_tot = _safe_float(frozen_w_row.get("pairs_total"))
            # 条件分岐: `p_acc is not None and p_tot is not None and p_tot > 0` を満たす経路を評価する。
            if p_acc is not None and p_tot is not None and p_tot > 0:
                acc_rate = float(p_acc / p_tot)

        # 条件分岐: `acc_rate is not None` を満たす経路を評価する。

        if acc_rate is not None:
            items["dark_count"] = _make_item(base_sigma * max(acc_rate, 0.0), "proxy_indicator", "sigma_stat * accidental fraction")
        else:
            items["dark_count"] = _make_item(base_sigma * 0.20, "proxy_default", "fallback 0.2*sigma_stat")

        # 条件分岐: `counts_imb is not None` を満たす経路を評価する。

        if counts_imb is not None:
            items["polarization_correction"] = _make_item(
                base_sigma * max(counts_imb, 0.0),
                "proxy_indicator",
                "sigma_stat * setting-count imbalance",
            )
        else:
            items["polarization_correction"] = _make_item(base_sigma * 0.30, "proxy_default", "fallback 0.3*sigma_stat")

        click_asym = None
        # 条件分岐: `isinstance(frozen_w_row, dict)` を満たす経路を評価する。
        if isinstance(frozen_w_row, dict):
            a_sum = _finite(frozen_w_row.get("click_a_by_setting") or [])
            b_sum = _finite(frozen_w_row.get("click_b_by_setting") or [])
            # 条件分岐: `a_sum and b_sum` を満たす経路を評価する。
            if a_sum and b_sum:
                av = float(sum(a_sum))
                bv = float(sum(b_sum))
                den = max((av + bv) * 0.5, 1e-12)
                click_asym = float(abs(av - bv) / den)

        # 条件分岐: `click_asym is not None` を満たす経路を評価する。

        if click_asym is not None:
            items["transmission_loss"] = _make_item(base_sigma * max(click_asym, 0.0), "proxy_indicator", "sigma_stat * Alice/Bob click asymmetry")
        else:
            items["transmission_loss"] = _make_item(base_sigma * 0.25, "proxy_default", "fallback 0.25*sigma_stat")

        ks_legacy = None
        # 条件分岐: `isinstance(fc.get("ks_delay"), dict)` を満たす経路を評価する。
        if isinstance(fc.get("ks_delay"), dict):
            ks_vals = _finite(fc.get("ks_delay").values())
            # 条件分岐: `ks_vals` を満たす経路を評価する。
            if ks_vals:
                ks_legacy = float(max(ks_vals))

        # 条件分岐: `frozen_offset_ns is not None and frozen_window_ns is not None and frozen_wind...` を満たす経路を評価する。

        if frozen_offset_ns is not None and frozen_window_ns is not None and frozen_window_ns > 0:
            items["clock_sync"] = _make_item(
                base_sigma * abs(float(frozen_offset_ns) / float(frozen_window_ns)),
                "proxy_indicator",
                "sigma_stat * |frozen_offset|/frozen_window",
            )
        # 条件分岐: 前段条件が不成立で、`ks_legacy is not None` を追加評価する。
        elif ks_legacy is not None:
            items["clock_sync"] = _make_item(base_sigma * ks_legacy, "proxy_indicator", "sigma_stat * KS delay proxy")
        else:
            items["clock_sync"] = _make_item(base_sigma * 0.15, "proxy_default", "fallback 0.15*sigma_stat")

        pair_density = None
        # 条件分岐: `isinstance(frozen_w_row, dict) and w_rows` を満たす経路を評価する。
        if isinstance(frozen_w_row, dict) and w_rows:
            p0 = _safe_float(frozen_w_row.get("pairs_total"))
            pmax = max(_finite([_safe_float(row.get("pairs_total")) for row in w_rows if isinstance(row, dict)]), default=None)
            # 条件分岐: `p0 is not None and pmax is not None and pmax > 0` を満たす経路を評価する。
            if p0 is not None and pmax is not None and pmax > 0:
                pair_density = float(p0 / pmax)

        # 条件分岐: `pair_density is not None` を満たす経路を評価する。

        if pair_density is not None:
            items["dead_time"] = _make_item(base_sigma * max(pair_density, 0.0) * 0.20, "proxy_indicator", "sigma_stat * pair-density proxy")
        else:
            items["dead_time"] = _make_item(base_sigma * 0.20, "proxy_default", "fallback 0.2*sigma_stat")

        rel_noise = 0.0
        # 条件分岐: `base_stat is not None and base_sigma is not None` を満たす経路を評価する。
        if base_stat is not None and base_sigma is not None:
            den = max(abs(float(base_stat)), max(float(base_sigma), 1e-12))
            rel_noise = float(abs(float(base_sigma)) / den)

        items["electronic_noise"] = _make_item(base_sigma * rel_noise, "proxy_indicator", "sigma_stat * relative bootstrap noise")

        spread_vals: list[float] = []
        # 条件分岐: `w_rows` を満たす経路を評価する。
        if w_rows:
            spread_vals.extend(_finite([_safe_float(row.get("S_fixed_abs")) for row in w_rows if isinstance(row, dict)]))
            spread_vals.extend(_finite([_safe_float(row.get("J_prob")) for row in w_rows if isinstance(row, dict)]))

        # 条件分岐: `o_rows` を満たす経路を評価する。

        if o_rows:
            spread_vals.extend(_finite([_safe_float(row.get("S")) for row in o_rows if isinstance(row, dict)]))
            spread_vals.extend(_finite([_safe_float(row.get("S_combined")) for row in o_rows if isinstance(row, dict)]))

        spread_rel = _relative_spread(spread_vals)
        # 条件分岐: `spread_rel is not None` を満たす経路を評価する。
        if spread_rel is not None:
            items["environmental_variation"] = _make_item(
                base_sigma * max(spread_rel, 0.0),
                "proxy_indicator",
                "sigma_stat * relative spread across sweep",
            )
        else:
            items["environmental_variation"] = _make_item(base_sigma * 0.25, "proxy_default", "fallback 0.25*sigma_stat")

        # Final guard: ensure all 15 items exist with finite values.

        for key in item_ids:
            # 条件分岐: `key not in items` を満たす経路を評価する。
            if key not in items:
                items[key] = _make_item(base_sigma * 0.20, "proxy_default", "fill-missing fallback")

            v = _safe_float(items[key].get("delta_stat"))
            # 条件分岐: `v is None` を満たす経路を評価する。
            if v is None:
                items[key]["delta_stat"] = float(base_sigma * 0.20)

        vals = [float(items[k]["delta_stat"]) for k in item_ids]
        # 条件分岐: `base_sigma > 0` を満たす経路を評価する。
        if base_sigma > 0:
            vals_sigma = [float(v / base_sigma) for v in vals]
        else:
            vals_sigma = [0.0 for _ in vals]

        matrix_sigma.append(vals_sigma)

        total_l2 = float(math.sqrt(float(np.sum(np.asarray(vals, dtype=float) ** 2))))
        total_l1 = float(np.sum(np.asarray(vals, dtype=float)))
        sys_over_stat_l2 = float(total_l2 / base_sigma) if base_sigma > 0 else None

        ds_entry = {
            "dataset_id": ds,
            "display_name": _dataset_display_name(ds),
            "statistic": str(stat_kind),
            "stat_unit": stat_unit,
            "baseline": {
                "statistic": base_stat,
                "sigma_stat": float(base_sigma),
                "frozen_window_ns": frozen_window_ns,
                "frozen_offset_ns": frozen_offset_ns,
            },
            "items": items,
            "summary": {
                "delta_stat_total_l2": total_l2,
                "delta_stat_total_l1": total_l1,
                "sys_over_stat_l2": sys_over_stat_l2,
            },
        }
        per_dataset.append(ds_entry)

        for key in item_ids:
            v = float(items[key]["delta_stat"])
            v_sigma = float(v / base_sigma) if base_sigma > 0 else None
            csv_rows.append(
                {
                    "dataset_id": ds,
                    "display_name": _dataset_display_name(ds),
                    "statistic": str(stat_kind),
                    "item_id": key,
                    "item_label": item_label[key],
                    "delta_stat": v,
                    "delta_over_sigma": v_sigma,
                    "source": str(items[key].get("source") or ""),
                    "method": str(items[key].get("method") or ""),
                }
            )

    X = np.asarray(matrix_sigma, dtype=float) if matrix_sigma else np.zeros((0, len(item_ids)), dtype=float)
    corr_items = _corrcoef_columns(X) if X.size else np.zeros((len(item_ids), len(item_ids)), dtype=float)
    pair_rel: list[dict[str, Any]] = []
    corr_abs_th = 0.6
    for i in range(len(item_ids)):
        for j in range(i + 1, len(item_ids)):
            c = _safe_float(corr_items[i, j])
            # 条件分岐: `c is None` を満たす経路を評価する。
            if c is None:
                continue

            pair_rel.append(
                {
                    "item_i": item_ids[i],
                    "item_j": item_ids[j],
                    "corr": float(c),
                    "relation": "correlated" if abs(float(c)) >= corr_abs_th else "weak_or_independent",
                }
            )

    pair_rel.sort(key=lambda d: abs(float(d.get("corr") or 0.0)), reverse=True)

    item_summary: list[dict[str, Any]] = []
    for idx, key in enumerate(item_ids):
        vals = [float(d["items"][key]["delta_stat"]) for d in per_dataset]
        vals_sigma = [float(v) for v in X[:, idx].tolist()] if X.shape[0] > 0 else []
        item_summary.append(
            {
                "item_id": key,
                "item_label": item_label[key],
                "median_delta_stat": float(np.median(np.asarray(vals, dtype=float))) if vals else None,
                "max_delta_stat": float(np.max(np.asarray(vals, dtype=float))) if vals else None,
                "median_delta_over_sigma": float(np.median(np.asarray(vals_sigma, dtype=float))) if vals_sigma else None,
                "max_delta_over_sigma": float(np.max(np.asarray(vals_sigma, dtype=float))) if vals_sigma else None,
            }
        )

    payload = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 7, "step": "7.16.8", "name": "Bell: systematics full decomposition (15 items)"},
        "main_script": {"path": "scripts/quantum/bell_primary_products.py", "repro": "python -B scripts/quantum/bell_primary_products.py"},
        "definitions": {
            "delta_stat": "absolute contribution in observed Bell statistic unit (|S| for CHSH datasets, J_prob for CH datasets)",
            "delta_over_sigma": "delta_stat / sigma_stat",
            "correlated_if_abs_corr_ge": float(corr_abs_th),
            "note": "Items are operationally decomposed from direct sweep metrics and fixed proxy indicators when direct sweeps are unavailable.",
        },
        "item_catalog": [{"item_id": k, "label": item_label[k]} for k in item_ids],
        "datasets": per_dataset,
        "item_summary": item_summary,
        "item_correlation": {
            "item_order": item_ids,
            "matrix": _matrix_to_json(corr_items),
            "top_pairs": pair_rel[:30],
        },
        "outputs": {
            "json": "output/public/quantum/bell/systematics_decomposition_15items.json",
            "csv": "output/public/quantum/bell/systematics_decomposition_15items.csv",
            "png": "output/public/quantum/bell/systematics_decomposition_15items.png",
        },
    }
    _write_json(out_json, payload)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_id",
                "display_name",
                "statistic",
                "item_id",
                "item_label",
                "delta_stat",
                "delta_over_sigma",
                "source",
                "method",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        payload["plot_written"] = False
        _write_json(out_json, payload)
        return payload

    med_sigma = [float(np.median(X[:, i])) if X.shape[0] > 0 else 0.0 for i in range(len(item_ids))]
    order = np.argsort(np.asarray(med_sigma, dtype=float))[::-1]
    labels_sorted = [item_label[item_ids[int(i)]] for i in order]
    vals_sorted = [med_sigma[int(i)] for i in order]
    ds_labels = [str(d.get("display_name") or d.get("dataset_id") or "") for d in per_dataset]
    sys_over = [d.get("summary", {}).get("sys_over_stat_l2") for d in per_dataset]
    sys_over_plot = [float(v) if _safe_float(v) is not None else 0.0 for v in sys_over]

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.2), dpi=170)
    y = np.arange(len(labels_sorted))
    axes[0].barh(y, vals_sorted, color="tab:blue", alpha=0.82)
    axes[0].set_yticks(y, labels_sorted, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("median |Δstat| / σ_stat")
    axes[0].set_title("15-item systematics budget (median)")
    axes[0].grid(True, axis="x", alpha=0.3, ls=":")

    axes[1].bar(np.arange(len(ds_labels)), sys_over_plot, color="tab:orange", alpha=0.85)
    axes[1].set_xticks(np.arange(len(ds_labels)), ds_labels, rotation=25, ha="right")
    axes[1].set_ylabel("sys/stat (L2)")
    axes[1].set_title("Per-dataset total systematics")
    axes[1].grid(True, axis="y", alpha=0.3, ls=":")

    im = axes[2].imshow(corr_items, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[2].set_xticks(np.arange(len(item_ids)), [item_ids[i] for i in range(len(item_ids))], rotation=90, fontsize=7)
    axes[2].set_yticks(np.arange(len(item_ids)), [item_ids[i] for i in range(len(item_ids))], fontsize=7)
    axes[2].set_title("Item correlation (across datasets)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Bell systematics decomposition (15 items; operational)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    payload["plot_written"] = True
    _write_json(out_json, payload)
    return payload


# 関数: `_recommend_plateau_x` の入出力契約と処理意図を定義する。

def _recommend_plateau_x(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    plateau_fraction: float,
) -> float | None:
    pts: list[tuple[float, float]] = []
    for r in rows:
        try:
            x = float(r.get(x_key))
            y = float(r.get(y_key))
        except Exception:
            continue

        # 条件分岐: `math.isfinite(x) and math.isfinite(y)` を満たす経路を評価する。

        if math.isfinite(x) and math.isfinite(y):
            pts.append((x, y))

    # 条件分岐: `not pts` を満たす経路を評価する。

    if not pts:
        return None

    pts.sort(key=lambda t: t[0])
    y_max = max(y for _, y in pts)
    target = float(plateau_fraction) * float(y_max)
    for x, y in pts:
        # 条件分岐: `y >= target` を満たす経路を評価する。
        if y >= target:
            return float(x)

    return float(pts[-1][0])


# 関数: `_load_time_tag_times_seconds` の入出力契約と処理意図を定義する。

def _load_time_tag_times_seconds(*, ds_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    npz_path = ds_dir / "normalized_events.npz"
    # 条件分岐: `not npz_path.exists()` を満たす経路を評価する。
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path)
    except Exception:
        return None

    # 条件分岐: `"a_t_s" in data.files and "b_t_s" in data.files` を満たす経路を評価する。

    if "a_t_s" in data.files and "b_t_s" in data.files:
        t_a = np.asarray(data["a_t_s"], dtype=np.float64)
        t_b = np.asarray(data["b_t_s"], dtype=np.float64)
        return t_a, t_b, {"schema": "a_t_s/b_t_s", "time_unit": "s"}

    # NIST normalized clicks (timetag counts + seconds_per_timetag)

    if (
        "alice_click_t" in data.files
        and "bob_click_t" in data.files
        and "seconds_per_timetag" in data.files
    ):
        seconds_per = float(np.asarray(data["seconds_per_timetag"], dtype=np.float64).reshape(-1)[0])
        t_a = np.asarray(data["alice_click_t"], dtype=np.float64) * seconds_per
        t_b = np.asarray(data["bob_click_t"], dtype=np.float64) * seconds_per
        return t_a, t_b, {"schema": "alice_click_t/bob_click_t * seconds_per_timetag", "time_unit": "s"}

    return None


# 関数: `_recommend_time_tag_window_from_dt_peak` の入出力契約と処理意図を定義する。

def _recommend_time_tag_window_from_dt_peak(
    *,
    t_a_s: np.ndarray,
    t_b_s: np.ndarray,
    sample_max: int = 200_000,
    hist_half_range_ns: float = 200.0,
    hist_bin_width_ns: float = 0.5,
    background_exclusion_ns: float = 50.0,
    background_threshold_ksigma: float = 3.0,
    signal_fraction: float = 0.99,
    drift_window_ns: float = 10.0,
    drift_chunks: int = 10,
    drift_min_points: int = 30,
) -> tuple[float | None, dict[str, Any] | None]:
    """
    Define a 'natural' coincidence half-window from the timing distribution only
    (delay/jitter/drift + accidental background proxy), without maximizing CHSH/CH.

    Method:
    - For sampled Alice events, compute nearest-neighbor dt to Bob.
    - Histogram dt in a fixed range around the median; find the peak (coincidence offset).
    - Estimate a background level from far-from-peak bins.
    - Convert histogram counts into "signal above background" with a fixed kσ threshold.
    - Choose the smallest |dt| that contains a fixed fraction of that signal.
    - Add half the robust drift span (p95-p05 of chunk medians near the peak).
    """
    t_a = np.asarray(t_a_s, dtype=np.float64)
    t_b = np.asarray(t_b_s, dtype=np.float64)
    # 条件分岐: `t_a.size == 0 or t_b.size == 0` を満たす経路を評価する。
    if t_a.size == 0 or t_b.size == 0:
        return None, None

    # 条件分岐: `t_a.size > sample_max` を満たす経路を評価する。

    if t_a.size > sample_max:
        idx = np.linspace(0, t_a.size - 1, sample_max, dtype=np.int64)
        a = t_a[idx]
    else:
        a = t_a

    j = np.searchsorted(t_b, a)
    j2 = np.clip(j, 0, t_b.size - 1)
    j1 = np.clip(j - 1, 0, t_b.size - 1)
    choose_j2 = np.abs(t_b[j2] - a) < np.abs(t_b[j1] - a)
    jn = np.where(choose_j2, j2, j1)
    dt = t_b[jn] - a
    med = float(np.median(dt))

    half = float(hist_half_range_ns) * 1e-9
    m = np.abs(dt - med) <= half
    # 条件分岐: `int(np.sum(m)) < 1000` を満たす経路を評価する。
    if int(np.sum(m)) < 1000:
        return None, None

    sel_dt = dt[m]
    sel_t = a[m]

    bin_w = float(hist_bin_width_ns) * 1e-9
    n_bins = int(max(50, math.ceil((2.0 * half) / bin_w)))
    edges = np.linspace(med - half, med + half, n_bins + 1, dtype=float)
    counts, _ = np.histogram(sel_dt, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_idx = int(np.argmax(counts))
    peak_center = float(centers[peak_idx])

    bg_excl = float(background_exclusion_ns) * 1e-9
    bg_mask = np.abs(centers - peak_center) >= bg_excl
    bg = float(np.median(counts[bg_mask])) if int(np.sum(bg_mask)) > 0 else float(np.median(counts))
    bg = max(0.0, bg)
    thr = bg + float(background_threshold_ksigma) * math.sqrt(bg if bg > 0 else 1.0)
    signal = np.maximum(0.0, counts.astype(np.float64) - thr)

    total_signal = float(np.sum(signal))
    # 条件分岐: `not math.isfinite(total_signal) or total_signal <= 0` を満たす経路を評価する。
    if not math.isfinite(total_signal) or total_signal <= 0:
        return None, None

    order = np.argsort(np.abs(centers - peak_center))
    dist = np.abs(centers[order] - peak_center)
    cs = np.cumsum(signal[order])
    k = int(np.searchsorted(cs, float(signal_fraction) * float(cs[-1])))
    k = min(k, int(dist.size) - 1)
    width_s = float(dist[k])
    width_ns = float(width_s * 1e9)

    # Drift estimate (robust span of chunk medians near the peak)
    drift_half_ns = 0.0
    near_half_s = float(drift_window_ns) * 1e-9
    near_mask = np.abs(sel_dt - peak_center) <= near_half_s
    near_dt = sel_dt[near_mask]
    near_t = sel_t[near_mask]
    # 条件分岐: `near_dt.size >= 10 * drift_min_points` を満たす経路を評価する。
    if near_dt.size >= 10 * drift_min_points:
        t0 = float(np.min(near_t))
        t1 = float(np.max(near_t))
        # 条件分岐: `math.isfinite(t0) and math.isfinite(t1) and t1 > t0` を満たす経路を評価する。
        if math.isfinite(t0) and math.isfinite(t1) and t1 > t0:
            edges_t = np.linspace(t0, t1, int(drift_chunks) + 1, dtype=float)
            meds: list[float] = []
            for i in range(int(drift_chunks)):
                mm = (near_t >= edges_t[i]) & (near_t < edges_t[i + 1])
                # 条件分岐: `int(np.sum(mm)) < int(drift_min_points)` を満たす経路を評価する。
                if int(np.sum(mm)) < int(drift_min_points):
                    continue

                meds.append(float(np.median(near_dt[mm])))

            # 条件分岐: `len(meds) >= 2` を満たす経路を評価する。

            if len(meds) >= 2:
                drift_span_ns = float(
                    (np.quantile(np.asarray(meds, dtype=float), 0.95) - np.quantile(np.asarray(meds, dtype=float), 0.05))
                    * 1e9
                )
                drift_half_ns = 0.5 * drift_span_ns if math.isfinite(drift_span_ns) and drift_span_ns > 0 else 0.0

    rec = width_ns + drift_half_ns
    method: dict[str, Any] = {
        "name": "dt_peak_hist_signal_fraction",
        "signal_fraction": float(signal_fraction),
        "sample_max": int(sample_max),
        "hist_half_range_ns": float(hist_half_range_ns),
        "hist_bin_width_ns": float(hist_bin_width_ns),
        "background_exclusion_ns": float(background_exclusion_ns),
        "background_median_per_bin": float(bg),
        "background_threshold_ksigma": float(background_threshold_ksigma),
        "peak_center_ns": float(peak_center * 1e9),
        "selected_dt_count": int(sel_dt.size),
        "near_peak_count": int(near_dt.size),
        "width_ns": float(width_ns),
        "drift_window_ns": float(drift_window_ns),
        "drift_half_ns": float(drift_half_ns),
    }
    return float(rec), method


# 関数: `_nist_trial_match_recommended_window_ns` の入出力契約と処理意図を定義する。

def _nist_trial_match_recommended_window_ns(
    *, ds_dir: Path, rows: list[dict[str, Any]]
) -> tuple[float | None, dict[str, Any] | None]:
    """
    For NIST time-tag data we also have an independent, trial-based coincidence definition
    from the published build.hdf5. To avoid "p-hacking" optics, define a single recommended
    time-tag half-window as the window where the greedy coincidence-based pairs_total best
    matches the trial-based coincidence total.

    This does NOT maximize J or S; it matches an independent coincidence definition.
    """
    trial_path = ds_dir / "trial_based_counts.json"
    # 条件分岐: `not trial_path.exists()` を満たす経路を評価する。
    if not trial_path.exists():
        return None, None

    try:
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    coinc = ((trial.get("counts") or {}).get("coinc_by_setting_pair") or [])
    try:
        trial_total = int(sum(int(x) for row in coinc for x in row))
    except Exception:
        return None, None

    best: tuple[int, float, int] | None = None  # (abs_delta_pairs, window_ns, pairs_total)
    for r in rows:
        try:
            w = float(r.get("window_ns"))
            p = int(r.get("pairs_total"))
        except Exception:
            continue

        # 条件分岐: `not math.isfinite(w) or w <= 0` を満たす経路を評価する。

        if not math.isfinite(w) or w <= 0:
            continue

        cand = (abs(p - trial_total), float(w), int(p))
        # 条件分岐: `best is None or cand < best` を満たす経路を評価する。
        if best is None or cand < best:
            best = cand

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        return None, None

    delta_pairs, rec_w, pairs_total = best
    method = {
        "name": "trial_match_pairs_total",
        "note": "Recommended window matches the trial-based coincidence total (independent pipeline).",
        "trial_total_coinc": int(trial_total),
        "selected_pairs_total": int(pairs_total),
        "abs_delta_pairs": int(delta_pairs),
    }
    return float(rec_w), method


# 関数: `_recommend_natural_window` の入出力契約と処理意図を定義する。

def _recommend_natural_window(
    *,
    dataset_id: str,
    ds_dir: Path,
    rows: list[dict[str, Any]],
    plateau_fraction: float,
) -> tuple[float | None, dict[str, Any]]:
    # 条件分岐: `dataset_id.startswith("nist_")` を満たす経路を評価する。
    if dataset_id.startswith("nist_"):
        rec, method = _nist_trial_match_recommended_window_ns(ds_dir=ds_dir, rows=rows)
        # 条件分岐: `rec is not None and method is not None` を満たす経路を評価する。
        if rec is not None and method is not None:
            return rec, method

    # 条件分岐: `dataset_id.startswith("kwiat2013_")` を満たす経路を評価する。

    if dataset_id.startswith("kwiat2013_"):
        # dataset recommendation (Illinois page): "use a coincidence window around 18,000 timebins"
        # (PC on for 2us = 12,800 bins at 6.4 GHz).
        try:
            wj = json.loads((ds_dir / "window_sweep_metrics.json").read_text(encoding="utf-8"))
            cfg = wj.get("config") if isinstance(wj.get("config"), dict) else {}
            bins_per_s = cfg.get("timebins_per_second")
            rec_bins = cfg.get("recommended_window_bins") or cfg.get("ref_window_bins") or 18_000
            # 条件分岐: `bins_per_s is not None and float(bins_per_s) > 0` を満たす経路を評価する。
            if bins_per_s is not None and float(bins_per_s) > 0:
                rec = float(rec_bins) / float(bins_per_s) * 1e9
                method = {
                    "name": "dataset_recommendation",
                    "recommended_window_bins": int(rec_bins),
                    "timebins_per_second": float(bins_per_s),
                    "note": "Use the public dataset guidance (data_organization.txt) to avoid p-hacking optics.",
                }
                return float(rec), method
        except Exception:
            pass

    # For time-tag datasets: derive from dt peak width/drift rather than maximizing a Bell statistic.

    time_tag = _load_time_tag_times_seconds(ds_dir=ds_dir)
    # 条件分岐: `time_tag is not None` を満たす経路を評価する。
    if time_tag is not None:
        t_a, t_b, meta = time_tag
        rec, method = _recommend_time_tag_window_from_dt_peak(t_a_s=t_a, t_b_s=t_b)
        # 条件分岐: `rec is not None and method is not None` を満たす経路を評価する。
        if rec is not None and method is not None:
            method = dict(method)
            method["time_tag_schema"] = meta
            return rec, method

    rec = _recommend_plateau_x(
        rows=rows,
        x_key="window_ns",
        y_key="pairs_total",
        plateau_fraction=float(plateau_fraction),
    )
    return rec, {"name": "pairs_plateau", "plateau_fraction": float(plateau_fraction)}


# クラス: `ChshVariant` の責務と境界条件を定義する。

@dataclass(frozen=True)
class ChshVariant:
    swap_a: bool
    swap_b: bool
    sign_matrix: tuple[tuple[int, int], tuple[int, int]]  # each entry ±1


# 関数: `_chsh_sign_patterns` の入出力契約と処理意図を定義する。

def _chsh_sign_patterns() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for mask in range(16):
        s = np.array(
            [
                [1 if (mask >> 0) & 1 else -1, 1 if (mask >> 1) & 1 else -1],
                [1 if (mask >> 2) & 1 else -1, 1 if (mask >> 3) & 1 else -1],
            ],
            dtype=np.int8,
        )
        # 条件分岐: `int(np.prod(s)) == -1` を満たす経路を評価する。
        if int(np.prod(s)) == -1:
            patterns.append(s)

    return patterns


_CHSH_SIGNS = _chsh_sign_patterns()


# 関数: `_apply_chsh_variant` の入出力契約と処理意図を定義する。
def _apply_chsh_variant(E: np.ndarray, variant: ChshVariant) -> float:
    E2 = np.asarray(E, dtype=float).copy()
    # 条件分岐: `E2.shape != (2, 2)` を満たす経路を評価する。
    if E2.shape != (2, 2):
        raise ValueError("E must be 2x2")

    # 条件分岐: `variant.swap_a` を満たす経路を評価する。

    if variant.swap_a:
        E2 = E2[[1, 0], :]

    # 条件分岐: `variant.swap_b` を満たす経路を評価する。

    if variant.swap_b:
        E2 = E2[:, [1, 0]]

    s = np.asarray(variant.sign_matrix, dtype=np.int8)
    return float(np.sum(s * E2))


# 関数: `_best_chsh_variant` の入出力契約と処理意図を定義する。

def _best_chsh_variant(E: np.ndarray) -> tuple[ChshVariant, float]:
    # 条件分岐: `np.asarray(E).shape != (2, 2)` を満たす経路を評価する。
    if np.asarray(E).shape != (2, 2):
        raise ValueError("E must be 2x2")

    best: tuple[ChshVariant, float] | None = None
    for swap_a in (False, True):
        for swap_b in (False, True):
            E2 = np.asarray(E, dtype=float).copy()
            # 条件分岐: `swap_a` を満たす経路を評価する。
            if swap_a:
                E2 = E2[[1, 0], :]

            # 条件分岐: `swap_b` を満たす経路を評価する。

            if swap_b:
                E2 = E2[:, [1, 0]]

            for s in _CHSH_SIGNS:
                v = float(np.sum(s * E2))
                # 条件分岐: `best is None or abs(v) > abs(best[1])` を満たす経路を評価する。
                if best is None or abs(v) > abs(best[1]):
                    variant = ChshVariant(
                        swap_a=swap_a,
                        swap_b=swap_b,
                        sign_matrix=((int(s[0, 0]), int(s[0, 1])), (int(s[1, 0]), int(s[1, 1]))),
                    )
                    best = (variant, v)

    assert best is not None
    return best


# 関数: `_bootstrap_chsh_s_sigma` の入出力契約と処理意図を定義する。

def _bootstrap_chsh_s_sigma(
    *,
    E: np.ndarray,
    n: np.ndarray,
    variant: ChshVariant,
    n_boot: int,
    seed: int,
) -> float:
    E = np.asarray(E, dtype=float)
    n = np.asarray(n, dtype=np.int64)
    # 条件分岐: `E.shape != (2, 2) or n.shape != (2, 2)` を満たす経路を評価する。
    if E.shape != (2, 2) or n.shape != (2, 2):
        raise ValueError("E and n must be 2x2")

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(n_boot)):
        Eboot = np.empty((2, 2), dtype=float)
        for a in (0, 1):
            for b in (0, 1):
                N = int(n[a, b])
                # 条件分岐: `N <= 0 or not math.isfinite(float(E[a, b]))` を満たす経路を評価する。
                if N <= 0 or not math.isfinite(float(E[a, b])):
                    Eboot[a, b] = float("nan")
                    continue

                p = 0.5 * (1.0 + float(E[a, b]))
                p = min(1.0, max(0.0, p))
                k = int(rng.binomial(N, p))
                Eboot[a, b] = (2.0 * k - N) / N

        # 条件分岐: `np.isfinite(Eboot).all()` を満たす経路を評価する。

        if np.isfinite(Eboot).all():
            samples.append(_apply_chsh_variant(Eboot, variant))

    return _nanstd(samples)


# 関数: `_bootstrap_ch_j_sigma` の入出力契約と処理意図を定義する。

def _bootstrap_ch_j_sigma(
    *,
    n_trials: np.ndarray,
    n_coinc: np.ndarray,
    n_trials_a: np.ndarray,
    n_trials_b: np.ndarray,
    n_click_a: np.ndarray,
    n_click_b: np.ndarray,
    a1: int,
    b1: int,
    n_boot: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n_trials = np.asarray(n_trials, dtype=np.int64)
    n_coinc = np.asarray(n_coinc, dtype=np.int64)
    n_trials_a = np.asarray(n_trials_a, dtype=np.int64)
    n_trials_b = np.asarray(n_trials_b, dtype=np.int64)
    n_click_a = np.asarray(n_click_a, dtype=np.int64)
    n_click_b = np.asarray(n_click_b, dtype=np.int64)

    a1 = int(a1)
    b1 = int(b1)
    a2 = 1 - a1
    b2 = 1 - b1

    # 関数: `_binom` の入出力契約と処理意図を定義する。
    def _binom(n: int, k: int) -> int:
        # 条件分岐: `n <= 0` を満たす経路を評価する。
        if n <= 0:
            return 0

        p = k / n
        p = min(1.0, max(0.0, float(p)))
        return int(rng.binomial(int(n), p))

    samples: list[float] = []
    for _ in range(int(n_boot)):
        c = np.zeros((2, 2), dtype=np.int64)
        for a in (0, 1):
            for b in (0, 1):
                c[a, b] = _binom(int(n_trials[a, b]), int(n_coinc[a, b]))

        ca1 = _binom(int(n_trials_a[a1]), int(n_click_a[a1]))
        cb1 = _binom(int(n_trials_b[b1]), int(n_click_b[b1]))

        p_a1 = ca1 / max(1, int(n_trials_a[a1]))
        p_b1 = cb1 / max(1, int(n_trials_b[b1]))
        j = (
            c[a1, b1] / max(1, int(n_trials[a1, b1]))
            + c[a1, b2] / max(1, int(n_trials[a1, b2]))
            + c[a2, b1] / max(1, int(n_trials[a2, b1]))
            - c[a2, b2] / max(1, int(n_trials[a2, b2]))
            - p_a1
            - p_b1
        )
        # 条件分岐: `math.isfinite(float(j))` を満たす経路を評価する。
        if math.isfinite(float(j)):
            samples.append(float(j))

    return _nanstd(samples)


# 関数: `_load_csv_dicts` の入出力契約と処理意図を定義する。

def _load_csv_dicts(path: Path) -> list[dict[str, str]]:
    import csv

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))

    return rows


# 関数: `_parse_float` の入出力契約と処理意図を定義する。

def _parse_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


# 関数: `_parse_int` の入出力契約と処理意図を定義する。

def _parse_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


# 関数: `_weihs1998_dataset` の入出力契約と処理意図を定義する。

def _weihs1998_dataset(*, dataset_id: str, overwrite: bool) -> dict[str, Any]:
    mod = _load_script_module(rel_path="scripts/quantum/weihs1998_time_tag_reanalysis.py", name="_weihs1998")

    out_dir = OUT_BASE / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = ROOT / "data" / "quantum" / "sources" / "zenodo_7185335"
    subdir = "longdist"
    run = "longdist1"
    encoding: Literal["bit0-setting", "bit0-detector"] = "bit0-setting"

    # Extended to include the "natural window" range derived from dt peak width (see _recommend_time_tag_window_from_dt_peak).
    windows_ns = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    ref_window_ns = 1.0
    accidental_shift_ns = 100_000.0  # time-shift (ns) for accidental coincidence estimation
    accidental_shift_s = float(accidental_shift_ns) * 1e-9

    # --- normalized events
    npz_path = out_dir / "normalized_events.npz"
    meta_path = out_dir / "normalized_events.json"
    # 条件分岐: `overwrite or (not npz_path.exists()) or (not meta_path.exists())` を満たす経路を評価する。
    if overwrite or (not npz_path.exists()) or (not meta_path.exists()):
        arrays = mod._load_run_arrays(src_dir=src_dir, subdir=subdir, run=run)
        t_a = arrays["a_t"]
        c_a = arrays["a_c"]
        t_b = arrays["b_t"]
        c_b = arrays["b_c"]

        offset_info = mod._estimate_offset_s(t_a, t_b)

        np.savez_compressed(
            npz_path,
            a_t_s=t_a.astype(np.float64, copy=False),
            a_c=c_a.astype(np.uint16, copy=False),
            b_t_s=t_b.astype(np.float64, copy=False),
            b_c=c_b.astype(np.uint16, copy=False),
        )
        _write_json(
            meta_path,
            {
                "generated_utc": _utc_now(),
                "dataset_id": dataset_id,
                "source": {
                    "name": "Weihs et al. 1998 time-tag (Zenodo 7185335)",
                    "path": str(src_dir),
                    "alice_zip": str(src_dir / "Alice.zip"),
                    "bob_zip": str(src_dir / "Bob.zip"),
                },
                "schema": {
                    "a_t_s": "float64 seconds (event time; Alice)",
                    "a_c": "uint16 code in {0,1,2,3} (setting/outcome encoding; see encoding)",
                    "b_t_s": "float64 seconds (event time; Bob)",
                    "b_c": "uint16 code in {0,1,2,3} (setting/outcome encoding; see encoding)",
                    "encoding": encoding,
                },
                "counts": {"alice_events": int(t_a.size), "bob_events": int(t_b.size)},
                "offset_estimate": offset_info,
            },
        )

    data = np.load(npz_path)
    t_a = data["a_t_s"]
    c_a = data["a_c"]
    t_b = data["b_t_s"]
    c_b = data["b_c"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    offset_s = float((meta.get("offset_estimate") or {}).get("offset_s"))

    # --- fixed variant (derived at ref window)
    n_ref, sum_prod_ref, _pairs_ref = mod._pair_and_accumulate(
        t_a,
        c_a,
        t_b,
        c_b,
        offset_s=offset_s,
        window_s=float(ref_window_ns) * 1e-9,
        encoding=encoding,
    )
    E_ref = mod._safe_div(sum_prod_ref, n_ref)
    variant_fixed, s_ref_best = _best_chsh_variant(E_ref)

    # --- window sweep
    sweep_rows: list[dict[str, Any]] = []
    for wi, w_ns in enumerate(windows_ns):
        n, sum_prod, pairs_total = mod._pair_and_accumulate(
            t_a,
            c_a,
            t_b,
            c_b,
            offset_s=offset_s,
            window_s=float(w_ns) * 1e-9,
            encoding=encoding,
        )
        E = mod._safe_div(sum_prod, n)
        S = _apply_chsh_variant(E, variant_fixed) if np.isfinite(E).all() else float("nan")
        S_sigma = _bootstrap_chsh_s_sigma(E=E, n=n, variant=variant_fixed, n_boot=3000, seed=123)

        # Accidental estimate by a fixed time-shift (no tuning; same greedy pairing).
        n_acc, sum_prod_acc, pairs_acc = mod._pair_and_accumulate(
            t_a,
            c_a,
            t_b,
            c_b,
            offset_s=float(offset_s + accidental_shift_s),
            window_s=float(w_ns) * 1e-9,
            encoding=encoding,
        )
        E_acc = mod._safe_div(sum_prod_acc, n_acc)
        S_acc = _apply_chsh_variant(E_acc, variant_fixed) if np.isfinite(E_acc).all() else float("nan")

        n_sub = n - n_acc
        sum_prod_sub = sum_prod - sum_prod_acc
        E_sub = mod._safe_div(sum_prod_sub, n_sub)
        S_sub = _apply_chsh_variant(E_sub, variant_fixed) if np.isfinite(E_sub).all() else float("nan")
        S_sub_sigma = _bootstrap_chsh_s_sigma(E=E_sub, n=n_sub, variant=variant_fixed, n_boot=2000, seed=200 + wi)

        sweep_rows.append(
            {
                "window_ns": float(w_ns),
                "pairs_total": int(pairs_total),
                "n_by_setting": n.astype(int).tolist(),
                "S_fixed": float(S),
                "S_fixed_abs": float(abs(S)) if math.isfinite(float(S)) else None,
                "S_fixed_sigma_boot": float(S_sigma),
                "pairs_total_accidental": int(pairs_acc),
                "pairs_total_subtracted": int(pairs_total - pairs_acc),
                "n_by_setting_accidental": n_acc.astype(int).tolist(),
                "n_by_setting_subtracted": n_sub.astype(int).tolist(),
                "S_fixed_accidental": float(S_acc),
                "S_fixed_accidental_abs": float(abs(S_acc)) if math.isfinite(float(S_acc)) else None,
                "S_fixed_accidental_subtracted": float(S_sub),
                "S_fixed_accidental_subtracted_abs": float(abs(S_sub)) if math.isfinite(float(S_sub)) else None,
                "S_fixed_accidental_subtracted_sigma_boot": float(S_sub_sigma),
            }
        )

    s_abs = [r.get("S_fixed_abs") for r in sweep_rows]
    s_min, s_max = _min_max(s_abs)
    delta_s = (float(s_max) - float(s_min)) if (s_min is not None and s_max is not None) else None

    s_abs_sub = [r.get("S_fixed_accidental_subtracted_abs") for r in sweep_rows]
    s_sub_min, s_sub_max = _min_max(s_abs_sub)
    delta_s_sub = (float(s_sub_max) - float(s_sub_min)) if (s_sub_min is not None and s_sub_max is not None) else None

    _write_json(
        out_dir / "window_sweep_metrics.json",
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "statistic": {"name": "CHSH |S| (fixed variant)", "local_bound": 2.0},
            "config": {
                "windows_ns": windows_ns,
                "ref_window_ns": ref_window_ns,
                "encoding": encoding,
                "pairing": "greedy 1-1 pairing (time-ordered)",
                "bootstrap": {"method": "parametric (binomial on product)", "n_boot": 3000, "seed": 123},
                "accidental_estimation": {
                    "method": "time_shift",
                    "shift_ns": float(accidental_shift_ns),
                    "note": "Estimate accidentals by shifting the relative alignment by a fixed amount and re-running the same pairing.",
                },
            },
            "offset_ns_used": float(offset_s * 1e9),
            "fixed_variant": {
                "swap_a": bool(variant_fixed.swap_a),
                "swap_b": bool(variant_fixed.swap_b),
                "sign_matrix": variant_fixed.sign_matrix,
                "S_ref_best": float(s_ref_best),
            },
            "rows": sweep_rows,
            "summary": {
                "S_abs_min": s_min,
                "S_abs_max": s_max,
                "delta_S_abs": delta_s,
                "S_abs_accidental_subtracted_min": s_sub_min,
                "S_abs_accidental_subtracted_max": s_sub_max,
                "delta_S_abs_accidental_subtracted": delta_s_sub,
            },
        },
    )

    # --- offset sweep (ref window; small grid)
    off_half_ns = 20.0
    off_step_ns = 2.0
    offsets_ns = np.arange(-off_half_ns, off_half_ns + 0.5 * off_step_ns, off_step_ns, dtype=float)
    offset_rows: list[dict[str, Any]] = []
    for d_ns in offsets_ns.tolist():
        off_s = offset_s + float(d_ns) * 1e-9
        n, sum_prod, pairs_total = mod._pair_and_accumulate(
            t_a,
            c_a,
            t_b,
            c_b,
            offset_s=off_s,
            window_s=float(ref_window_ns) * 1e-9,
            encoding=encoding,
        )
        E = mod._safe_div(sum_prod, n)
        S = _apply_chsh_variant(E, variant_fixed) if np.isfinite(E).all() else float("nan")
        offset_rows.append(
            {
                "delta_offset_ns": float(d_ns),
                "offset_ns": float(off_s * 1e9),
                "pairs_total": int(pairs_total),
                "S_fixed": float(S),
                "S_fixed_abs": float(abs(S)) if math.isfinite(float(S)) else None,
            }
        )

    s_abs2 = [r.get("S_fixed_abs") for r in offset_rows]
    s2_min, s2_max = _min_max(s_abs2)
    delta_s2 = (float(s2_max) - float(s2_min)) if (s2_min is not None and s2_max is not None) else None

    _write_json(
        out_dir / "offset_sweep_metrics.json",
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "statistic": {"name": "CHSH |S| (fixed variant)", "local_bound": 2.0},
            "config": {
                "ref_window_ns": ref_window_ns,
                "delta_offset_ns_range": [-off_half_ns, off_half_ns],
                "delta_offset_ns_step": off_step_ns,
            },
            "rows": offset_rows,
            "summary": {"S_abs_min": s2_min, "S_abs_max": s2_max, "delta_S_abs": delta_s2},
        },
    )

    # --- delay setting dependence (KS) at ref window
    dt_by_ab: dict[tuple[int, int], list[float]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    i = 0
    j = 0
    na = int(t_a.size)
    nb = int(t_b.size)
    w_s = float(ref_window_ns) * 1e-9
    while i < na and j < nb:
        dt = float(t_b[j] - t_a[i] - offset_s)
        # 条件分岐: `dt < -w_s` を満たす経路を評価する。
        if dt < -w_s:
            j += 1
            continue

        # 条件分岐: `dt > w_s` を満たす経路を評価する。

        if dt > w_s:
            i += 1
            continue

        a_set, _ = mod._extract_setting_and_outcome(int(c_a[i]), encoding=encoding)
        b_set, _ = mod._extract_setting_and_outcome(int(c_b[j]), encoding=encoding)
        dt_by_ab[(int(a_set), int(b_set))].append(dt * 1e9)
        i += 1
        j += 1

    ks_a = max(
        _ks_distance(np.asarray(dt_by_ab[(0, b)], dtype=float), np.asarray(dt_by_ab[(1, b)], dtype=float))
        for b in (0, 1)
    )
    ks_b = max(
        _ks_distance(np.asarray(dt_by_ab[(a, 0)], dtype=float), np.asarray(dt_by_ab[(a, 1)], dtype=float))
        for a in (0, 1)
    )

    eps_sig = 0.1
    # Delay signature: represent "setting-dependent delay distribution" as Δmedian(ns) with an approximate z-score.
    # For Weihs, dt depends on (a,b), so compare conditional distributions and pick the most significant (max |z|).
    sig_a_best: dict[str, Any] | None = None
    sig_a_fixed_b: int | None = None
    for b in (0, 1):
        sig = _delay_signature_delta_median(dt_by_ab[(0, b)], dt_by_ab[(1, b)], epsilon=eps_sig)
        z = sig.get("z_delta_median")
        # 条件分岐: `not isinstance(z, (int, float)) or not math.isfinite(float(z))` を満たす経路を評価する。
        if not isinstance(z, (int, float)) or not math.isfinite(float(z)):
            continue

        # 条件分岐: `sig_a_best is None or abs(float(z)) > abs(float(sig_a_best.get("z_delta_media...` を満たす経路を評価する。

        if sig_a_best is None or abs(float(z)) > abs(float(sig_a_best.get("z_delta_median") or 0.0)):
            sig_a_best = sig
            sig_a_fixed_b = int(b)

    sig_b_best: dict[str, Any] | None = None
    sig_b_fixed_a: int | None = None
    for a in (0, 1):
        sig = _delay_signature_delta_median(dt_by_ab[(a, 0)], dt_by_ab[(a, 1)], epsilon=eps_sig)
        z = sig.get("z_delta_median")
        # 条件分岐: `not isinstance(z, (int, float)) or not math.isfinite(float(z))` を満たす経路を評価する。
        if not isinstance(z, (int, float)) or not math.isfinite(float(z)):
            continue

        # 条件分岐: `sig_b_best is None or abs(float(z)) > abs(float(sig_b_best.get("z_delta_media...` を満たす経路を評価する。

        if sig_b_best is None or abs(float(z)) > abs(float(sig_b_best.get("z_delta_median") or 0.0)):
            sig_b_best = sig
            sig_b_fixed_a = int(a)

    delay_signature = {
        "method": {
            "name": "delta_median",
            "epsilon": float(eps_sig),
            "definition": "dt = t_b - t_a - offset_s, paired within ref_window; compare conditional dt distributions by local setting.",
        },
        "Alice": (None if sig_a_best is None else {"fixed_b": sig_a_fixed_b, **sig_a_best}),
        "Bob": (None if sig_b_best is None else {"fixed_a": sig_b_fixed_a, **sig_b_best}),
    }

    # --- summary plot
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib not available: {e}")
    else:
        xs = [float(r["window_ns"]) for r in sweep_rows]
        ys = [float(r["S_fixed_abs"]) if r.get("S_fixed_abs") is not None else float("nan") for r in sweep_rows]
        ys_sub = [
            float(r["S_fixed_accidental_subtracted_abs"])
            if r.get("S_fixed_accidental_subtracted_abs") is not None
            else float("nan")
            for r in sweep_rows
        ]
        ps = [float(r["pairs_total"]) for r in sweep_rows]

        fig, ax = plt.subplots(2, 2, figsize=(12.8, 8.0), dpi=170)

        ax00 = ax[0, 0]
        ax00.plot(xs, ys, marker="o", lw=1.8, label="raw")
        # 条件分岐: `any(math.isfinite(v) for v in ys_sub)` を満たす経路を評価する。
        if any(math.isfinite(v) for v in ys_sub):
            ax00.plot(xs, ys_sub, marker="s", lw=1.6, color="tab:orange", label="accidental-subtracted")

        ax00.axhline(2.0, color="0.2", ls="--", lw=1.0)
        ax00.set_xscale("log")
        ax00.set_xlabel("window (ns)")
        ax00.set_ylabel("|S| (fixed)")
        ax00.set_title("Window sweep")
        ax00.grid(True, which="both", alpha=0.3, ls=":")
        ax00.legend(fontsize=9, frameon=True)

        ax01 = ax[0, 1]
        ax01.plot(xs, ps, marker="o", lw=1.8, color="tab:blue")
        ax01.set_xscale("log")
        ax01.set_xlabel("window (ns)")
        ax01.set_ylabel("pairs")
        ax01.set_title("Pairs vs window")
        ax01.grid(True, which="both", alpha=0.3, ls=":")

        ax10 = ax[1, 0]
        offx = [float(r["delta_offset_ns"]) for r in offset_rows]
        offy = [float(r["S_fixed_abs"]) if r.get("S_fixed_abs") is not None else float("nan") for r in offset_rows]
        ax10.plot(offx, offy, marker="o", lw=1.6)
        ax10.axhline(2.0, color="0.2", ls="--", lw=1.0)
        ax10.set_xlabel("delta offset (ns)")
        ax10.set_ylabel("|S| (fixed; ref window)")
        ax10.set_title("Offset sweep (ref window)")
        ax10.grid(True, alpha=0.3, ls=":")

        ax11 = ax[1, 1]
        dt00 = np.asarray(dt_by_ab[(0, 0)], dtype=float)
        dt11 = np.asarray(dt_by_ab[(1, 1)], dtype=float)
        # 条件分岐: `dt00.size and dt11.size` を満たす経路を評価する。
        if dt00.size and dt11.size:
            allv = np.concatenate([dt00, dt11])
            hi = float(np.percentile(allv, 99.5))
            lo = float(np.percentile(allv, 0.5))
            rng = max(5.0, min(hi - lo, 200.0))
            mid = 0.5 * (hi + lo)
            bins = np.linspace(mid - 0.5 * rng, mid + 0.5 * rng, 120)
            ax11.hist(dt00, bins=bins, alpha=0.55, label="dt (a=0,b=0)")
            ax11.hist(dt11, bins=bins, alpha=0.55, label="dt (a=1,b=1)")
            ax11.set_xlabel("dt (ns)")
            ax11.set_ylabel("count")
            ax11.set_title(f"Delay (KS_A={ks_a:.3f}, KS_B={ks_b:.3f})")
            ax11.legend(fontsize=8, frameon=True)
            ax11.grid(True, alpha=0.3, ls=":")
        else:
            ax11.axis("off")
            ax11.text(0.1, 0.5, "dt histogram: n/a", fontsize=10)

        fig.suptitle(f"Bell (Weihs1998): {dataset_id}", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / "summary.png", bbox_inches="tight")
        plt.close(fig)

    return {
        "dataset_id": dataset_id,
        "statistic": "CHSH|S|",
        "window_delta": delta_s,
        "window_delta_accidental_subtracted": delta_s_sub,
        "offset_delta": delta_s2,
        "ks_delay": {"A": float(ks_a), "B": float(ks_b)},
        "delay_signature": delay_signature,
        "baseline": {
            "ref_window_ns": ref_window_ns,
            "offset_ns": float(offset_s * 1e9),
            "S_ref": float(_apply_chsh_variant(E_ref, variant_fixed)) if np.isfinite(E_ref).all() else float("nan"),
        },
    }


# 関数: `_delft_datasets` の入出力契約と処理意図を定義する。

def _delft_datasets(*, overwrite: bool) -> list[dict[str, Any]]:
    mod = _load_script_module(rel_path="scripts/quantum/delft_hensen2015_chsh_reanalysis.py", name="_delft")
    results: list[dict[str, Any]] = []

    # 関数: `_write_trial_only_window_unsupported` の入出力契約と処理意図を定義する。
    def _write_trial_only_window_unsupported(dataset_id: str) -> None:
        _write_json(
            OUT_BASE / dataset_id / "window_sweep_metrics.json",
            {
                "generated_utc": _utc_now(),
                "dataset_id": dataset_id,
                "supported": False,
                "reason": "trial log (event-ready); no coincidence-window pairing step",
            },
        )

    # --- 2015

    dataset_id = "delft_hensen2015"
    out_dir = OUT_BASE / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ROOT / "data" / "quantum" / "sources" / dataset_id / "data.zip"
    member = "bell_open_data.txt"
    _ts, data = mod._load_table_from_zip(zip_path, member=member)
    p = mod.Params()

    baseline = mod._analyze(data, p=p, start_offset_ps=0)
    offsets_ps = np.linspace(p.start_offset_min_ps, p.start_offset_max_ps, p.start_offset_points).astype(int)
    offset_rows: list[dict[str, Any]] = []
    for off in offsets_ps.tolist():
        res = mod._analyze(data, p=p, start_offset_ps=int(off))
        offset_rows.append({"start_offset_ps": int(off), "S": float(res.s), "S_err": float(res.s_err)})

    sweep_s = [float(r.get("S")) for r in offset_rows]
    s_min, s_max = _min_max(sweep_s)
    delta_s = (float(s_max) - float(s_min)) if (s_min is not None and s_max is not None) else None

    # normalized events (minimal view)
    norm_json = out_dir / "normalized_events.json"
    # 条件分岐: `overwrite or (not norm_json.exists())` を満たす経路を評価する。
    if overwrite or (not norm_json.exists()):
        random_number_a = data[:, 6].astype(np.int8, copy=False)
        random_number_b = data[:, 7].astype(np.int8, copy=False)
        pair_counts = [
            [int(np.sum((random_number_a == 0) & (random_number_b == 0))), int(np.sum((random_number_a == 0) & (random_number_b == 1)))],
            [int(np.sum((random_number_a == 1) & (random_number_b == 0))), int(np.sum((random_number_a == 1) & (random_number_b == 1)))],
        ]
        _write_json(
            norm_json,
            {
                "generated_utc": _utc_now(),
                "dataset_id": dataset_id,
                "source": {"name": "Hensen et al. 2015 (Delft; event-ready trial log)", "zip": str(zip_path)},
                "members": {"trials": member},
                "schema": {"a_setting": "0/1", "b_setting": "0/1", "a_outcome": "±1", "b_outcome": "±1"},
                "counts": {"trials_total": int(data.shape[0])},
                "preview": {
                    "a_setting_counts": [int(np.sum(random_number_a == 0)), int(np.sum(random_number_a == 1))],
                    "b_setting_counts": [int(np.sum(random_number_b == 0)), int(np.sum(random_number_b == 1))],
                    "pair_setting_counts": pair_counts,
                },
            },
        )

    _write_trial_only_window_unsupported(dataset_id)
    _write_json(
        out_dir / "offset_sweep_metrics.json",
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "supported": True,
            "statistic": {"name": "CHSH S (event-ready trials)", "local_bound": 2.0},
            "baseline": {
                "start_offset_ps": 0,
                "n_trials": int(baseline.n_trials),
                "S": float(baseline.s),
                "S_err": float(baseline.s_err),
                "p_value_bound": float(baseline.p_value),
            },
            "config": {
                "offset_param": "event_ready_window_start (ps)",
                "range_ps": [int(p.start_offset_min_ps), int(p.start_offset_max_ps)],
                "points": int(p.start_offset_points),
            },
            "rows": offset_rows,
            "summary": {"S_min": s_min, "S_max": s_max, "delta_S": delta_s},
        },
    )

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib not available: {e}")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9.6, 4.8), dpi=170)
        ax.plot(offsets_ps.astype(float) / 1000.0, sweep_s, lw=1.8)
        ax.axhline(2.0, color="0.2", ls="--", lw=1.0)
        ax.axhline(float(baseline.s), color="tab:orange", lw=1.0, ls=":")
        ax.set_xlabel("event-ready start offset (ns)")
        ax.set_ylabel("CHSH S")
        ax.set_title(f"Bell (Delft 2015): S={baseline.s:.3f}±{baseline.s_err:.3f}")
        ax.grid(True, alpha=0.3, ls=":")
        fig.tight_layout()
        fig.savefig(out_dir / "summary.png", bbox_inches="tight")
        plt.close(fig)

    results.append(
        {
            "dataset_id": dataset_id,
            "statistic": "CHSH_S",
            "window_delta": None,
            "offset_delta": delta_s,
            "ks_delay": None,
            "baseline": {"S": float(baseline.s), "S_err": float(baseline.s_err), "n_trials": int(baseline.n_trials)},
        }
    )

    # --- 2016
    dataset_id = "delft_hensen2016_srep30289"
    out_dir = OUT_BASE / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ROOT / "data" / "quantum" / "sources" / dataset_id / "data.zip"
    old_member = "bell_open_data_2_old_detector.txt"
    new_member = "bell_open_data_2_new_detector.txt"
    _ts_old, data_old = mod._load_table_from_zip(zip_path, member=old_member)
    _ts_new, data_new = mod._load_table_from_zip(zip_path, member=new_member)

    p2 = mod.Hensen2016Params()
    baseline2 = mod._analyze_hensen2016(data_old, data_new, p=p2, start_offset_ps=0)
    offsets_ps2 = np.linspace(p2.start_offset_min_ps, p2.start_offset_max_ps, p2.start_offset_points).astype(int)
    offset_rows2: list[dict[str, Any]] = []
    for off in offsets_ps2.tolist():
        res = mod._analyze_hensen2016(data_old, data_new, p=p2, start_offset_ps=int(off))
        offset_rows2.append(
            {
                "start_offset_ps": int(off),
                "S_combined": float(res.s_combined),
                "S_combined_err": float(res.s_err_combined),
            }
        )

    sweep_s2 = [float(r.get("S_combined")) for r in offset_rows2]
    s2_min, s2_max = _min_max(sweep_s2)
    delta_s2 = (float(s2_max) - float(s2_min)) if (s2_min is not None and s2_max is not None) else None

    norm_json = out_dir / "normalized_events.json"
    # 条件分岐: `overwrite or (not norm_json.exists())` を満たす経路を評価する。
    if overwrite or (not norm_json.exists()):
        random_a_old = data_old[:, 6].astype(np.int8, copy=False)
        random_b_old = data_old[:, 7].astype(np.int8, copy=False)
        random_a_new = data_new[:, 6].astype(np.int8, copy=False)
        random_b_new = data_new[:, 7].astype(np.int8, copy=False)
        random_a_all = np.concatenate([random_a_old, random_a_new], axis=0)
        random_b_all = np.concatenate([random_b_old, random_b_new], axis=0)
        pair_counts_all = [
            [int(np.sum((random_a_all == 0) & (random_b_all == 0))), int(np.sum((random_a_all == 0) & (random_b_all == 1)))],
            [int(np.sum((random_a_all == 1) & (random_b_all == 0))), int(np.sum((random_a_all == 1) & (random_b_all == 1)))],
        ]
        _write_json(
            norm_json,
            {
                "generated_utc": _utc_now(),
                "dataset_id": dataset_id,
                "source": {"name": "Hensen et al. 2016 (Sci Rep 6, 30289; event-ready trial log)", "zip": str(zip_path)},
                "members": {"old_detector": old_member, "new_detector": new_member},
                "counts": {
                    "trials_old": int(data_old.shape[0]),
                    "trials_new": int(data_new.shape[0]),
                    "trials_total": int(data_old.shape[0] + data_new.shape[0]),
                },
                "preview": {
                    "a_setting_counts": [int(np.sum(random_a_all == 0)), int(np.sum(random_a_all == 1))],
                    "b_setting_counts": [int(np.sum(random_b_all == 0)), int(np.sum(random_b_all == 1))],
                    "pair_setting_counts": pair_counts_all,
                },
            },
        )

    _write_trial_only_window_unsupported(dataset_id)
    _write_json(
        out_dir / "offset_sweep_metrics.json",
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "supported": True,
            "statistic": {"name": "CHSH S (combined; event-ready trials)", "local_bound": 2.0},
            "baseline": {
                "start_offset_ps": 0,
                "n_trials_total": int(baseline2.n_trials_total),
                "S_combined": float(baseline2.s_combined),
                "S_combined_err": float(baseline2.s_err_combined),
            },
            "config": {
                "offset_param": "event_ready_window_start (ps)",
                "range_ps": [int(p2.start_offset_min_ps), int(p2.start_offset_max_ps)],
                "points": int(p2.start_offset_points),
            },
            "rows": offset_rows2,
            "summary": {"S_min": s2_min, "S_max": s2_max, "delta_S": delta_s2},
        },
    )

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib not available: {e}")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9.6, 4.8), dpi=170)
        ax.plot(offsets_ps2.astype(float) / 1000.0, sweep_s2, lw=1.8)
        ax.axhline(2.0, color="0.2", ls="--", lw=1.0)
        ax.axhline(float(baseline2.s_combined), color="tab:orange", lw=1.0, ls=":")
        ax.set_xlabel("event-ready start offset (ns)")
        ax.set_ylabel("CHSH S (combined)")
        ax.set_title(f"Bell (Delft 2016): S={baseline2.s_combined:.3f}±{baseline2.s_err_combined:.3f}")
        ax.grid(True, alpha=0.3, ls=":")
        fig.tight_layout()
        fig.savefig(out_dir / "summary.png", bbox_inches="tight")
        plt.close(fig)

    results.append(
        {
            "dataset_id": dataset_id,
            "statistic": "CHSH_S",
            "window_delta": None,
            "offset_delta": delta_s2,
            "ks_delay": None,
            "baseline": {
                "S_combined": float(baseline2.s_combined),
                "S_combined_err": float(baseline2.s_err_combined),
                "n_trials_total": int(baseline2.n_trials_total),
            },
        }
    )

    return results


# 関数: `_nist_dataset` の入出力契約と処理意図を定義する。

def _nist_dataset(*, overwrite: bool) -> dict[str, Any]:
    dataset_id = "nist_03_43_afterfixingModeLocking_s3600"
    out_dir = OUT_BASE / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    nist_tt = _load_script_module(rel_path="scripts/quantum/nist_belltest_time_tag_reanalysis.py", name="_nist_tt")
    nist_trial = _load_script_module(rel_path="scripts/quantum/nist_belltest_trial_based_reanalysis.py", name="_nist_trial")

    src_dir = ROOT / "data" / "quantum" / "sources" / "nist_belltestdata"
    alice_zip = (
        src_dir
        / "compressed"
        / "alice"
        / "2015_09_18"
        / "03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking.alice.dat.compressed.zip"
    )
    bob_zip = (
        src_dir
        / "compressed"
        / "bob"
        / "2015_09_18"
        / "03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking.bob.dat.compressed.zip"
    )
    hdf5_path = (
        src_dir
        / "processed_compressed"
        / "hdf5"
        / "2015_09_18"
        / "03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking.dat.compressed.build.hdf5"
    )
    max_seconds = 3600
    accidental_shift_ns = 100_000.0  # time-shift (ns) for accidental coincidence estimation

    # --- trial-based denominators (sync×slot)
    trial_counts_path = out_dir / "trial_based_counts.json"
    # 条件分岐: `overwrite or (not trial_counts_path.exists())` を満たす経路を評価する。
    if overwrite or (not trial_counts_path.exists()):
        try:
            import h5py  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "[fail] missing dependency: h5py\nInstall with: python -m pip install -U h5py\n" f"Import error: {e}"
            )

        with h5py.File(hdf5_path, "r") as f:
            a_setting = f["alice/settings"]
            b_setting = f["bob/settings"]
            a_clicks = f["alice/clicks"]
            b_clicks = f["bob/clicks"]
            n = int(a_clicks.shape[0])
            n_use = int(n)

            bad: set[int] = set()
            for side in ("alice", "bob"):
                g = f[side]
                # 条件分岐: `"badSyncInfo" not in g` を満たす経路を評価する。
                if "badSyncInfo" not in g:
                    continue

                flat = np.asarray(g["badSyncInfo"][()]).reshape(-1)
                bad |= set(map(int, flat.tolist()))

            bad_idx = np.asarray(sorted(i for i in bad if 0 <= i < n_use), dtype=np.int64)

            counts_obj = nist_trial._compute_trial_counts(
                a_setting_raw=a_setting[:n_use],
                b_setting_raw=b_setting[:n_use],
                a_clicks=a_clicks[:n_use],
                b_clicks=b_clicks[:n_use],
                bad_sync_idx=bad_idx,
            )

        ch = nist_trial._ch_j_variants(counts_obj)
        _write_json(
            trial_counts_path,
            {
                "generated_utc": _utc_now(),
                "dataset_id": dataset_id,
                "source": {"hdf5": str(hdf5_path)},
                "counts": {
                    "syncs_used": int(counts_obj.n_total),
                    "bad_sync_unique": int(counts_obj.n_bad_sync),
                    "invalid_settings": int(counts_obj.n_invalid_settings),
                    "trials_by_setting_pair": counts_obj.n_trials.astype(int).tolist(),
                    "coinc_by_setting_pair": counts_obj.n_coinc.astype(int).tolist(),
                    "alice_trials_by_setting": counts_obj.n_trials_a.astype(int).tolist(),
                    "bob_trials_by_setting": counts_obj.n_trials_b.astype(int).tolist(),
                    "alice_clicks_by_setting": counts_obj.n_click_a.astype(int).tolist(),
                    "bob_clicks_by_setting": counts_obj.n_click_b.astype(int).tolist(),
                },
                "ch_j_variants": ch,
            },
        )

    trial = json.loads(trial_counts_path.read_text(encoding="utf-8"))
    counts = trial["counts"]
    n_trials = np.asarray(counts["trials_by_setting_pair"], dtype=np.int64)
    n_coinc_trial = np.asarray(counts["coinc_by_setting_pair"], dtype=np.int64)
    n_trials_a = np.asarray(counts["alice_trials_by_setting"], dtype=np.int64)
    n_trials_b = np.asarray(counts["bob_trials_by_setting"], dtype=np.int64)
    n_click_a = np.asarray(counts["alice_clicks_by_setting"], dtype=np.int64)
    n_click_b = np.asarray(counts["bob_clicks_by_setting"], dtype=np.int64)

    a1 = 0
    b1 = 0
    a2 = 1
    b2 = 1
    p_a1 = float(n_click_a[a1] / max(1, int(n_trials_a[a1])))
    p_b1 = float(n_click_b[b1] / max(1, int(n_trials_b[b1])))
    j_trial = (
        n_coinc_trial[a1, b1] / max(1, int(n_trials[a1, b1]))
        + n_coinc_trial[a1, b2] / max(1, int(n_trials[a1, b2]))
        + n_coinc_trial[a2, b1] / max(1, int(n_trials[a2, b1]))
        - n_coinc_trial[a2, b2] / max(1, int(n_trials[a2, b2]))
        - p_a1
        - p_b1
    )
    j_sigma_trial = _bootstrap_ch_j_sigma(
        n_trials=n_trials,
        n_coinc=n_coinc_trial,
        n_trials_a=n_trials_a,
        n_trials_b=n_trials_b,
        n_click_a=n_click_a,
        n_click_b=n_click_b,
        a1=a1,
        b1=b1,
        n_boot=5000,
        seed=7,
    )

    # --- time-tag: read first max_seconds for KS (setting dependence of click_delay)
    norm_npz = out_dir / "normalized_events.npz"
    norm_meta = out_dir / "normalized_events.json"
    # 条件分岐: `overwrite or (not norm_npz.exists()) or (not norm_meta.exists())` を満たす経路を評価する。
    if overwrite or (not norm_npz.exists()) or (not norm_meta.exists()):
        cfg = nist_tt.Config(max_seconds=max_seconds)
        a = nist_tt._read_side(alice_zip, max_seconds=cfg.max_seconds)
        b = nist_tt._read_side(bob_zip, max_seconds=cfg.max_seconds)
        align = nist_tt._estimate_pps_offset(a.pps_t, b.pps_t, max_shift=5)
        offset_counts = int(align["offset_counts"])

        np.savez_compressed(
            norm_npz,
            seconds_per_timetag=np.asarray([float(cfg.seconds_per_timetag)], dtype=np.float64),
            max_seconds=np.asarray([int(max_seconds)], dtype=np.int64),
            offset_counts=np.asarray([int(offset_counts)], dtype=np.int64),
            alice_click_t=a.click_t.astype(np.uint64, copy=False),
            alice_click_setting=a.click_setting.astype(np.uint8, copy=False),
            alice_click_delay=a.click_delay.astype(np.uint64, copy=False),
            alice_pps_t=a.pps_t.astype(np.uint64, copy=False),
            bob_click_t=(b.click_t + offset_counts).astype(np.uint64, copy=False),
            bob_click_setting=b.click_setting.astype(np.uint8, copy=False),
            bob_click_delay=b.click_delay.astype(np.uint64, copy=False),
            bob_pps_t=(b.pps_t + offset_counts).astype(np.uint64, copy=False),
        )
        _write_json(
            norm_meta,
            {
                "generated_utc": _utc_now(),
                "dataset_id": dataset_id,
                "source": {
                    "name": "NIST belltestdata (Shalm et al. 2015; time-tag repository)",
                    "alice_zip": str(alice_zip),
                    "bob_zip": str(bob_zip),
                    "hdf5_build": str(hdf5_path),
                },
                "config": {"max_seconds": int(max_seconds), "seconds_per_timetag": float(cfg.seconds_per_timetag)},
                "schema": {
                    "alice_click_t": "uint64 timetag counts (aligned to Alice PPS=0)",
                    "alice_click_setting": "uint8 {0,1}",
                    "alice_click_delay": "uint64 timetag counts since last sync",
                    "bob_click_t": "uint64 timetag counts (PPS-aligned; offset_counts applied)",
                    "bob_click_setting": "uint8 {0,1}",
                    "bob_click_delay": "uint64 timetag counts since last sync",
                    "offset_counts": "int64 counts added to Bob times to align PPS",
                },
                "notes": [
                    "normalized_events.npz is a compressed export of the time-tag clicks for max_seconds.",
                    "Trial-based denominators are computed from the published hdf5 build.",
                ],
            },
        )

    data = np.load(norm_npz)
    seconds_per_timetag = float(data["seconds_per_timetag"][0])
    a_delay_ns = data["alice_click_delay"].astype(np.float64) * seconds_per_timetag * 1e9
    b_delay_ns = data["bob_click_delay"].astype(np.float64) * seconds_per_timetag * 1e9
    a_set = data["alice_click_setting"].astype(np.int8)
    b_set = data["bob_click_setting"].astype(np.int8)
    a_t_full = data["alice_click_t"].astype(np.int64)
    b_t_full = data["bob_click_t"].astype(np.int64)
    a0 = a_delay_ns[a_set == 0]
    a1v = a_delay_ns[a_set == 1]
    b0 = b_delay_ns[b_set == 0]
    b1v = b_delay_ns[b_set == 1]
    ks_a = _ks_distance(a0, a1v)
    ks_b = _ks_distance(b0, b1v)

    eps_sig = 0.1
    delay_signature = {
        "method": {
            "name": "delta_median",
            "epsilon": float(eps_sig),
            "definition": "click_delay from sync (ns) by local setting; z is an approximate significance of Δmedian.",
        },
        "Alice": _delay_signature_delta_median(a0, a1v, epsilon=eps_sig),
        "Bob": _delay_signature_delta_median(b0, b1v, epsilon=eps_sig),
    }

    # --- window sweep: prefer existing CSV (fast); fallback to recompute from normalized clicks.
    sweep_csv = ROOT / "output" / "public" / "quantum" / "nist_belltest_coincidence_sweep__03_43_afterfixingModeLocking_s3600.csv"
    sweep_source: dict[str, Any] = {}
    # 条件分岐: `sweep_csv.exists()` を満たす経路を評価する。
    if sweep_csv.exists():
        rows = _load_csv_dicts(sweep_csv)
        windows_ns = [_parse_float(r, "window_ns") for r in rows]
        c00 = np.asarray([_parse_int(r, "c00") for r in rows], dtype=np.int64)
        c01 = np.asarray([_parse_int(r, "c01") for r in rows], dtype=np.int64)
        c10 = np.asarray([_parse_int(r, "c10") for r in rows], dtype=np.int64)
        c11 = np.asarray([_parse_int(r, "c11") for r in rows], dtype=np.int64)
        pairs_total = np.asarray([_parse_int(r, "pairs_total") for r in rows], dtype=np.int64)
        sweep_source = {"mode": "csv", "path": str(sweep_csv)}
    else:
        cfg_win = nist_tt.Config(max_seconds=max_seconds)
        windows_ns = list(map(float, cfg_win.windows_ns))
        windows_counts = np.asarray([w * 1e-9 / seconds_per_timetag for w in windows_ns], dtype=np.float64)
        counts_arr, pairs_arr = nist_tt._coincidence_sweep(a_t_full, a_set, b_t_full, b_set, windows_counts)
        c00 = counts_arr[:, 0, 0].astype(np.int64)
        c01 = counts_arr[:, 0, 1].astype(np.int64)
        c10 = counts_arr[:, 1, 0].astype(np.int64)
        c11 = counts_arr[:, 1, 1].astype(np.int64)
        pairs_total = pairs_arr.astype(np.int64)
        sweep_source = {"mode": "computed", "pairing": "greedy 1-1 pairing (time-ordered)"}

    windows_counts = np.asarray([w * 1e-9 / seconds_per_timetag for w in windows_ns], dtype=np.float64)
    shift_counts = int(round(float(accidental_shift_ns) * 1e-9 / float(seconds_per_timetag)))
    counts_acc_arr, pairs_acc_arr = nist_tt._coincidence_sweep(
        a_t_full,
        a_set,
        (b_t_full + shift_counts).astype(np.int64, copy=False),
        b_set,
        windows_counts,
    )
    c00_acc = counts_acc_arr[:, 0, 0].astype(np.int64)
    c01_acc = counts_acc_arr[:, 0, 1].astype(np.int64)
    c10_acc = counts_acc_arr[:, 1, 0].astype(np.int64)
    c11_acc = counts_acc_arr[:, 1, 1].astype(np.int64)
    pairs_total_acc = pairs_acc_arr.astype(np.int64)

    j_sweep = (
        c00 / np.maximum(1, n_trials[0, 0])
        + c01 / np.maximum(1, n_trials[0, 1])
        + c10 / np.maximum(1, n_trials[1, 0])
        - c11 / np.maximum(1, n_trials[1, 1])
        - p_a1
        - p_b1
    ).astype(np.float64)
    j_sweep_acc = (
        c00_acc / np.maximum(1, n_trials[0, 0])
        + c01_acc / np.maximum(1, n_trials[0, 1])
        + c10_acc / np.maximum(1, n_trials[1, 0])
        - c11_acc / np.maximum(1, n_trials[1, 1])
        - p_a1
        - p_b1
    ).astype(np.float64)
    c00_sub = (c00 - c00_acc).astype(np.int64)
    c01_sub = (c01 - c01_acc).astype(np.int64)
    c10_sub = (c10 - c10_acc).astype(np.int64)
    c11_sub = (c11 - c11_acc).astype(np.int64)
    c00_sub_clip = np.maximum(0, c00_sub).astype(np.int64)
    c01_sub_clip = np.maximum(0, c01_sub).astype(np.int64)
    c10_sub_clip = np.maximum(0, c10_sub).astype(np.int64)
    c11_sub_clip = np.maximum(0, c11_sub).astype(np.int64)
    j_sweep_sub = (
        c00_sub / np.maximum(1, n_trials[0, 0])
        + c01_sub / np.maximum(1, n_trials[0, 1])
        + c10_sub / np.maximum(1, n_trials[1, 0])
        - c11_sub / np.maximum(1, n_trials[1, 1])
        - p_a1
        - p_b1
    ).astype(np.float64)
    j_sweep_sub_clip = (
        c00_sub_clip / np.maximum(1, n_trials[0, 0])
        + c01_sub_clip / np.maximum(1, n_trials[0, 1])
        + c10_sub_clip / np.maximum(1, n_trials[1, 0])
        - c11_sub_clip / np.maximum(1, n_trials[1, 1])
        - p_a1
        - p_b1
    ).astype(np.float64)

    j_min, j_max = _min_max(j_sweep.tolist())
    delta_j = (float(j_max) - float(j_min)) if (j_min is not None and j_max is not None) else None

    j_sub_min, j_sub_max = _min_max(j_sweep_sub_clip.tolist())
    delta_j_sub = (float(j_sub_max) - float(j_sub_min)) if (j_sub_min is not None and j_sub_max is not None) else None

    j_sigma_by_window: list[float] = []
    for i in range(len(windows_ns)):
        n_coinc = np.array([[c00[i], c01[i]], [c10[i], c11[i]]], dtype=np.int64)
        j_sigma_by_window.append(
            _bootstrap_ch_j_sigma(
                n_trials=n_trials,
                n_coinc=n_coinc,
                n_trials_a=n_trials_a,
                n_trials_b=n_trials_b,
                n_click_a=n_click_a,
                n_click_b=n_click_b,
                a1=a1,
                b1=b1,
                n_boot=2000,
                seed=100 + i,
            )
        )

    rows_out: list[dict[str, Any]] = []
    for i, w in enumerate(windows_ns):
        rows_out.append(
            {
                "window_ns": float(w),
                "pairs_total": int(pairs_total[i]),
                "J_prob": float(j_sweep[i]),
                "J_sigma_boot": float(j_sigma_by_window[i]),
                "coinc_by_setting_pair": [
                    [int(c00[i]), int(c01[i])],
                    [int(c10[i]), int(c11[i])],
                ],
                "pairs_total_accidental": int(pairs_total_acc[i]),
                "J_prob_accidental": float(j_sweep_acc[i]),
                "coinc_by_setting_pair_accidental": [
                    [int(c00_acc[i]), int(c01_acc[i])],
                    [int(c10_acc[i]), int(c11_acc[i])],
                ],
                "pairs_total_subtracted": int(pairs_total[i] - pairs_total_acc[i]),
                "J_prob_subtracted": float(j_sweep_sub[i]),
                "J_prob_subtracted_clipped": float(j_sweep_sub_clip[i]),
                "coinc_by_setting_pair_subtracted": [
                    [int(c00_sub[i]), int(c01_sub[i])],
                    [int(c10_sub[i]), int(c11_sub[i])],
                ],
                "coinc_by_setting_pair_subtracted_clipped": [
                    [int(c00_sub_clip[i]), int(c01_sub_clip[i])],
                    [int(c10_sub_clip[i]), int(c11_sub_clip[i])],
                ],
            }
        )

    _write_json(
        out_dir / "window_sweep_metrics.json",
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "statistic": {"name": "CH J_prob (A1=0,B1=0)", "local_bound": 0.0},
            "trial_based": {"J_prob": float(j_trial), "J_sigma_boot": float(j_sigma_trial), "a1": a1, "b1": b1},
            "time_tag_delay_ks": {"alice": float(ks_a), "bob": float(ks_b)},
            "config": {
                "window_sweep_source": sweep_source,
                "accidental_estimation": {
                    "method": "time_shift",
                    "shift_ns": float(accidental_shift_ns),
                    "shift_counts": int(shift_counts),
                    "note": "Estimate accidentals by shifting Bob timestamps by a fixed amount and re-running the same greedy pairing.",
                },
                "subtraction": {
                    "method": "coinc_counts_minus_shifted_counts",
                    "clip_negative_to_zero": True,
                    "note": "Subtracted_clipped is the conservative (non-negative) version used for plots/summary; raw subtracted may go negative due to variance.",
                },
            },
            "rows": rows_out,
            "summary": {
                "J_min": j_min,
                "J_max": j_max,
                "delta_J": delta_j,
                "J_subtracted_clipped_min": j_sub_min,
                "J_subtracted_clipped_max": j_sub_max,
                "delta_J_subtracted_clipped": delta_j_sub,
            },
        },
    )

    # --- offset sweep: coincidence counts on a fixed window for a subset of events (cost control)
    win_ns = 200.0
    dt_step_ns = 5.0
    dt_half_ns = 50.0
    offsets_ns = np.arange(-dt_half_ns, dt_half_ns + 0.5 * dt_step_ns, dt_step_ns, dtype=float)
    max_events = 400_000

    a_t = data["alice_click_t"][:max_events].astype(np.int64)
    a_s = data["alice_click_setting"][:max_events].astype(np.int8)
    b_t0 = data["bob_click_t"][:max_events].astype(np.int64)
    b_s = data["bob_click_setting"][:max_events].astype(np.int8)

    w_counts = float(win_ns) * 1e-9 / seconds_per_timetag
    offset_rows: list[dict[str, Any]] = []
    for idx, d_ns in enumerate(offsets_ns.tolist()):
        delta_counts = int(round(float(d_ns) * 1e-9 / seconds_per_timetag))
        b_t = b_t0 + delta_counts
        counts2, pairs2 = nist_tt._coincidence_sweep(a_t, a_s, b_t, b_s, np.asarray([w_counts], dtype=np.float64))
        c = counts2[0]
        j = (
            c[0, 0] / max(1, int(n_trials[0, 0]))
            + c[0, 1] / max(1, int(n_trials[0, 1]))
            + c[1, 0] / max(1, int(n_trials[1, 0]))
            - c[1, 1] / max(1, int(n_trials[1, 1]))
            - p_a1
            - p_b1
        )
        j_sigma = _bootstrap_ch_j_sigma(
            n_trials=n_trials,
            n_coinc=c.astype(np.int64, copy=False),
            n_trials_a=n_trials_a,
            n_trials_b=n_trials_b,
            n_click_a=n_click_a,
            n_click_b=n_click_b,
            a1=a1,
            b1=b1,
            n_boot=1500,
            seed=500 + idx,
        )
        offset_rows.append(
            {
                "delta_offset_ns": float(d_ns),
                "delta_offset_counts": int(delta_counts),
                "pairs_total": int(pairs2[0]),
                "J_prob": float(j),
                "J_sigma_boot": float(j_sigma),
                "coinc_by_setting_pair": c.astype(int).tolist(),
            }
        )

    js = [r["J_prob"] for r in offset_rows]
    j2_min, j2_max = _min_max(js)
    delta_j2 = (float(j2_max) - float(j2_min)) if (j2_min is not None and j2_max is not None) else None

    _write_json(
        out_dir / "offset_sweep_metrics.json",
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "supported": True,
            "statistic": {"name": "CH J_prob (A1=0,B1=0)", "local_bound": 0.0},
            "config": {
                "window_ns": float(win_ns),
                "delta_offset_ns_range": [-dt_half_ns, dt_half_ns],
                "delta_offset_ns_step": dt_step_ns,
                "max_events_each_side": int(max_events),
                "note": "Offset sweep uses a prefix slice of the time-tag clicks for cost control.",
            },
            "rows": offset_rows,
            "summary": {"J_min": j2_min, "J_max": j2_max, "delta_J": delta_j2},
        },
    )

    # --- summary plot
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib not available: {e}")
    else:
        fig = plt.figure(figsize=(16.0, 4.8), dpi=170)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.2])

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(windows_ns, j_sweep, marker="o", lw=1.6, label="raw")
        # 条件分岐: `any(math.isfinite(float(v)) for v in j_sweep_sub_clip.tolist())` を満たす経路を評価する。
        if any(math.isfinite(float(v)) for v in j_sweep_sub_clip.tolist()):
            ax0.plot(windows_ns, j_sweep_sub_clip, marker="s", lw=1.4, color="tab:green", label="accidental-subtracted (clipped)")

        ax0.axhline(0.0, color="0.2", ls="--", lw=1.0)
        ax0.axhline(float(j_trial), color="tab:orange", ls=":", lw=1.2, label="trial-based")
        ax0.set_xscale("log")
        ax0.set_xlabel("window (ns)")
        ax0.set_ylabel("CH J_prob")
        ax0.set_title("Window sweep")
        ax0.grid(True, which="both", alpha=0.3, ls=":")
        ax0.legend(fontsize=9, frameon=True)

        ax1 = fig.add_subplot(gs[0, 1])
        offx = [r["delta_offset_ns"] for r in offset_rows]
        offy = [r["J_prob"] for r in offset_rows]
        ax1.plot(offx, offy, marker="o", lw=1.6)
        ax1.axhline(0.0, color="0.2", ls="--", lw=1.0)
        ax1.axhline(float(j_trial), color="tab:orange", ls=":", lw=1.2)
        ax1.set_xlabel("delta offset (ns)")
        ax1.set_ylabel("CH J_prob (subset)")
        ax1.set_title(f"Offset sweep (window={win_ns:g} ns; subset)")
        ax1.grid(True, alpha=0.3, ls=":")

        ax2 = fig.add_subplot(gs[0, 2])
        allv = np.concatenate([a0, a1v, b0, b1v])
        hi = float(np.percentile(allv, 99.5))
        hi = max(50.0, min(hi, 2000.0))
        bins = np.linspace(0.0, hi, 120)
        ax2.hist(a0, bins=bins, alpha=0.35, label="Alice (set=0)")
        ax2.hist(a1v, bins=bins, alpha=0.35, label="Alice (set=1)")
        ax2.hist(b0, bins=bins, alpha=0.35, label="Bob (set=0)")
        ax2.hist(b1v, bins=bins, alpha=0.35, label="Bob (set=1)")
        ax2.set_xlabel("click delay from sync (ns)")
        ax2.set_ylabel("count")
        ax2.set_title(f"Delay (KS_A={ks_a:.3f}, KS_B={ks_b:.3f})")
        ax2.legend(fontsize=8, frameon=True)
        ax2.grid(True, alpha=0.3, ls=":")

        fig.suptitle(f"Bell (NIST): {dataset_id}", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / "summary.png", bbox_inches="tight")
        plt.close(fig)

    return {
        "dataset_id": dataset_id,
        "statistic": "CH_J",
        "window_delta": delta_j,
        "window_delta_accidental_subtracted": delta_j_sub,
        "offset_delta": delta_j2,
        "ks_delay": {"Alice": float(ks_a), "Bob": float(ks_b)},
        "delay_signature": delay_signature,
        "baseline": {
            "J_trial": float(j_trial),
            "J_trial_sigma_boot": float(j_sigma_trial),
            "syncs_used": int(counts["syncs_used"]),
        },
    }


# 関数: `_kwiat2013_dataset` の入出力契約と処理意図を定義する。

def _kwiat2013_dataset(*, dataset_id: str, overwrite: bool) -> dict[str, Any]:
    """
    Christensen et al. 2013 (PRL 111, 130406; Kwiat group) CH Bell-test data.

    Public dataset notes (data_organization.txt):
    - channel=15 is the PC trigger for a trial
    - PC triggers twice per channel-15 timetag (hidden midpoint trigger)
    - recommended coincidence window: ~18,000 timebins (PC on=2us=12,800 bins at 6.4 GHz)

    Here we interpret a "trial" as each PC trigger (incl. hidden midpoint triggers), and count a trial hit
    if a channel (Alice=1 / Bob=2) fires within a configurable window after the trigger.
    """
    out_dir = OUT_BASE / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ROOT / "data" / "quantum" / "sources" / "kwiat2013_prl111_130406" / "CH_Bell_Data.zip"
    # 条件分岐: `not zip_path.exists()` を満たす経路を評価する。
    if not zip_path.exists():
        raise SystemExit(
            "[fail] missing Kwiat/Christensen 2013 data. Run:\n"
            "  python -B scripts/quantum/fetch_kwiat2013_prl111_130406.py"
        )

    # Representative dataset folder with 300 mat files.

    dataset_folder = "05082013_15 t753 a252ap38"
    prefix = f"CH_Bell_Data/{dataset_folder}/"

    # Window is in "timebins" (6.4e9 bins/sec). We sweep around the dataset guidance (18,000 bins).
    ref_window_bins = 18_000
    windows_bins = [
        2_000,
        4_000,
        6_000,
        8_000,
        10_000,
        12_000,
        12_800,  # 2 us gate (dataset note)
        15_000,
        18_000,  # dataset recommendation
        22_000,
        30_000,
        40_000,
        50_000,
    ]
    windows_bins = sorted({int(x) for x in windows_bins if int(x) > 0})
    # 条件分岐: `ref_window_bins not in windows_bins` を満たす経路を評価する。
    if ref_window_bins not in windows_bins:
        windows_bins.append(int(ref_window_bins))
        windows_bins.sort()

    npz_path = out_dir / "normalized_events.npz"
    meta_path = out_dir / "normalized_events.json"
    trial_counts_path = out_dir / "trial_based_counts.json"
    w_path = out_dir / "window_sweep_metrics.json"
    delay_sig_ref_npz = out_dir / "delay_signature_ref_samples.npz"

    if (
        (not overwrite)
        and npz_path.exists()
        and meta_path.exists()
        and trial_counts_path.exists()
        and w_path.exists()
        and delay_sig_ref_npz.exists()
    ):
        wj = json.loads(w_path.read_text(encoding="utf-8"))
        summ = wj.get("summary") if isinstance(wj.get("summary"), dict) else {}
        delta_j = summ.get("delta_J")
        tb = wj.get("trial_based") if isinstance(wj.get("trial_based"), dict) else {}
        ks = wj.get("time_tag_delay_ks") if isinstance(wj.get("time_tag_delay_ks"), dict) else {}
        delay_sig = wj.get("delay_signature") if isinstance(wj.get("delay_signature"), dict) else None
        return {
            "dataset_id": dataset_id,
            "statistic": "CH_J",
            "window_delta": float(delta_j) if delta_j is not None and math.isfinite(float(delta_j)) else None,
            "offset_delta": None,
            "ks_delay": {"Alice": float(ks.get("alice")) if ks.get("alice") is not None else None, "Bob": float(ks.get("bob")) if ks.get("bob") is not None else None},
            "delay_signature": delay_sig,
            "baseline": {
                "J_trial": float(tb.get("J_prob")) if tb.get("J_prob") is not None else None,
                "J_trial_sigma_boot": float(tb.get("J_sigma_boot")) if tb.get("J_sigma_boot") is not None else None,
                "ref_window_ns": float(tb.get("ref_window_ns")) if tb.get("ref_window_ns") is not None else None,
            },
        }

    # 関数: `_dt_min_by_trial` の入出力契約と処理意図を定義する。

    def _dt_min_by_trial(*, click_t: np.ndarray, triggers: np.ndarray) -> np.ndarray:
        max_i64 = np.iinfo(np.int64).max
        out = np.full(int(triggers.size), max_i64, dtype=np.int64)
        # 条件分岐: `int(click_t.size) == 0 or int(triggers.size) == 0` を満たす経路を評価する。
        if int(click_t.size) == 0 or int(triggers.size) == 0:
            return out

        idx = np.searchsorted(triggers, click_t, side="right") - 1
        m = idx >= 0
        # 条件分岐: `int(np.sum(m)) == 0` を満たす経路を評価する。
        if int(np.sum(m)) == 0:
            return out

        idx2 = idx[m].astype(np.int64, copy=False)
        dt = click_t[m] - triggers[idx2]
        m2 = dt >= 0
        # 条件分岐: `int(np.sum(m2)) == 0` を満たす経路を評価する。
        if int(np.sum(m2)) == 0:
            return out

        np.minimum.at(out, idx2[m2], dt[m2])
        return out

    import io  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    try:
        from scipy.io import loadmat  # noqa: PLC0415
    except Exception as e:
        raise SystemExit(f"[fail] scipy is required to read MATLAB v5 .mat files: {e}")

    mat_names: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        mat_names = sorted([n for n in zf.namelist() if n.startswith(prefix) and n.lower().endswith(".mat")])

    # 条件分岐: `not mat_names` を満たす経路を評価する。

    if not mat_names:
        raise SystemExit(f"[fail] no .mat files found under: {prefix} (zip={zip_path})")

    bins_per_s: float | None = None
    bin_s: float | None = None

    # Counts (window sweep)
    n_w = len(windows_bins)
    n_trials = np.zeros((2, 2), dtype=np.int64)
    n_trials_a = np.zeros(2, dtype=np.int64)
    n_trials_b = np.zeros(2, dtype=np.int64)
    n_coinc = np.zeros((n_w, 2, 2), dtype=np.int64)
    n_click_a = np.zeros((n_w, 2), dtype=np.int64)
    n_click_b = np.zeros((n_w, 2), dtype=np.int64)

    # Delay distributions for KS (relative to trigger; for ref window only)
    dt_a_ref_ns: dict[int, list[float]] = {0: [], 1: []}
    dt_b_ref_ns: dict[int, list[float]] = {0: [], 1: []}

    # Normalized event times (seconds)
    a_t_s_list: list[np.ndarray] = []
    b_t_s_list: list[np.ndarray] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for k, name in enumerate(mat_names):
            # 条件分岐: `k % 50 == 0` を満たす経路を評価する。
            if k % 50 == 0:
                print(f"  ... kwi at {k}/{len(mat_names)}: {name.rsplit('/', 1)[-1]}")

            raw = zf.read(name)
            m = loadmat(io.BytesIO(raw), squeeze_me=True)

            try:
                a_set = int(np.asarray(m.get("asetting")).reshape(-1)[0]) - 1
                b_set = int(np.asarray(m.get("bsetting")).reshape(-1)[0]) - 1
            except Exception:
                continue

            # 条件分岐: `a_set not in (0, 1) or b_set not in (0, 1)` を満たす経路を評価する。

            if a_set not in (0, 1) or b_set not in (0, 1):
                continue

            try:
                tb = float(np.asarray(m.get("timebins")).reshape(-1)[0])
            except Exception:
                continue

            # 条件分岐: `not math.isfinite(tb) or tb <= 0` を満たす経路を評価する。

            if not math.isfinite(tb) or tb <= 0:
                continue

            # 条件分岐: `bins_per_s is None` を満たす経路を評価する。

            if bins_per_s is None:
                bins_per_s = tb
                bin_s = 1.0 / float(bins_per_s)
            else:
                # 条件分岐: `abs(tb - float(bins_per_s)) > 1e-6` を満たす経路を評価する。
                if abs(tb - float(bins_per_s)) > 1e-6:
                    raise RuntimeError(f"inconsistent timebins in .mat: {tb} != {bins_per_s} ({name})")

            chan_keys = [kk for kk in m.keys() if kk.startswith("channel") and kk[7:].isdigit()]
            time_keys = [kk for kk in m.keys() if kk.startswith("timetags") and kk[8:].isdigit()]
            # 条件分岐: `not chan_keys or not time_keys` を満たす経路を評価する。
            if not chan_keys or not time_keys:
                continue

            chan_key = chan_keys[0]
            time_key = time_keys[0]

            ch = np.asarray(m[chan_key], dtype=np.int16).reshape(-1)
            tt = np.asarray(m[time_key], dtype=np.int64).reshape(-1)
            # 条件分岐: `ch.size != tt.size or ch.size == 0` を満たす経路を評価する。
            if ch.size != tt.size or ch.size == 0:
                continue

            # Click times (raw)

            t_a = tt[ch == 1]
            t_b = tt[ch == 2]
            # 条件分岐: `t_a.size` を満たす経路を評価する。
            if t_a.size:
                a_t_s_list.append(t_a.astype(np.float64, copy=False) * float(bin_s))

            # 条件分岐: `t_b.size` を満たす経路を評価する。

            if t_b.size:
                b_t_s_list.append(t_b.astype(np.float64, copy=False) * float(bin_s))

            # Trigger times (channel 15) + hidden midpoint triggers.

            t15 = np.sort(tt[ch == 15])
            # 条件分岐: `int(t15.size) < 2` を満たす経路を評価する。
            if int(t15.size) < 2:
                continue

            mids = ((t15[:-1] + t15[1:]) // 2).astype(np.int64, copy=False)
            trig = np.empty(int(2 * t15.size - 1), dtype=np.int64)
            trig[0] = int(t15[0])
            trig[1::2] = mids
            trig[2::2] = t15[1:]
            n_trial = int(trig.size)
            # 条件分岐: `n_trial <= 0` を満たす経路を評価する。
            if n_trial <= 0:
                continue

            n_trials[a_set, b_set] += n_trial
            n_trials_a[a_set] += n_trial
            n_trials_b[b_set] += n_trial

            dt_a = _dt_min_by_trial(click_t=np.asarray(t_a, dtype=np.int64), triggers=trig)
            dt_b = _dt_min_by_trial(click_t=np.asarray(t_b, dtype=np.int64), triggers=trig)

            # KS uses ref window only
            if bin_s is not None:
                ma = dt_a <= int(ref_window_bins)
                mb = dt_b <= int(ref_window_bins)
                # 条件分岐: `int(np.sum(ma)) > 0` を満たす経路を評価する。
                if int(np.sum(ma)) > 0:
                    dt_a_ref_ns[a_set].extend((dt_a[ma].astype(np.float64) * float(bin_s) * 1e9).tolist())

                # 条件分岐: `int(np.sum(mb)) > 0` を満たす経路を評価する。

                if int(np.sum(mb)) > 0:
                    dt_b_ref_ns[b_set].extend((dt_b[mb].astype(np.float64) * float(bin_s) * 1e9).tolist())

            # Window sweep counts

            for wi, w_bins in enumerate(windows_bins):
                a_hit = dt_a <= int(w_bins)
                b_hit = dt_b <= int(w_bins)
                n_click_a[wi, a_set] += int(np.sum(a_hit))
                n_click_b[wi, b_set] += int(np.sum(b_hit))
                n_coinc[wi, a_set, b_set] += int(np.sum(a_hit & b_hit))

    # 条件分岐: `bins_per_s is None or bin_s is None` を満たす経路を評価する。

    if bins_per_s is None or bin_s is None:
        raise SystemExit("[fail] could not determine timebins from .mat files")

    # KS distances (setting dependence of click delay from trigger)

    a0 = np.asarray(dt_a_ref_ns[0], dtype=float)
    a1v = np.asarray(dt_a_ref_ns[1], dtype=float)
    b0 = np.asarray(dt_b_ref_ns[0], dtype=float)
    b1v = np.asarray(dt_b_ref_ns[1], dtype=float)

    # Export representative ref-window delay samples for deterministic null tests (do not require re-reading .mat files).
    np.savez_compressed(
        delay_sig_ref_npz,
        alice_setting0_dt_ns=a0.astype(np.float64, copy=False),
        alice_setting1_dt_ns=a1v.astype(np.float64, copy=False),
        bob_setting0_dt_ns=b0.astype(np.float64, copy=False),
        bob_setting1_dt_ns=b1v.astype(np.float64, copy=False),
    )
    ks_a = _ks_distance(a0, a1v)
    ks_b = _ks_distance(b0, b1v)

    eps_sig = 0.1
    delay_signature = {
        "method": {
            "name": "delta_median",
            "epsilon": float(eps_sig),
            "definition": "dt = min(click_t - trigger_t) per trial (ns) at ref window; compare by local setting.",
        },
        "Alice": _delay_signature_delta_median(a0, a1v, epsilon=eps_sig),
        "Bob": _delay_signature_delta_median(b0, b1v, epsilon=eps_sig),
    }

    # Build sweep rows and bootstrap σ(J) per window
    a1 = 0
    b1 = 0
    windows_ns = [float(w) * float(bin_s) * 1e9 for w in windows_bins]
    rows: list[dict[str, Any]] = []
    j_values: list[float] = []
    j_sigmas: list[float] = []
    for i in range(n_w):
        c = n_coinc[i].astype(np.int64, copy=False)
        ca = n_click_a[i].astype(np.int64, copy=False)
        cb = n_click_b[i].astype(np.int64, copy=False)

        pa1 = float(ca[a1] / max(1, int(n_trials_a[a1])))
        pb1 = float(cb[b1] / max(1, int(n_trials_b[b1])))
        j = (
            c[0, 0] / max(1, int(n_trials[0, 0]))
            + c[0, 1] / max(1, int(n_trials[0, 1]))
            + c[1, 0] / max(1, int(n_trials[1, 0]))
            - c[1, 1] / max(1, int(n_trials[1, 1]))
            - pa1
            - pb1
        )
        j_sigma = _bootstrap_ch_j_sigma(
            n_trials=n_trials,
            n_coinc=c,
            n_trials_a=n_trials_a,
            n_trials_b=n_trials_b,
            n_click_a=ca,
            n_click_b=cb,
            a1=a1,
            b1=b1,
            n_boot=2000,
            seed=900 + i,
        )
        rows.append(
            {
                "window_ns": float(windows_ns[i]),
                "window_bins": int(windows_bins[i]),
                "pairs_total": int(np.sum(c)),
                "J_prob": float(j),
                "J_sigma_boot": float(j_sigma),
                "click_a_by_setting": ca.astype(int).tolist(),
                "click_b_by_setting": cb.astype(int).tolist(),
                "coinc_by_setting_pair": c.astype(int).tolist(),
            }
        )
        j_values.append(float(j))
        j_sigmas.append(float(j_sigma))

    j_min, j_max = _min_max(j_values)
    delta_j = (float(j_max) - float(j_min)) if (j_min is not None and j_max is not None) else None

    # Baseline at the dataset recommendation (18,000 bins)
    try:
        ref_idx = windows_bins.index(int(ref_window_bins))
    except ValueError:
        ref_idx = int(np.argmin(np.abs(np.asarray(windows_bins, dtype=float) - float(ref_window_bins))))

    j_ref = float(rows[ref_idx]["J_prob"])
    j_ref_sigma = float(rows[ref_idx]["J_sigma_boot"])
    ref_window_ns = float(rows[ref_idx]["window_ns"])

    # trial-based counts (for covariance / denominators): store baseline window counts
    _write_json(
        trial_counts_path,
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "source": {
                "name": "Christensen et al. 2013 (PRL 111, 130406) CH Bell-test data (Illinois QI page)",
                "zip": str(zip_path),
                "folder": dataset_folder,
            },
            "config": {
                "include_hidden_midpoint_trigger": True,
                "timebins_per_second": float(bins_per_s),
                "ref_window_bins": int(ref_window_bins),
                "ref_window_ns": float(ref_window_ns),
            },
            "counts": {
                "trials_by_setting_pair": n_trials.astype(int).tolist(),
                "alice_trials_by_setting": n_trials_a.astype(int).tolist(),
                "bob_trials_by_setting": n_trials_b.astype(int).tolist(),
                "alice_clicks_by_setting": n_click_a[ref_idx].astype(int).tolist(),
                "bob_clicks_by_setting": n_click_b[ref_idx].astype(int).tolist(),
                "coinc_by_setting_pair": n_coinc[ref_idx].astype(int).tolist(),
            },
        },
    )

    _write_json(
        w_path,
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "statistic": {"name": "CH J_prob (A1=0,B1=0)", "local_bound": 0.0},
            "trial_based": {
                "J_prob": float(j_ref),
                "J_sigma_boot": float(j_ref_sigma),
                "a1": int(a1),
                "b1": int(b1),
                "ref_window_bins": int(ref_window_bins),
                "ref_window_ns": float(ref_window_ns),
            },
            "time_tag_delay_ks": {"alice": float(ks_a), "bob": float(ks_b)},
            "delay_signature": delay_signature,
            "config": {
                "dataset_folder": dataset_folder,
                "zip": str(zip_path),
                "include_hidden_midpoint_trigger": True,
                "timebins_per_second": float(bins_per_s),
                "recommended_window_bins": int(ref_window_bins),
                "windows_bins": windows_bins,
                "windows_ns": windows_ns,
                "bootstrap": {"method": "parametric (binomial trials)", "n_boot": 2000, "seed_base": 900},
            },
            "rows": rows,
            "summary": {"J_min": j_min, "J_max": j_max, "delta_J": delta_j},
        },
    )

    # Normalized click times in seconds (for generic dt-peak window heuristics / bookkeeping)
    a_t_s = np.sort(np.concatenate(a_t_s_list)) if a_t_s_list else np.asarray([], dtype=np.float64)
    b_t_s = np.sort(np.concatenate(b_t_s_list)) if b_t_s_list else np.asarray([], dtype=np.float64)
    np.savez_compressed(npz_path, a_t_s=a_t_s, b_t_s=b_t_s)
    _write_json(
        meta_path,
        {
            "generated_utc": _utc_now(),
            "dataset_id": dataset_id,
            "source": {
                "name": "Christensen et al. 2013 (PRL 111, 130406) CH Bell-test data (Illinois QI page)",
                "zip": str(zip_path),
                "folder": dataset_folder,
            },
            "schema": {"a_t_s": "float64 seconds (event time; Alice)", "b_t_s": "float64 seconds (event time; Bob)"},
            "counts": {"alice_events": int(a_t_s.size), "bob_events": int(b_t_s.size)},
            "notes": [
                "Raw time-tags are in 'timebins' (6.4e9 bins/sec).",
                "Trial triggers are channel 15 plus hidden midpoint triggers between consecutive channel-15 tags.",
                "Bell statistic is computed as CH J_prob from trial hits within a window after the trigger.",
            ],
        },
    )

    # Summary plot (optional)
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        pass
    else:
        xs = [float(r["window_ns"]) for r in rows]
        ys = [float(r["J_prob"]) for r in rows]
        es = [float(r["J_sigma_boot"]) for r in rows]
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.2), dpi=170)
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4, label="J(window)")
        ax.fill_between(xs, np.asarray(ys) - np.asarray(es), np.asarray(ys) + np.asarray(es), alpha=0.12)
        ax.axhline(0.0, color="0.25", ls="--", lw=1.0, label="local bound J=0")
        ax.axvline(float(ref_window_ns), color="0.25", ls=":", lw=1.0, label="recommended (~18000 bins)")
        ax.set_xscale("log")
        ax.set_xlabel("window half-width (ns)")
        ax.set_ylabel("CH J_prob (A1=0,B1=0)")
        ax.set_title(f"Kwiat/Christensen2013: {dataset_id} (KS_A={ks_a:.3f}, KS_B={ks_b:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / "summary.png", bbox_inches="tight")
        plt.close(fig)

    return {
        "dataset_id": dataset_id,
        "statistic": "CH_J",
        "window_delta": delta_j,
        "offset_delta": None,
        "ks_delay": {"Alice": float(ks_a), "Bob": float(ks_b)},
        "delay_signature": delay_signature,
        "baseline": {
            "J_trial": float(j_ref),
            "J_trial_sigma_boot": float(j_ref_sigma),
            "ref_window_ns": float(ref_window_ns),
            "trials_total": int(np.sum(n_trials)),
            "files": int(len(mat_names)),
        },
    }


# 関数: `_build_table1_row` の入出力契約と処理意図を定義する。

def _build_table1_row(*, results: list[dict[str, Any]]) -> dict[str, Any]:
    weihs = next((r for r in results if r.get("dataset_id") == "weihs1998_longdist_longdist1"), None)
    nist = next((r for r in results if r.get("dataset_id") == "nist_03_43_afterfixingModeLocking_s3600"), None)
    kwiat = next((r for r in results if r.get("dataset_id") == "kwiat2013_prl111_130406_05082013_15"), None)
    d15 = next((r for r in results if r.get("dataset_id") == "delft_hensen2015"), None)
    d16 = next((r for r in results if r.get("dataset_id") == "delft_hensen2016_srep30289"), None)

    parts: list[str] = []
    # 条件分岐: `weihs is not None and weihs.get("window_delta") is not None` を満たす経路を評価する。
    if weihs is not None and weihs.get("window_delta") is not None:
        parts.append(f"Weihs1998 Δ|S|≈{float(weihs['window_delta']):.3g}")

    # 条件分岐: `d15 is not None and d15.get("offset_delta") is not None` を満たす経路を評価する。

    if d15 is not None and d15.get("offset_delta") is not None:
        parts.append(f"Delft2015 ΔS≈{float(d15['offset_delta']):.3g}")

    # 条件分岐: `d16 is not None and d16.get("offset_delta") is not None` を満たす経路を評価する。

    if d16 is not None and d16.get("offset_delta") is not None:
        parts.append(f"Delft2016 ΔS≈{float(d16['offset_delta']):.3g}")

    # 条件分岐: `nist is not None and nist.get("window_delta") is not None` を満たす経路を評価する。

    if nist is not None and nist.get("window_delta") is not None:
        parts.append(f"NIST ΔJ≈{float(nist['window_delta']):.3g}")

    # 条件分岐: `kwiat is not None and kwiat.get("window_delta") is not None` を満たす経路を評価する。

    if kwiat is not None and kwiat.get("window_delta") is not None:
        parts.append(f"Kwiat2013 ΔJ≈{float(kwiat['window_delta']):.3g}")

    metric_public = " / ".join(parts) if parts else "n/a"
    return {
        "topic": "Bell（公開一次データ; selection感度）",
        "observable": "統計量の選別依存（window/offset sweep）",
        "data": "NIST(time-tag+trial) + Weihs1998(time-tag) + Delft2015/2016(trial) + Kwiat2013(time-tag+trigger)",
        "n": int(len([x for x in [weihs, nist, kwiat, d15, d16] if x is not None])),
        "reference": "|S|≤2（CHSH）/ J≤0（CH）",
        "pmodel": "selectionノブで統計量が動く（Δ）",
        "metric": "統計: parametric bootstrap（固定）; 系統: sweep変動; delay(Δmedian; z)＋KS(proxy)併記",
        "metric_public": metric_public,
    }


# 関数: `_build_falsification_pack` の入出力契約と処理意図を定義する。

def _build_falsification_pack(*, results: list[dict[str, Any]]) -> dict[str, Any]:
    # Operational falsification examples:
    # - selection_origin: if sweep delta is < 1σ(stat), then "selection origin" is disfavored.
    # - setting_delay: if delay signature z is < 3, then "setting-dependent delay weighting" is disfavored.
    pack: dict[str, Any] = {
        "generated_utc": _utc_now(),
        "thresholds": {},
        "natural_window": {},
        "datasets": [],
        "conditions": [],
    }
    pack["thresholds"] = {
        "selection_origin_ratio_min": 1.0,
        "delay_signature_z_min": 3.0,
        "ks_delay_min_legacy": 0.01,
        "note": "These are operational thresholds for this repository (not universal physics thresholds).",
    }
    pack["natural_window"] = {
        "plateau_fraction_pairs": 0.99,
        "overrides": {
            "nist_*": "trial_match_pairs_total (match trial-based coincidence total)",
            "kwiat2013_*": "dataset_recommendation (≈18000 timebins; PC-triggered trials)",
        },
        "note": "Natural window is defined operationally (to avoid p-hacking). See longterm_consistency.json.",
    }

    for r in results:
        ds = str(r.get("dataset_id"))
        ds_dir = OUT_BASE / ds
        stat = str(r.get("statistic") or "")
        entry: dict[str, Any] = {"dataset_id": ds, "statistic": stat}

        # 条件分岐: `stat == "CHSH_S"` を満たす経路を評価する。
        if stat == "CHSH_S":
            # Delft event-ready experiments: selection knob is the event-ready offset (no coincidence-window pairing step).
            entry["selection_knob"] = "event_ready_offset_ps"
            try:
                o = json.loads((ds_dir / "offset_sweep_metrics.json").read_text(encoding="utf-8"))
                rows = list(o.get("rows") or [])
                s_vals = []
                s_errs_raw = []
                for row in rows:
                    # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
                    if not isinstance(row, dict):
                        continue

                    v = row.get("S")
                    # 条件分岐: `v is None` を満たす経路を評価する。
                    if v is None:
                        v = row.get("S_combined")

                    s_vals.append(v)
                    e = row.get("S_err")
                    # 条件分岐: `e is None` を満たす経路を評価する。
                    if e is None:
                        e = row.get("S_combined_err")

                    s_errs_raw.append(e)

                s_lo, s_hi = _min_max(s_vals)
                delta = float(s_hi - s_lo) if (s_lo is not None and s_hi is not None and s_hi >= s_lo) else None
                s_errs = _finite(s_errs_raw)
                # 条件分岐: `s_errs` を満たす経路を評価する。
                if s_errs:
                    sigma_med = float(np.nanmedian(np.asarray(s_errs, dtype=float)))
                else:
                    baseline = o.get("baseline") if isinstance(o.get("baseline"), dict) else {}
                    b = baseline.get("S_err")
                    # 条件分岐: `b is None` を満たす経路を評価する。
                    if b is None:
                        b = baseline.get("S_combined_err")

                    sigma_med = float(b) if b is not None else None
            except Exception:
                delta = None
                sigma_med = None

            entry["delta_offset"] = r.get("offset_delta") if r.get("offset_delta") is not None else delta
            entry["sigma_stat_med"] = sigma_med
            # 条件分岐: `entry.get("delta_offset") is not None and sigma_med is not None and sigma_med...` を満たす経路を評価する。
            if entry.get("delta_offset") is not None and sigma_med is not None and sigma_med > 0:
                entry["ratio"] = float(entry["delta_offset"]) / float(sigma_med)
            else:
                entry["ratio"] = None

            entry["recommended_start_offset_ps"] = 0
            entry["recommended_window_ns"] = None
            entry["recommended_window_method"] = {
                "name": "event_ready_protocol",
                "note": "Use the published event-ready protocol baseline; treat offset sweep as a systematic (not an optimizer).",
            }
        # 条件分岐: 前段条件が不成立で、`stat.startswith("CHSH")` を追加評価する。
        elif stat.startswith("CHSH"):
            entry["selection_knob"] = "window_ns"
            try:
                w = json.loads((ds_dir / "window_sweep_metrics.json").read_text(encoding="utf-8"))
                sigmas = [row.get("S_fixed_sigma_boot") for row in w.get("rows", [])]
                sigma_med = (
                    float(np.nanmedian(np.asarray(_finite(sigmas), dtype=float))) if _finite(sigmas) else None
                )
                rec_window, rec_method = _recommend_natural_window(
                    dataset_id=ds,
                    ds_dir=ds_dir,
                    rows=list(w.get("rows") or []),
                    plateau_fraction=float(pack["natural_window"]["plateau_fraction_pairs"]),
                )
            except Exception:
                sigma_med = None
                rec_window = None
                rec_method = {"name": "unknown"}

            entry["delta_window"] = r.get("window_delta")
            entry["sigma_stat_med"] = sigma_med
            # 条件分岐: `r.get("window_delta") is not None and sigma_med is not None and sigma_med > 0` を満たす経路を評価する。
            if r.get("window_delta") is not None and sigma_med is not None and sigma_med > 0:
                entry["ratio"] = float(r["window_delta"]) / float(sigma_med)
            else:
                entry["ratio"] = None

            entry["recommended_window_ns"] = rec_window
            entry["recommended_window_method"] = rec_method
        # 条件分岐: 前段条件が不成立で、`stat == "CH_J"` を追加評価する。
        elif stat == "CH_J":
            entry["selection_knob"] = "window_ns"
            try:
                w = json.loads((ds_dir / "window_sweep_metrics.json").read_text(encoding="utf-8"))
                sigmas = [row.get("J_sigma_boot") for row in w.get("rows", [])]
                sigma_med = (
                    float(np.nanmedian(np.asarray(_finite(sigmas), dtype=float))) if _finite(sigmas) else None
                )
                rec_window, rec_method = _recommend_natural_window(
                    dataset_id=ds,
                    ds_dir=ds_dir,
                    rows=list(w.get("rows") or []),
                    plateau_fraction=float(pack["natural_window"]["plateau_fraction_pairs"]),
                )
            except Exception:
                sigma_med = None
                rec_window = None
                rec_method = {"name": "unknown"}

            entry["delta_window"] = r.get("window_delta")
            entry["sigma_stat_med"] = sigma_med
            # 条件分岐: `r.get("window_delta") is not None and sigma_med is not None and sigma_med > 0` を満たす経路を評価する。
            if r.get("window_delta") is not None and sigma_med is not None and sigma_med > 0:
                entry["ratio"] = float(r["window_delta"]) / float(sigma_med)
            else:
                entry["ratio"] = None

            entry["recommended_window_ns"] = rec_window
            entry["recommended_window_method"] = rec_method

        entry["ks_delay"] = r.get("ks_delay")
        entry["delay_signature"] = r.get("delay_signature")
        # 条件分岐: `ds.startswith("delft_")` を満たす経路を評価する。
        if ds.startswith("delft_"):
            entry["recommended_start_offset_ps"] = 0

        pack["datasets"].append(entry)

    pack["conditions"] = [
        {
            "id": "selection_origin_falsified_if_ratio_lt_1",
            "statement": "selection sweep（window/offset）の変動幅 Δ(stat) が 1σ(stat) 未満なら、selection 起源仮説は棄却（このデータ範囲では）",
        },
        {
            "id": "setting_delay_falsified_if_delay_z_lt_3",
            "statement": "delay シグネチャ（Δmedian; z）が 3 未満なら、setting 依存重み仮説は棄却（このΔmedian定義では）",
        },
        {
            "id": "setting_delay_falsified_if_ks_lt_0p01_legacy",
            "statement": "legacy: delay 分布の setting 依存（KS）が 0.01 未満なら、setting 依存重み仮説は棄却（KSはproxyとして併記）",
        },
    ]
    return pack


# 関数: `_enrich_falsification_pack_cross_dataset` の入出力契約と処理意図を定義する。

def _enrich_falsification_pack_cross_dataset(*, pack_path: Path) -> None:
    """
    Make the Bell falsification pack self-contained by attaching cross-dataset summaries.

    The pack should stay lightweight: we embed only summaries and references, not full matrices.
    """
    # 条件分岐: `not pack_path.exists()` を満たす経路を評価する。
    if not pack_path.exists():
        return

    pack = _read_json(pack_path)
    bell_dir = pack_path.parent

    longterm_path = bell_dir / "longterm_consistency.json"
    sys15_path = bell_dir / "systematics_decomposition_15items.json"
    cov_path = bell_dir / "cross_dataset_covariance.json"
    loophole_path = bell_dir / "selection_loophole_quantification.json"
    null_tests_path = bell_dir / "null_tests_summary.json"
    freeze_policy_path = bell_dir / "freeze_policy.json"
    pairing_xc_path = bell_dir / "crosscheck_pairing_summary.json"

    longterm = _read_json_or_none(longterm_path)
    sys15 = _read_json_or_none(sys15_path)
    cov = _read_json_or_none(cov_path)
    loophole = _read_json_or_none(loophole_path)
    null_tests = _read_json_or_none(null_tests_path)
    freeze_policy = _read_json_or_none(freeze_policy_path)
    pairing_xc = _read_json_or_none(pairing_xc_path)

    thresholds = pack.get("thresholds") if isinstance(pack.get("thresholds"), dict) else {}
    ratio_th = float(thresholds.get("selection_origin_ratio_min", 1.0))
    delay_z_th = float(thresholds.get("delay_signature_z_min", 3.0))

    ds_list = pack.get("datasets") if isinstance(pack.get("datasets"), list) else []
    ds_list = [d for d in ds_list if isinstance(d, dict)]

    # 関数: `_median` の入出力契約と処理意図を定義する。
    def _median(xs: list[float]) -> float | None:
        vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
        # 条件分岐: `not vals` を満たす経路を評価する。
        if not vals:
            return None

        return float(np.nanmedian(np.asarray(vals, dtype=float)))

    # 関数: `_min` の入出力契約と処理意図を定義する。

    def _min(xs: list[float]) -> float | None:
        vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
        return float(min(vals)) if vals else None

    # 関数: `_max` の入出力契約と処理意図を定義する。

    def _max(xs: list[float]) -> float | None:
        vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
        return float(max(vals)) if vals else None

    # --- per-statistic group summary (computed from the pack itself)

    def _group_key(stat: str) -> str:
        s = str(stat)
        # 条件分岐: `s.startswith("CHSH")` を満たす経路を評価する。
        if s.startswith("CHSH"):
            return "CHSH"

        # 条件分岐: `s.startswith("CH_")` を満たす経路を評価する。

        if s.startswith("CH_"):
            return "CH"

        return "other"

    groups: dict[str, list[dict[str, Any]]] = {}
    for d in ds_list:
        groups.setdefault(_group_key(str(d.get("statistic") or "")), []).append(d)

    group_summaries: dict[str, Any] = {}
    for g, items in groups.items():
        ratios = [float(x.get("ratio")) for x in items if x.get("ratio") is not None]
        zvals = [_delay_signature_z_max(x.get("delay_signature")) for x in items]
        zvals_f = [float(z) for z in zvals if z is not None]
        group_summaries[g] = {
            "n_datasets": int(len(items)),
            "dataset_ids": [str(x.get("dataset_id") or "") for x in items],
            "selection_ratio": {
                "threshold_min": ratio_th,
                "min": _min(ratios),
                "median": _median(ratios),
                "max": _max(ratios),
            },
            "delay_signature_z": {
                "threshold_min": delay_z_th,
                "min": float(min(zvals_f)) if zvals_f else None,
                "max": float(max(zvals_f)) if zvals_f else None,
                "note": "Only defined for time-tag datasets; event-ready/trial protocols are n/a.",
            },
        }

    # --- systematics 15-items summary (top contributors + correlation pairs)

    sys15_summary: dict[str, Any] = {"supported": False}
    # 条件分岐: `isinstance(sys15, dict)` を満たす経路を評価する。
    if isinstance(sys15, dict):
        item_summary = sys15.get("item_summary") if isinstance(sys15.get("item_summary"), list) else []
        item_summary = [x for x in item_summary if isinstance(x, dict)]
        item_summary = sorted(
            item_summary,
            key=lambda x: float(x.get("median_delta_over_sigma") or 0.0),
            reverse=True,
        )
        top_items = [
            {
                "item_id": str(x.get("item_id") or ""),
                "item_label": str(x.get("item_label") or ""),
                "median_delta_over_sigma": float(x.get("median_delta_over_sigma")),
                "max_delta_over_sigma": float(x.get("max_delta_over_sigma")),
            }
            for x in item_summary[:8]
            if x.get("median_delta_over_sigma") is not None
            and math.isfinite(float(x.get("median_delta_over_sigma")))
            and x.get("max_delta_over_sigma") is not None
            and math.isfinite(float(x.get("max_delta_over_sigma")))
        ]
        top_pairs = (
            (sys15.get("item_correlation") or {}).get("top_pairs")
            if isinstance(sys15.get("item_correlation"), dict)
            else None
        )
        top_pairs = top_pairs if isinstance(top_pairs, list) else []
        top_pairs = [x for x in top_pairs if isinstance(x, dict)]
        sys15_summary = {
            "supported": True,
            "top_items_by_median_delta_over_sigma": top_items,
            "top_item_correlation_pairs": top_pairs[:12],
        }

    # --- cross-dataset covariance summary (rank/condition number; strongest corr pair)

    cov_summary: dict[str, Any] = {"supported": False}
    # 条件分岐: `isinstance(longterm, dict) and isinstance(longterm.get("cross_dataset_covaria...` を満たす経路を評価する。
    if isinstance(longterm, dict) and isinstance(longterm.get("cross_dataset_covariance_summary"), dict):
        cov_summary = dict(longterm.get("cross_dataset_covariance_summary") or {})
    # 条件分岐: 前段条件が不成立で、`isinstance(cov, dict)` を追加評価する。
    elif isinstance(cov, dict):
        matrices = cov.get("matrices") if isinstance(cov.get("matrices"), dict) else {}
        eig = matrices.get("profile_corr_eigen") if isinstance(matrices.get("profile_corr_eigen"), dict) else {}
        cov_summary = {
            "supported": bool(eig.get("supported")),
            "rank_eps_1e-10": eig.get("rank_eps_1e-10"),
            "condition_number_abs": eig.get("condition_number_abs"),
        }

    # --- selection loophole (summary only)

    loophole_summary: dict[str, Any] | None = None
    # 条件分岐: `isinstance(loophole, dict) and isinstance(loophole.get("summary"), dict)` を満たす経路を評価する。
    if isinstance(loophole, dict) and isinstance(loophole.get("summary"), dict):
        loophole_summary = dict(loophole.get("summary") or {})

    # --- Attach

    pack["version"] = "1.4"
    pack["generated_utc"] = _utc_now()
    pack.setdefault("policy", {})
    # 条件分岐: `isinstance(pack.get("policy"), dict)` を満たす経路を評価する。
    if isinstance(pack.get("policy"), dict):
        pack["policy"]["blind_freeze"] = {
            "note": (
                "Freeze natural window/offset by an ex-ante rule (no access to Bell statistic). "
                "Sweeps are treated as systematics, not optimizers."
            ),
            "freeze_policy_json": {
                "path": _relpath_from_root(freeze_policy_path),
                "sha256": _sha256(freeze_policy_path) if freeze_policy_path.exists() else None,
            },
            "freeze_policy": freeze_policy,
        }

    pack["cross_dataset"] = {
        "generated_utc": _utc_now(),
        "summary": dict(longterm.get("summary") or {}) if isinstance(longterm, dict) else None,
        "statistic_groups": group_summaries,
        "covariance_summary": cov_summary,
        "systematics_15items_summary": sys15_summary,
        "selection_loophole_summary": loophole_summary,
        "null_tests_summary": null_tests,
        "pairing_crosscheck_summary": pairing_xc,
        "inputs": {
            "longterm_consistency_json": {
                "path": _relpath_from_root(longterm_path),
                "sha256": _sha256(longterm_path) if longterm_path.exists() else None,
            },
            "systematics_decomposition_15items_json": {
                "path": _relpath_from_root(sys15_path),
                "sha256": _sha256(sys15_path) if sys15_path.exists() else None,
            },
            "cross_dataset_covariance_json": {
                "path": _relpath_from_root(cov_path),
                "sha256": _sha256(cov_path) if cov_path.exists() else None,
            },
            "selection_loophole_quantification_json": {
                "path": _relpath_from_root(loophole_path),
                "sha256": _sha256(loophole_path) if loophole_path.exists() else None,
            },
            "null_tests_summary_json": {
                "path": _relpath_from_root(null_tests_path),
                "sha256": _sha256(null_tests_path) if null_tests_path.exists() else None,
            },
            "freeze_policy_json": {
                "path": _relpath_from_root(freeze_policy_path),
                "sha256": _sha256(freeze_policy_path) if freeze_policy_path.exists() else None,
            },
            "crosscheck_pairing_summary_json": {
                "path": _relpath_from_root(pairing_xc_path),
                "sha256": _sha256(pairing_xc_path) if pairing_xc_path.exists() else None,
            },
        },
    }
    pack.setdefault("outputs", {})
    # 条件分岐: `isinstance(pack.get("outputs"), dict)` を満たす経路を評価する。
    if isinstance(pack.get("outputs"), dict):
        pack["outputs"].update(
            {
                "falsification_pack_json": _relpath_from_root(pack_path),
                "falsification_pack_png": _relpath_from_root(bell_dir / "falsification_pack.png"),
                "longterm_consistency_json": _relpath_from_root(longterm_path),
                "longterm_consistency_png": _relpath_from_root(bell_dir / "longterm_consistency.png"),
                "systematics_decomposition_15items_json": _relpath_from_root(sys15_path),
                "systematics_decomposition_15items_png": _relpath_from_root(bell_dir / "systematics_decomposition_15items.png"),
                "cross_dataset_covariance_json": _relpath_from_root(cov_path),
                "cross_dataset_covariance_png": _relpath_from_root(bell_dir / "cross_dataset_covariance.png"),
                "selection_loophole_quantification_json": _relpath_from_root(loophole_path),
                "selection_loophole_quantification_png": _relpath_from_root(bell_dir / "selection_loophole_quantification.png"),
                "null_tests_summary_json": _relpath_from_root(null_tests_path),
                "freeze_policy_json": _relpath_from_root(freeze_policy_path),
                "crosscheck_pairing_summary_json": _relpath_from_root(pairing_xc_path),
            }
        )

    _write_json(pack_path, pack)


# 関数: `_write_covariance_products` の入出力契約と処理意図を定義する。

def _write_covariance_products(*, results: list[dict[str, Any]]) -> None:
    """
    Phase 7 / Step 7.16.7-7.16.9:
    - output/public/quantum/bell/<dataset>/covariance.json
    - output/public/quantum/bell/<dataset>/covariance_bootstrap.json
    - output/public/quantum/bell/cross_dataset_covariance.json / .png
    - output/public/quantum/bell/systematics_decomposition_15items.json / .csv / .png
    - output/public/quantum/bell/systematics_templates.json
    - output/public/quantum/bell/longterm_consistency.json / .png
    """

    # 関数: `_attach_cov_diagnostics` の入出力契約と処理意図を定義する。
    def _attach_cov_diagnostics(obj: dict[str, Any]) -> None:
        # 条件分岐: `not isinstance(obj, dict)` を満たす経路を評価する。
        if not isinstance(obj, dict):
            return

        for sweep_key in ("window_sweep", "offset_sweep"):
            sweep = obj.get(sweep_key)
            # 条件分岐: `not isinstance(sweep, dict)` を満たす経路を評価する。
            if not isinstance(sweep, dict):
                continue

            # 条件分岐: `not bool(sweep.get("supported"))` を満たす経路を評価する。

            if not bool(sweep.get("supported")):
                continue

            cov_json = sweep.get("cov")
            cov_arr = _matrix_from_json(cov_json)
            # 条件分岐: `cov_arr.size == 0` を満たす経路を評価する。
            if cov_arr.size == 0:
                continue

            # 条件分岐: `not isinstance(sweep.get("diag_sigma"), list)` を満たす経路を評価する。

            if not isinstance(sweep.get("diag_sigma"), list):
                sweep["diag_sigma"] = _diag_sigma_from_cov(cov_arr)

            sweep["eigen_summary"] = _cov_eigen_summary(cov_arr)

    n_bootstrap = 10_000
    cross_n_u = 81

    # --- per dataset covariance
    cov_index: list[dict[str, Any]] = []
    for r in results:
        ds = str(r.get("dataset_id") or "")
        ds_dir = OUT_BASE / ds
        cov_path = ds_dir / "covariance.json"
        cov_boot_path = ds_dir / "covariance_bootstrap.json"

        cov_obj: dict[str, Any] = {
            "generated_utc": _utc_now(),
            "dataset_id": ds,
            "supported": False,
            "window_sweep": {"supported": False},
            "offset_sweep": {"supported": False},
        }

        cov_boot_obj: dict[str, Any] = {
            "generated_utc": _utc_now(),
            "dataset_id": ds,
            "supported": False,
            "window_sweep": {"supported": False},
            "offset_sweep": {"supported": False},
        }

        # 条件分岐: `ds.startswith("weihs1998_")` を満たす経路を評価する。
        if ds.startswith("weihs1998_"):
            mod = _load_script_module(rel_path="scripts/quantum/weihs1998_time_tag_reanalysis.py", name="_weihs1998_cov")
            npz_path = ds_dir / "normalized_events.npz"
            meta_path = ds_dir / "normalized_events.json"
            w_path = ds_dir / "window_sweep_metrics.json"
            o_path = ds_dir / "offset_sweep_metrics.json"
            # 条件分岐: `npz_path.exists() and meta_path.exists() and w_path.exists() and o_path.exists()` を満たす経路を評価する。
            if npz_path.exists() and meta_path.exists() and w_path.exists() and o_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                wj = json.loads(w_path.read_text(encoding="utf-8"))
                oj = json.loads(o_path.read_text(encoding="utf-8"))

                offset_s = float((meta.get("offset_estimate") or {}).get("offset_s"))
                encoding = str(((wj.get("config") or {}).get("encoding") or "bit0-setting"))
                windows_ns = [float(x) for x in (wj.get("config") or {}).get("windows_ns") or []]
                ref_window_ns = float((wj.get("config") or {}).get("ref_window_ns") or 1.0)
                fv = wj.get("fixed_variant") or {}
                sign = fv.get("sign_matrix") or [[1, 1], [1, -1]]
                variant = ChshVariant(
                    swap_a=bool(fv.get("swap_a")),
                    swap_b=bool(fv.get("swap_b")),
                    sign_matrix=((int(sign[0][0]), int(sign[0][1])), (int(sign[1][0]), int(sign[1][1]))),
                )

                data = np.load(npz_path)
                t_a = data["a_t_s"].astype(np.float64, copy=False)
                c_a = data["a_c"].astype(np.uint16, copy=False)
                t_b = data["b_t_s"].astype(np.float64, copy=False)
                c_b = data["b_c"].astype(np.uint16, copy=False)

                n_chunks = 10
                t0 = float(min(float(t_a[0]), float(t_b[0])))
                t1 = float(max(float(t_a[-1]), float(t_b[-1])))
                edges = np.linspace(t0, t1, n_chunks + 1, dtype=float)

                Xw = np.full((n_chunks, len(windows_ns)), float("nan"), dtype=float)
                orows = list(oj.get("rows") or [])
                Xo = np.full((n_chunks, len(orows)), float("nan"), dtype=float)
                Nw = np.zeros((n_chunks, len(windows_ns), 2, 2), dtype=np.int64)
                Sw = np.zeros((n_chunks, len(windows_ns), 2, 2), dtype=np.int64)
                No = np.zeros((n_chunks, len(orows), 2, 2), dtype=np.int64)
                So = np.zeros((n_chunks, len(orows), 2, 2), dtype=np.int64)

                for k in range(n_chunks):
                    a0 = int(np.searchsorted(t_a, edges[k], side="left"))
                    a1 = int(np.searchsorted(t_a, edges[k + 1], side="left"))
                    b0 = int(np.searchsorted(t_b, edges[k], side="left"))
                    b1 = int(np.searchsorted(t_b, edges[k + 1], side="left"))
                    ta = t_a[a0:a1]
                    ca = c_a[a0:a1]
                    tb = t_b[b0:b1]
                    cb = c_b[b0:b1]

                    for i, w_ns in enumerate(windows_ns):
                        n, sum_prod, _ = mod._pair_and_accumulate(
                            ta,
                            ca,
                            tb,
                            cb,
                            offset_s=float(offset_s),
                            window_s=float(w_ns) * 1e-9,
                            encoding=encoding,
                        )
                        Nw[k, i] = np.asarray(n, dtype=np.int64)
                        Sw[k, i] = np.asarray(sum_prod, dtype=np.int64)
                        E = mod._safe_div(sum_prod, n)
                        s = _apply_chsh_variant(E, variant) if np.isfinite(E).all() else float("nan")
                        Xw[k, i] = abs(float(s)) if math.isfinite(float(s)) else float("nan")

                    for j, row in enumerate(orows):
                        try:
                            d_ns = float(row.get("delta_offset_ns"))
                        except Exception:
                            continue

                        off_s = float(offset_s) + float(d_ns) * 1e-9
                        n, sum_prod, _ = mod._pair_and_accumulate(
                            ta,
                            ca,
                            tb,
                            cb,
                            offset_s=off_s,
                            window_s=float(ref_window_ns) * 1e-9,
                            encoding=encoding,
                        )
                        No[k, j] = np.asarray(n, dtype=np.int64)
                        So[k, j] = np.asarray(sum_prod, dtype=np.int64)
                        E = mod._safe_div(sum_prod, n)
                        s = _apply_chsh_variant(E, variant) if np.isfinite(E).all() else float("nan")
                        Xo[k, j] = abs(float(s)) if math.isfinite(float(s)) else float("nan")

                cov_w = _nan_cov(Xw) if Xw.size else np.zeros((0, 0), dtype=float)
                cov_o = _nan_cov(Xo) if Xo.size else np.zeros((0, 0), dtype=float)

                cov_obj = {
                    "generated_utc": _utc_now(),
                    "dataset_id": ds,
                    "supported": True,
                    "method": {
                        "window_sweep": {
                            "name": "block_sample_cov",
                            "block_definition": "time-equal chunks on t_a/t_b (seconds)",
                            "n_chunks": n_chunks,
                            "edges_s": [float(x) for x in edges.tolist()],
                        },
                        "offset_sweep": {"name": "block_sample_cov", "n_chunks": n_chunks},
                    },
                    "natural_window": {},
                    "window_sweep": {
                        "supported": True,
                        "param_name": "window_ns",
                        "values": windows_ns,
                        "statistic_name": "CHSH |S| (fixed variant)",
                        "cov": _matrix_to_json(cov_w),
                        "diag_sigma": [
                            float(math.sqrt(float(v))) if math.isfinite(float(v)) and float(v) >= 0 else None
                            for v in np.diag(cov_w).tolist()
                        ],
                    },
                    "offset_sweep": {
                        "supported": True,
                        "param_name": "delta_offset_ns",
                        "values": [float(row.get("delta_offset_ns")) for row in orows],
                        "statistic_name": "CHSH |S| (fixed variant; ref window)",
                        "cov": _matrix_to_json(cov_o),
                    },
                }
                rec_w, rec_m = _recommend_natural_window(
                    dataset_id=ds,
                    ds_dir=ds_dir,
                    rows=list(wj.get("rows") or []),
                    plateau_fraction=0.99,
                )
                cov_obj["natural_window"] = {"method": rec_m, "recommended_window_ns": rec_w}

                n_boot = int(n_bootstrap)
                seed = _stable_seed("bell", "cov_boot", ds, "weihs1998")
                rng = np.random.default_rng(seed)
                boot_w = np.full((n_boot, len(windows_ns)), float("nan"), dtype=float)
                boot_o = np.full((n_boot, len(orows)), float("nan"), dtype=float)
                for bi in range(n_boot):
                    idx = rng.integers(0, n_chunks, size=n_chunks, endpoint=False)
                    n_sum_w = np.sum(Nw[idx], axis=0)
                    sum_sum_w = np.sum(Sw[idx], axis=0)
                    E_sum_w = _safe_div_array(sum_sum_w, n_sum_w)
                    for wi in range(len(windows_ns)):
                        E = E_sum_w[wi]
                        s = _apply_chsh_variant(E, variant) if np.isfinite(E).all() else float("nan")
                        boot_w[bi, wi] = abs(float(s)) if math.isfinite(float(s)) else float("nan")

                    n_sum_o = np.sum(No[idx], axis=0)
                    sum_sum_o = np.sum(So[idx], axis=0)
                    E_sum_o = _safe_div_array(sum_sum_o, n_sum_o)
                    for oi in range(len(orows)):
                        E = E_sum_o[oi]
                        s = _apply_chsh_variant(E, variant) if np.isfinite(E).all() else float("nan")
                        boot_o[bi, oi] = abs(float(s)) if math.isfinite(float(s)) else float("nan")

                cov_boot_w = _nan_cov(boot_w) if boot_w.size else np.zeros((0, 0), dtype=float)
                cov_boot_o = _nan_cov(boot_o) if boot_o.size else np.zeros((0, 0), dtype=float)

                mean_w: list[float | None] = []
                for row in (wj.get("rows") or []):
                    try:
                        mean_w.append(float(row.get("S_fixed_abs")))
                    except Exception:
                        mean_w.append(None)

                mean_o: list[float | None] = []
                for row in orows:
                    try:
                        mean_o.append(float(row.get("S_fixed_abs")))
                    except Exception:
                        mean_o.append(None)

                cov_boot_obj = {
                    "generated_utc": _utc_now(),
                    "dataset_id": ds,
                    "supported": True,
                    "method": {
                        "window_sweep": {
                            "name": "block_bootstrap",
                            "block_definition": "time-equal chunks on t_a/t_b (seconds)",
                            "n_chunks": n_chunks,
                            "edges_s": [float(x) for x in edges.tolist()],
                            "n_boot": int(n_boot),
                            "seed": int(seed),
                        },
                        "offset_sweep": {
                            "name": "block_bootstrap",
                            "block_definition": "time-equal chunks on t_a/t_b (seconds)",
                            "n_chunks": n_chunks,
                            "edges_s": [float(x) for x in edges.tolist()],
                            "n_boot": int(n_boot),
                            "seed": int(seed),
                        },
                    },
                    "natural_window": cov_obj.get("natural_window") or {},
                    "window_sweep": {
                        "supported": True,
                        "param_name": "window_ns",
                        "values": windows_ns,
                        "statistic_name": "CHSH |S| (fixed variant)",
                        "mean": mean_w,
                        "cov": _matrix_to_json(cov_boot_w),
                        "diag_sigma": [
                            float(math.sqrt(float(v))) if math.isfinite(float(v)) and float(v) >= 0 else None
                            for v in np.diag(cov_boot_w).tolist()
                        ],
                    },
                    "offset_sweep": {
                        "supported": True,
                        "param_name": "delta_offset_ns",
                        "values": [float(row.get("delta_offset_ns")) for row in orows],
                        "statistic_name": "CHSH |S| (fixed variant; ref window)",
                        "mean": mean_o,
                        "cov": _matrix_to_json(cov_boot_o),
                    },
                }

        # 条件分岐: 前段条件が不成立で、`ds.startswith("delft_")` を追加評価する。
        elif ds.startswith("delft_"):
            o_path = ds_dir / "offset_sweep_metrics.json"
            # 条件分岐: `o_path.exists()` を満たす経路を評価する。
            if o_path.exists():
                oj = json.loads(o_path.read_text(encoding="utf-8"))
                orows = list(oj.get("rows") or [])
                # 条件分岐: `orows and ("S_err" in orows[0] or "S_combined_err" in orows[0])` を満たす経路を評価する。
                if orows and ("S_err" in orows[0] or "S_combined_err" in orows[0]):
                    # 条件分岐: `"S_err" in orows[0]` を満たす経路を評価する。
                    if "S_err" in orows[0]:
                        sig = np.asarray([float(row.get("S_err")) for row in orows], dtype=float)
                        stat = "CHSH S (event-ready trials)"
                    else:
                        sig = np.asarray([float(row.get("S_combined_err")) for row in orows], dtype=float)
                        stat = "CHSH S (combined; event-ready trials)"

                    cov = np.diag(sig**2)
                    cov_obj = {
                        "generated_utc": _utc_now(),
                        "dataset_id": ds,
                        "supported": True,
                        "method": {"offset_sweep": {"name": "diag_from_reported_err"}},
                        "natural_window": {"recommended_start_offset_ps": 0},
                        "window_sweep": {"supported": False, "reason": "trial log (event-ready); no coincidence pairing step"},
                        "offset_sweep": {
                            "supported": True,
                            "param_name": "start_offset_ps",
                            "values": [int(row.get("start_offset_ps")) for row in orows],
                            "statistic_name": stat,
                            "cov": _matrix_to_json(cov),
                        },
                    }

                    try:
                        n_boot = int(n_bootstrap)
                        seed = _stable_seed("bell", "cov_boot", ds, "delft", "offset_sweep")
                        rng = np.random.default_rng(seed)

                        # 条件分岐: `"S_err" in orows[0]` を満たす経路を評価する。
                        if "S_err" in orows[0]:
                            mu = np.asarray([float(row.get("S")) for row in orows], dtype=float)
                            sig = np.asarray([float(row.get("S_err")) for row in orows], dtype=float)
                        else:
                            mu = np.asarray([float(row.get("S_combined")) for row in orows], dtype=float)
                            sig = np.asarray([float(row.get("S_combined_err")) for row in orows], dtype=float)

                        samples = rng.normal(loc=mu[None, :], scale=sig[None, :], size=(n_boot, mu.size))
                        cov_boot = _nan_cov(samples)
                        cov_boot_obj = {
                            "generated_utc": _utc_now(),
                            "dataset_id": ds,
                            "supported": True,
                            "method": {
                                "offset_sweep": {
                                    "name": "gaussian_bootstrap_diag",
                                    "n_boot": int(n_boot),
                                    "seed": int(seed),
                                    "note": "Bootstrap uses Normal(mean=reported, sigma=reported_err) per sweep point; point-to-point covariance is set to 0 by construction.",
                                }
                            },
                            "natural_window": {"recommended_start_offset_ps": 0},
                            "window_sweep": {
                                "supported": False,
                                "reason": "trial log (event-ready); no coincidence-window pairing step",
                            },
                            "offset_sweep": {
                                "supported": True,
                                "param_name": "start_offset_ps",
                                "values": [int(row.get("start_offset_ps")) for row in orows],
                                "statistic_name": stat,
                                "mean": mu.tolist(),
                                "cov": _matrix_to_json(cov_boot),
                                "diag_sigma": sig.tolist(),
                            },
                        }
                    except Exception as e:
                        cov_boot_obj = {"generated_utc": _utc_now(), "dataset_id": ds, "supported": False, "reason": str(e)}

        # 条件分岐: 前段条件が不成立で、`ds.startswith("nist_")` を追加評価する。
        elif ds.startswith("nist_"):
            w_path = ds_dir / "window_sweep_metrics.json"
            trial_counts_path = ds_dir / "trial_based_counts.json"
            # 条件分岐: `w_path.exists() and trial_counts_path.exists()` を満たす経路を評価する。
            if w_path.exists() and trial_counts_path.exists():
                wj = json.loads(w_path.read_text(encoding="utf-8"))
                trial = json.loads(trial_counts_path.read_text(encoding="utf-8"))
                counts = trial.get("counts") or {}
                n_trials = np.asarray(counts.get("trials_by_setting_pair") or [[0, 0], [0, 0]], dtype=np.float64)
                n_trials_a = np.asarray(counts.get("alice_trials_by_setting") or [0, 0], dtype=np.float64)
                n_trials_b = np.asarray(counts.get("bob_trials_by_setting") or [0, 0], dtype=np.float64)
                n_click_a = np.asarray(counts.get("alice_clicks_by_setting") or [0, 0], dtype=np.float64)
                n_click_b = np.asarray(counts.get("bob_clicks_by_setting") or [0, 0], dtype=np.float64)

                rows = list(wj.get("rows") or [])
                coinc: list[np.ndarray] = []
                windows_ns: list[float] = []
                for row in rows:
                    c = row.get("coinc_by_setting_pair")
                    # 条件分岐: `not isinstance(c, list)` を満たす経路を評価する。
                    if not isinstance(c, list):
                        coinc = []
                        break

                    try:
                        c_arr = np.asarray(c, dtype=np.float64)
                        # 条件分岐: `c_arr.shape != (2, 2)` を満たす経路を評価する。
                        if c_arr.shape != (2, 2):
                            raise ValueError

                        coinc.append(c_arr)
                        windows_ns.append(float(row.get("window_ns")))
                    except Exception:
                        coinc = []
                        break

                # 条件分岐: `coinc` を満たす経路を評価する。

                if coinc:
                    n_points = len(coinc)
                    cov = np.zeros((n_points, n_points), dtype=float)
                    w = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=float)  # J(a1=0,b1=0): + + + -
                    denom2 = np.where(n_trials > 0, n_trials**2, float("nan"))

                    a1 = int(((wj.get("trial_based") or {}).get("a1") or 0))
                    b1 = int(((wj.get("trial_based") or {}).get("b1") or 0))
                    pa1 = float(n_click_a[a1] / max(1.0, float(n_trials_a[a1])))
                    pb1 = float(n_click_b[b1] / max(1.0, float(n_trials_b[b1])))
                    var_pa1 = pa1 * (1.0 - pa1) / max(1.0, float(n_trials_a[a1]))
                    var_pb1 = pb1 * (1.0 - pb1) / max(1.0, float(n_trials_b[b1]))

                    for i in range(n_points):
                        for j in range(n_points):
                            cmin = coinc[min(i, j)]
                            term = np.nansum((w**2) * cmin / denom2)
                            cov[i, j] = float(term + var_pa1 + var_pb1)

                    cov_obj = {
                        "generated_utc": _utc_now(),
                        "dataset_id": ds,
                        "supported": True,
                        "method": {"window_sweep": {"name": "nested_poisson_increments"}, "offset_sweep": {"name": "diag_from_bootstrap"}},
                        "natural_window": {},
                        "window_sweep": {
                            "supported": True,
                            "param_name": "window_ns",
                            "values": windows_ns,
                            "statistic_name": "CH J_prob (A1=0,B1=0)",
                            "cov": _matrix_to_json(cov),
                        },
                    }
                    rec_w, rec_m = _recommend_natural_window(
                        dataset_id=ds,
                        ds_dir=ds_dir,
                        rows=list(rows),
                        plateau_fraction=0.99,
                    )
                    cov_obj["natural_window"] = {"method": rec_m, "recommended_window_ns": rec_w}

                    o_path = ds_dir / "offset_sweep_metrics.json"
                    # 条件分岐: `o_path.exists()` を満たす経路を評価する。
                    if o_path.exists():
                        oj = json.loads(o_path.read_text(encoding="utf-8"))
                        orows = list(oj.get("rows") or [])
                        sig = _finite([row.get("J_sigma_boot") for row in orows])
                        # 条件分岐: `sig and len(sig) == len(orows)` を満たす経路を評価する。
                        if sig and len(sig) == len(orows):
                            cov_o = np.diag(np.asarray(sig, dtype=float) ** 2)
                            cov_obj["offset_sweep"] = {
                                "supported": True,
                                "param_name": "delta_offset_ns",
                                "values": [float(row.get("delta_offset_ns")) for row in orows],
                                "statistic_name": "CH J_prob (subset; offset sweep)",
                                "cov": _matrix_to_json(cov_o),
                            }
                        else:
                            cov_obj["offset_sweep"] = {"supported": False, "reason": "missing J_sigma_boot"}

                    try:
                        c_stack = np.asarray(coinc, dtype=np.int64)
                        # 条件分岐: `c_stack.ndim != 3 or c_stack.shape[1:] != (2, 2)` を満たす経路を評価する。
                        if c_stack.ndim != 3 or c_stack.shape[1:] != (2, 2):
                            raise ValueError("invalid coinc_by_setting_pair shape")

                        inc = np.empty_like(c_stack)
                        inc[0] = c_stack[0]
                        inc[1:] = c_stack[1:] - c_stack[:-1]
                        # 条件分岐: `np.any(inc < 0)` を満たす経路を評価する。
                        if np.any(inc < 0):
                            raise ValueError("non-monotone coincidence counts; cannot bootstrap nested increments")

                        n_boot = int(n_bootstrap)
                        seed = _stable_seed("bell", "cov_boot", ds, "nist", "window_sweep")
                        rng = np.random.default_rng(seed)
                        inc_boot = rng.poisson(lam=inc, size=(n_boot,) + inc.shape).astype(np.float64, copy=False)
                        c_boot = np.cumsum(inc_boot, axis=1)

                        pa1 = float(n_click_a[a1] / max(1.0, float(n_trials_a[a1])))
                        pb1 = float(n_click_b[b1] / max(1.0, float(n_trials_b[b1])))
                        ca1 = rng.binomial(int(n_trials_a[a1]), min(1.0, max(0.0, pa1)), size=n_boot).astype(float)
                        cb1 = rng.binomial(int(n_trials_b[b1]), min(1.0, max(0.0, pb1)), size=n_boot).astype(float)
                        p_a1 = ca1 / max(1.0, float(n_trials_a[a1]))
                        p_b1 = cb1 / max(1.0, float(n_trials_b[b1]))

                        w_sign = np.zeros((2, 2), dtype=float)
                        a2 = 1 - a1
                        b2 = 1 - b1
                        w_sign[a1, b1] = 1.0
                        w_sign[a1, b2] = 1.0
                        w_sign[a2, b1] = 1.0
                        w_sign[a2, b2] = -1.0

                        denom = np.where(n_trials > 0, n_trials, float("nan"))
                        j_samples = np.nansum(w_sign[None, None, :, :] * c_boot / denom[None, None, :, :], axis=(2, 3))
                        j_samples = j_samples - p_a1[:, None] - p_b1[:, None]
                        cov_boot_w = _nan_cov(j_samples)

                        cov_boot_obj = {
                            "generated_utc": _utc_now(),
                            "dataset_id": ds,
                            "supported": True,
                            "method": {
                                "window_sweep": {
                                    "name": "parametric_bootstrap_nested_poisson_increments",
                                    "n_boot": int(n_boot),
                                    "seed": int(seed),
                                    "note": "Poisson bootstrap on nested coincidence-count increments; singles P(A1),P(B1) are binomial-bootstrapped from trial-based counts.",
                                },
                                "offset_sweep": {"name": "diag_from_bootstrap"},
                            },
                            "natural_window": cov_obj.get("natural_window") or {},
                            "window_sweep": {
                                "supported": True,
                                "param_name": "window_ns",
                                "values": windows_ns,
                                "statistic_name": "CH J_prob (A1=0,B1=0)",
                                "mean": [float(row.get("J_prob")) for row in rows],
                                "cov": _matrix_to_json(cov_boot_w),
                                "diag_sigma": [
                                    float(math.sqrt(float(v))) if math.isfinite(float(v)) and float(v) >= 0 else None
                                    for v in np.diag(cov_boot_w).tolist()
                                ],
                            },
                        }

                        # 条件分岐: `o_path.exists()` を満たす経路を評価する。
                        if o_path.exists():
                            oj = json.loads(o_path.read_text(encoding="utf-8"))
                            orows = list(oj.get("rows") or [])
                            sig = _finite([row.get("J_sigma_boot") for row in orows])
                            # 条件分岐: `sig and len(sig) == len(orows)` を満たす経路を評価する。
                            if sig and len(sig) == len(orows):
                                cov_o = np.diag(np.asarray(sig, dtype=float) ** 2)
                                cov_boot_obj["offset_sweep"] = {
                                    "supported": True,
                                    "param_name": "delta_offset_ns",
                                    "values": [float(row.get("delta_offset_ns")) for row in orows],
                                    "statistic_name": "CH J_prob (subset; offset sweep)",
                                    "mean": _finite([row.get("J_prob") for row in orows]),
                                    "cov": _matrix_to_json(cov_o),
                                }
                            else:
                                cov_boot_obj["offset_sweep"] = {"supported": False, "reason": "missing J_sigma_boot"}
                    except Exception as e:
                        cov_boot_obj = {"generated_utc": _utc_now(), "dataset_id": ds, "supported": False, "reason": str(e)}

        # 条件分岐: 前段条件が不成立で、`ds.startswith("kwiat2013_")` を追加評価する。
        elif ds.startswith("kwiat2013_"):
            w_path = ds_dir / "window_sweep_metrics.json"
            trial_counts_path = ds_dir / "trial_based_counts.json"
            # 条件分岐: `w_path.exists() and trial_counts_path.exists()` を満たす経路を評価する。
            if w_path.exists() and trial_counts_path.exists():
                wj = json.loads(w_path.read_text(encoding="utf-8"))
                rows = list(wj.get("rows") or [])
                windows_ns: list[float] = []
                sig: list[float] = []
                for row in rows:
                    try:
                        windows_ns.append(float(row.get("window_ns")))
                        sig.append(float(row.get("J_sigma_boot")))
                    except Exception:
                        windows_ns = []
                        sig = []
                        break

                # 条件分岐: `windows_ns and sig and len(windows_ns) == len(sig)` を満たす経路を評価する。

                if windows_ns and sig and len(windows_ns) == len(sig):
                    cov = np.diag(np.asarray(sig, dtype=float) ** 2)
                    rec_w, rec_m = _recommend_natural_window(
                        dataset_id=ds,
                        ds_dir=ds_dir,
                        rows=list(rows),
                        plateau_fraction=0.99,
                    )
                    cov_obj = {
                        "generated_utc": _utc_now(),
                        "dataset_id": ds,
                        "supported": True,
                        "method": {"window_sweep": {"name": "diag_from_bootstrap"}, "offset_sweep": {"name": "n/a"}},
                        "natural_window": {"method": rec_m, "recommended_window_ns": rec_w},
                        "window_sweep": {
                            "supported": True,
                            "param_name": "window_ns",
                            "values": windows_ns,
                            "statistic_name": "CH J_prob (A1=0,B1=0)",
                            "cov": _matrix_to_json(cov),
                        },
                        "offset_sweep": {"supported": False, "reason": "no offset sweep for PC-triggered trials"},
                    }

                    try:
                        trial = json.loads(trial_counts_path.read_text(encoding="utf-8"))
                        counts = trial.get("counts") or {}
                        n_trials = np.asarray(counts.get("trials_by_setting_pair") or [[0, 0], [0, 0]], dtype=np.float64)
                        n_trials_a = np.asarray(counts.get("alice_trials_by_setting") or [0, 0], dtype=np.float64)
                        n_trials_b = np.asarray(counts.get("bob_trials_by_setting") or [0, 0], dtype=np.float64)

                        a1 = int(((wj.get("trial_based") or {}).get("a1") or 0))
                        b1 = int(((wj.get("trial_based") or {}).get("b1") or 0))

                        c_list: list[np.ndarray] = []
                        ca_list: list[np.ndarray] = []
                        cb_list: list[np.ndarray] = []
                        j_mean: list[float] = []
                        for row in rows:
                            c = np.asarray(row.get("coinc_by_setting_pair"), dtype=np.int64)
                            ca = np.asarray(row.get("click_a_by_setting"), dtype=np.int64)
                            cb = np.asarray(row.get("click_b_by_setting"), dtype=np.int64)
                            # 条件分岐: `c.shape != (2, 2) or ca.shape != (2,) or cb.shape != (2,)` を満たす経路を評価する。
                            if c.shape != (2, 2) or ca.shape != (2,) or cb.shape != (2,):
                                raise ValueError("missing click/coinc counts; regenerate window_sweep_metrics.json")

                            c_list.append(c)
                            ca_list.append(ca)
                            cb_list.append(cb)
                            j_mean.append(float(row.get("J_prob")))

                        c_stack = np.stack(c_list, axis=0).astype(np.int64, copy=False)
                        ca_stack = np.stack(ca_list, axis=0).astype(np.int64, copy=False)
                        cb_stack = np.stack(cb_list, axis=0).astype(np.int64, copy=False)

                        inc_c = np.empty_like(c_stack)
                        inc_ca = np.empty_like(ca_stack)
                        inc_cb = np.empty_like(cb_stack)
                        inc_c[0] = c_stack[0]
                        inc_ca[0] = ca_stack[0]
                        inc_cb[0] = cb_stack[0]
                        inc_c[1:] = c_stack[1:] - c_stack[:-1]
                        inc_ca[1:] = ca_stack[1:] - ca_stack[:-1]
                        inc_cb[1:] = cb_stack[1:] - cb_stack[:-1]
                        # 条件分岐: `np.any(inc_c < 0) or np.any(inc_ca < 0) or np.any(inc_cb < 0)` を満たす経路を評価する。
                        if np.any(inc_c < 0) or np.any(inc_ca < 0) or np.any(inc_cb < 0):
                            raise ValueError("non-monotone counts; cannot bootstrap nested increments")

                        n_boot = int(n_bootstrap)
                        seed = _stable_seed("bell", "cov_boot", ds, "kwiat", "window_sweep")
                        rng = np.random.default_rng(seed)
                        inc_c_boot = rng.poisson(lam=inc_c, size=(n_boot,) + inc_c.shape).astype(np.float64, copy=False)
                        inc_ca_boot = rng.poisson(lam=inc_ca, size=(n_boot,) + inc_ca.shape).astype(np.float64, copy=False)
                        inc_cb_boot = rng.poisson(lam=inc_cb, size=(n_boot,) + inc_cb.shape).astype(np.float64, copy=False)
                        c_boot = np.cumsum(inc_c_boot, axis=1)
                        ca_boot = np.cumsum(inc_ca_boot, axis=1)
                        cb_boot = np.cumsum(inc_cb_boot, axis=1)

                        w_sign = np.zeros((2, 2), dtype=float)
                        a2 = 1 - a1
                        b2 = 1 - b1
                        w_sign[a1, b1] = 1.0
                        w_sign[a1, b2] = 1.0
                        w_sign[a2, b1] = 1.0
                        w_sign[a2, b2] = -1.0

                        denom = np.where(n_trials > 0, n_trials, float("nan"))
                        j_samples = np.nansum(w_sign[None, None, :, :] * c_boot / denom[None, None, :, :], axis=(2, 3))
                        p_a1 = ca_boot[:, :, a1] / max(1.0, float(n_trials_a[a1]))
                        p_b1 = cb_boot[:, :, b1] / max(1.0, float(n_trials_b[b1]))
                        j_samples = j_samples - p_a1 - p_b1
                        cov_boot_w = _nan_cov(j_samples)

                        cov_boot_obj = {
                            "generated_utc": _utc_now(),
                            "dataset_id": ds,
                            "supported": True,
                            "method": {
                                "window_sweep": {
                                    "name": "parametric_bootstrap_nested_poisson_increments",
                                    "n_boot": int(n_boot),
                                    "seed": int(seed),
                                    "note": "Poisson bootstrap on nested increments of (click/coinc) counts; valid for rare clicks (binomial≈Poisson).",
                                },
                                "offset_sweep": {"name": "n/a"},
                            },
                            "natural_window": cov_obj.get("natural_window") or {},
                            "window_sweep": {
                                "supported": True,
                                "param_name": "window_ns",
                                "values": windows_ns,
                                "statistic_name": "CH J_prob (A1=0,B1=0)",
                                "mean": j_mean,
                                "cov": _matrix_to_json(cov_boot_w),
                                "diag_sigma": [
                                    float(math.sqrt(float(v))) if math.isfinite(float(v)) and float(v) >= 0 else None
                                    for v in np.diag(cov_boot_w).tolist()
                                ],
                            },
                            "offset_sweep": {"supported": False, "reason": "no offset sweep for PC-triggered trials"},
                        }
                    except Exception as e:
                        cov_boot_obj = {"generated_utc": _utc_now(), "dataset_id": ds, "supported": False, "reason": str(e)}

        nw_frozen_path = ds_dir / "natural_window_frozen.json"
        nw_frozen_obj: dict[str, Any] = {"generated_utc": _utc_now(), "dataset_id": ds, "supported": False}
        try:
            nat = cov_boot_obj.get("natural_window") or cov_obj.get("natural_window") or {}

            # 条件分岐: `isinstance(nat, dict) and "recommended_start_offset_ps" in nat` を満たす経路を評価する。
            if isinstance(nat, dict) and "recommended_start_offset_ps" in nat:
                vals = (cov_boot_obj.get("offset_sweep") or {}).get("values") or (cov_obj.get("offset_sweep") or {}).get(
                    "values"
                )
                means = (cov_boot_obj.get("offset_sweep") or {}).get("mean")
                sigmas = (cov_boot_obj.get("offset_sweep") or {}).get("diag_sigma")
                # 条件分岐: `not isinstance(vals, list) or not vals` を満たす経路を評価する。
                if not isinstance(vals, list) or not vals:
                    raise ValueError("missing offset_sweep.values")

                want = int(nat.get("recommended_start_offset_ps"))
                vv = np.asarray([int(x) for x in vals], dtype=int)
                idx = int(np.argmin(np.abs(vv - want)))
                frozen_off = int(vv[idx])
                frozen_stat = float(means[idx]) if isinstance(means, list) and idx < len(means) else None
                frozen_sigma = float(sigmas[idx]) if isinstance(sigmas, list) and idx < len(sigmas) else None
                nw_frozen_obj = {
                    "generated_utc": _utc_now(),
                    "dataset_id": ds,
                    "supported": True,
                    "natural_window": {"recommended_start_offset_ps": int(want), "frozen_start_offset_ps": int(frozen_off)},
                    "baseline": {
                        "param_name": "start_offset_ps",
                        "value": int(frozen_off),
                        "statistic_name": (cov_boot_obj.get("offset_sweep") or {}).get("statistic_name")
                        or (cov_obj.get("offset_sweep") or {}).get("statistic_name"),
                        "statistic": frozen_stat,
                        "sigma_boot": frozen_sigma,
                    },
                    "reject": {
                        "threshold_sigma": 3.0,
                        "rule": "abs(stat - frozen_stat) >= threshold_sigma * sigma_boot",
                    },
                    "sources": {
                        "covariance_bootstrap_json": str(cov_boot_path),
                        "offset_sweep_metrics_json": str(ds_dir / "offset_sweep_metrics.json"),
                    },
                }
            else:
                rec_w = None
                # 条件分岐: `isinstance(nat, dict)` を満たす経路を評価する。
                if isinstance(nat, dict):
                    rec_w = nat.get("recommended_window_ns")

                vals = (cov_boot_obj.get("window_sweep") or {}).get("values") or (cov_obj.get("window_sweep") or {}).get(
                    "values"
                )
                means = (cov_boot_obj.get("window_sweep") or {}).get("mean")
                sigmas = (cov_boot_obj.get("window_sweep") or {}).get("diag_sigma")
                # 条件分岐: `not isinstance(vals, list) or not vals` を満たす経路を評価する。
                if not isinstance(vals, list) or not vals:
                    raise ValueError("missing window_sweep.values")

                grid = [float(x) for x in vals]
                frozen_w = _snap_to_grid_ge(x=float(rec_w) if rec_w is not None else None, grid=grid) or float(grid[0])
                vv = np.asarray(grid, dtype=float)
                idx = int(np.argmin(np.abs(vv - float(frozen_w))))
                frozen_w2 = float(vv[idx])
                frozen_stat = float(means[idx]) if isinstance(means, list) and idx < len(means) and means[idx] is not None else None
                frozen_sigma = float(sigmas[idx]) if isinstance(sigmas, list) and idx < len(sigmas) and sigmas[idx] is not None else None
                nw_frozen_obj = {
                    "generated_utc": _utc_now(),
                    "dataset_id": ds,
                    "supported": True,
                    "natural_window": {
                        "recommended_window_ns": float(rec_w) if rec_w is not None else None,
                        "frozen_window_ns": float(frozen_w2),
                        "snap_rule": "min(window in grid where window >= recommended_window_ns); fallback: min grid",
                    },
                    "baseline": {
                        "param_name": "window_ns",
                        "value": float(frozen_w2),
                        "statistic_name": (cov_boot_obj.get("window_sweep") or {}).get("statistic_name")
                        or (cov_obj.get("window_sweep") or {}).get("statistic_name"),
                        "statistic": frozen_stat,
                        "sigma_boot": frozen_sigma,
                    },
                    "reject": {
                        "threshold_sigma": 3.0,
                        "rule": "abs(stat - frozen_stat) >= threshold_sigma * sigma_boot",
                    },
                    "sources": {
                        "covariance_bootstrap_json": str(cov_boot_path),
                        "window_sweep_metrics_json": str(ds_dir / "window_sweep_metrics.json"),
                    },
                }
        except Exception as e:
            nw_frozen_obj["reason"] = str(e)

        _attach_cov_diagnostics(cov_obj)
        _attach_cov_diagnostics(cov_boot_obj)
        sweep_profile = _extract_sweep_profile(dataset_id=ds, cov_obj=cov_obj, cov_boot_obj=cov_boot_obj)

        _write_json(nw_frozen_path, nw_frozen_obj)
        _write_json(cov_path, cov_obj)
        _write_json(cov_boot_path, cov_boot_obj)
        cov_index.append(
            {
                "dataset_id": ds,
                "supported": bool(cov_obj.get("supported")),
                "covariance_json": str(cov_path),
                "covariance_bootstrap_json": str(cov_boot_path),
                "natural_window_frozen_json": str(nw_frozen_path),
                "sweep_profile": sweep_profile,
            }
        )

    # --- longterm/systematics inputs

    fc_path = OUT_BASE / "falsification_pack.json"
    fc = json.loads(fc_path.read_text(encoding="utf-8")) if fc_path.exists() else {}
    thresholds = fc.get("thresholds") if isinstance(fc.get("thresholds"), dict) else {}
    ratio_th = float(thresholds.get("selection_origin_ratio_min", 1.0))
    delay_z_th = float(thresholds.get("delay_signature_z_min", 3.0))

    # --- systematics decomposition (Step 7.16.8)
    sys15 = _write_systematics_decomposition_15items(results=results, cov_index=cov_index, falsification_pack=fc)

    # --- systematics templates
    systematics_templates = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 7, "step": "7.16.8", "name": "Bell: covariance and systematics templates"},
        "main_script": {"path": "scripts/quantum/bell_primary_products.py", "repro": "python -B scripts/quantum/bell_primary_products.py"},
        "datasets": cov_index,
        "systematics_15items": {
            "json": "output/public/quantum/bell/systematics_decomposition_15items.json",
            "csv": "output/public/quantum/bell/systematics_decomposition_15items.csv",
            "png": "output/public/quantum/bell/systematics_decomposition_15items.png" if bool(sys15.get("plot_written")) else None,
        },
        "outputs": {
            "systematics_templates_json": "output/public/quantum/bell/systematics_templates.json",
            "systematics_15items_json": "output/public/quantum/bell/systematics_decomposition_15items.json",
            "systematics_15items_csv": "output/public/quantum/bell/systematics_decomposition_15items.csv",
            "systematics_15items_png": "output/public/quantum/bell/systematics_decomposition_15items.png"
            if bool(sys15.get("plot_written"))
            else None,
        },
    }
    _write_json(OUT_BASE / "systematics_templates.json", systematics_templates)

    # --- longterm consistency (cross-dataset integration)

    def _ks_max_legacy(x: Any) -> float | None:
        # 条件分岐: `not isinstance(x, dict)` を満たす経路を評価する。
        if not isinstance(x, dict):
            return None

        vals = _finite(x.values())
        return float(max(vals)) if vals else None

    datasets_longterm: list[dict[str, Any]] = []
    for d in fc.get("datasets", []) if isinstance(fc.get("datasets"), list) else []:
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        ds_id = str(d.get("dataset_id") or "")
        ratio = None
        try:
            # 条件分岐: `d.get("ratio") is not None` を満たす経路を評価する。
            if d.get("ratio") is not None:
                ratio = float(d.get("ratio"))
                # 条件分岐: `not math.isfinite(ratio)` を満たす経路を評価する。
                if not math.isfinite(ratio):
                    ratio = None
        except Exception:
            ratio = None

        delay_z = _delay_signature_z_max(d.get("delay_signature"))
        ks_max = _ks_max_legacy(d.get("ks_delay"))

        # Load frozen "natural window" baseline (operational freeze point).
        nw_obj = None
        for ci in cov_index:
            # 条件分岐: `str(ci.get("dataset_id") or "") != ds_id` を満たす経路を評価する。
            if str(ci.get("dataset_id") or "") != ds_id:
                continue

            p = Path(str(ci.get("natural_window_frozen_json") or ""))
            # 条件分岐: `p.exists()` を満たす経路を評価する。
            if p.exists():
                try:
                    nw_obj = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    nw_obj = None

            break

        ratio_pass = None if ratio is None else bool(ratio >= ratio_th)
        delay_pass = None if delay_z is None else bool(delay_z >= delay_z_th)
        datasets_longterm.append(
            {
                "dataset_id": ds_id,
                "display_name": _dataset_display_name(ds_id),
                "year": _dataset_year(ds_id),
                "selection_class": _dataset_selection_class(ds_id),
                "statistic_family": _dataset_statistic_family(ds_id),
                "selection_knob": d.get("selection_knob"),
                "ratio": ratio,
                "ratio_threshold": ratio_th,
                "ratio_pass": ratio_pass,
                "delay_z_max": delay_z,
                "delay_z_threshold": delay_z_th,
                "delay_z_pass": delay_pass,
                "ks_delay_max_legacy": ks_max,
                "natural_window_frozen": (
                    {
                        "natural_window": (nw_obj.get("natural_window") if isinstance(nw_obj, dict) else None),
                        "baseline": (nw_obj.get("baseline") if isinstance(nw_obj, dict) else None),
                        "reject": (nw_obj.get("reject") if isinstance(nw_obj, dict) else None),
                    }
                    if isinstance(nw_obj, dict)
                    else None
                ),
            }
        )

    ratios_all = [float(d["ratio"]) for d in datasets_longterm if d.get("ratio") is not None]
    ratio_min = float(min(ratios_all)) if ratios_all else None
    ratio_med = float(np.nanmedian(np.asarray(ratios_all, dtype=float))) if ratios_all else None

    # Focus "delay" longterm on fast-switching datasets (operational class by id prefix).
    zs_fast = [
        float(d.get("delay_z_max"))
        for d in datasets_longterm
        if d.get("delay_z_max") is not None and str(d.get("dataset_id") or "").startswith(("weihs1998_", "nist_"))
    ]
    delay_z_fast_min = float(min(zs_fast)) if zs_fast else None

    ratio_trend = _linear_trend_metrics((d.get("year"), d.get("ratio")) for d in datasets_longterm)
    delay_trend = _linear_trend_metrics((d.get("year"), d.get("delay_z_max")) for d in datasets_longterm)
    ratio_by_selection_class = _group_metric_summary(
        datasets_longterm,
        group_key="selection_class",
        value_key="ratio",
    )
    ratio_by_stat_family = _group_metric_summary(
        datasets_longterm,
        group_key="statistic_family",
        value_key="ratio",
    )
    delay_by_selection_class = _group_metric_summary(
        datasets_longterm,
        group_key="selection_class",
        value_key="delay_z_max",
    )
    delay_by_stat_family = _group_metric_summary(
        datasets_longterm,
        group_key="statistic_family",
        value_key="delay_z_max",
    )

    # --- cross-dataset covariance from sweep profiles (normalized knob u in [0,1])
    cross_cov_path = OUT_BASE / "cross_dataset_covariance.json"
    cross_cov_png = OUT_BASE / "cross_dataset_covariance.png"
    cross_cov_obj: dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 7, "step": "7.16.7", "name": "Bell: cross-dataset covariance (sweep profiles)"},
        "supported": False,
        "reason": None,
        "method": {
            "profile": "Interpolate each dataset sweep mean onto common u-grid in [0,1], then compute dataset x dataset covariance/correlation.",
            "u_grid_points": int(cross_n_u),
            "bootstrap": {"n_boot": int(n_bootstrap), "seed": None},
            "jackknife": {"type": "leave-one-u-grid-point-out", "n_delete": int(cross_n_u)},
        },
        "datasets": [],
        "matrices": {},
        "outputs": {
            "cross_dataset_covariance_json": "output/public/quantum/bell/cross_dataset_covariance.json",
            "cross_dataset_covariance_png": "output/public/quantum/bell/cross_dataset_covariance.png",
        },
    }
    profile_entries: list[dict[str, Any]] = []
    for item in cov_index:
        # 条件分岐: `not isinstance(item, dict)` を満たす経路を評価する。
        if not isinstance(item, dict):
            continue

        prof = item.get("sweep_profile")
        # 条件分岐: `not isinstance(prof, dict)` を満たす経路を評価する。
        if not isinstance(prof, dict):
            continue

        x_norm = prof.get("x_norm")
        y_vals = prof.get("y")
        # 条件分岐: `not isinstance(x_norm, list) or not isinstance(y_vals, list) or len(x_norm) !...` を満たす経路を評価する。
        if not isinstance(x_norm, list) or not isinstance(y_vals, list) or len(x_norm) != len(y_vals) or len(x_norm) < 3:
            continue

        xs = np.asarray([float(v) for v in x_norm], dtype=float)
        ys = np.asarray([float(v) for v in y_vals], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        # 条件分岐: `int(np.sum(mask)) < 3` を満たす経路を評価する。
        if int(np.sum(mask)) < 3:
            continue

        xs = xs[mask]
        ys = ys[mask]
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        unique_x, unique_idx = np.unique(xs, return_index=True)
        ys_unique = ys[unique_idx]
        # 条件分岐: `unique_x.size < 3` を満たす経路を評価する。
        if unique_x.size < 3:
            continue

        profile_entries.append(
            {
                "dataset_id": str(item.get("dataset_id") or ""),
                "display_name": _dataset_display_name(str(item.get("dataset_id") or "")),
                "source": str(prof.get("source") or ""),
                "sweep_key": str(prof.get("sweep_key") or ""),
                "param_name": str(prof.get("param_name") or ""),
                "n_points": int(unique_x.size),
                "param_min": prof.get("param_min"),
                "param_max": prof.get("param_max"),
                "x_norm": unique_x,
                "y": ys_unique,
            }
        )

    cross_png_written = False
    # 条件分岐: `len(profile_entries) >= 2` を満たす経路を評価する。
    if len(profile_entries) >= 2:
        u_grid = np.linspace(0.0, 1.0, int(cross_n_u), dtype=float)
        profile_matrix: list[np.ndarray] = []
        ds_labels: list[str] = []
        ds_ids: list[str] = []
        for prof in profile_entries:
            profile_matrix.append(np.interp(u_grid, np.asarray(prof["x_norm"], dtype=float), np.asarray(prof["y"], dtype=float)))
            ds_labels.append(str(prof.get("display_name") or prof.get("dataset_id") or ""))
            ds_ids.append(str(prof.get("dataset_id") or ""))

        profile_arr = np.asarray(profile_matrix, dtype=float)

        cov_profile = np.cov(profile_arr, ddof=1)
        corr_profile = _corrcoef_rows(profile_arr)
        boot_seed = _stable_seed("bell", "cross_dataset_cov", "bootstrap")
        boot_corr_mean, boot_corr_sigma, boot_corr_cov_flat = _bootstrap_corrcoef_rows(
            profile_arr,
            n_boot=int(n_bootstrap),
            seed=int(boot_seed),
        )
        jack_corr_mean, jack_corr_sigma = _jackknife_corrcoef_rows(profile_arr)

        cross_cov_obj["supported"] = True
        cross_cov_obj["method"]["bootstrap"]["seed"] = int(boot_seed)
        cross_cov_obj["datasets"] = [
            {
                "dataset_id": str(prof.get("dataset_id") or ""),
                "display_name": str(prof.get("display_name") or ""),
                "source": str(prof.get("source") or ""),
                "sweep_key": str(prof.get("sweep_key") or ""),
                "param_name": str(prof.get("param_name") or ""),
                "n_points": int(prof.get("n_points") or 0),
                "param_min": prof.get("param_min"),
                "param_max": prof.get("param_max"),
            }
            for prof in profile_entries
        ]
        cross_cov_obj["matrices"] = {
            "dataset_order": ds_ids,
            "u_grid": [float(v) for v in u_grid.tolist()],
            "profile_cov": _matrix_to_json(cov_profile),
            "profile_cov_diag_sigma": _diag_sigma_from_cov(cov_profile),
            "profile_cov_eigen": _cov_eigen_summary(cov_profile),
            "profile_corr": _matrix_to_json(corr_profile),
            "profile_corr_eigen": _cov_eigen_summary(corr_profile),
            "profile_corr_bootstrap_mean": _matrix_to_json(boot_corr_mean),
            "profile_corr_bootstrap_sigma": _matrix_to_json(boot_corr_sigma),
            "profile_corr_bootstrap_cov_flat": _matrix_to_json(boot_corr_cov_flat),
            "profile_corr_bootstrap_cov_flat_eigen": _cov_eigen_summary(boot_corr_cov_flat),
            "profile_corr_jackknife_mean": _matrix_to_json(jack_corr_mean),
            "profile_corr_jackknife_sigma": _matrix_to_json(jack_corr_sigma),
        }

        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except Exception:
            cross_png_written = False
        else:
            fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), dpi=170)
            im0 = axes[0].imshow(cov_profile, cmap="viridis")
            axes[0].set_title("Sweep-profile covariance")
            axes[0].set_xticks(np.arange(len(ds_labels)), ds_labels, rotation=25, ha="right")
            axes[0].set_yticks(np.arange(len(ds_labels)), ds_labels)
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(corr_profile, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            axes[1].set_title("Sweep-profile correlation")
            axes[1].set_xticks(np.arange(len(ds_labels)), ds_labels, rotation=25, ha="right")
            axes[1].set_yticks(np.arange(len(ds_labels)), ds_labels)
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            eig_vals = []
            eig_obj = cross_cov_obj["matrices"].get("profile_cov_eigen")
            # 条件分岐: `isinstance(eig_obj, dict)` を満たす経路を評価する。
            if isinstance(eig_obj, dict):
                eig_vals = [float(v) for v in eig_obj.get("eigenvalues_desc") or [] if v is not None]

            axes[2].bar(np.arange(len(eig_vals)), eig_vals, color="tab:blue", alpha=0.85)
            axes[2].set_title("Covariance eigenvalues")
            axes[2].set_xlabel("mode index")
            axes[2].set_ylabel("eigenvalue")
            axes[2].grid(True, axis="y", alpha=0.3, ls=":")

            fig.suptitle("Bell cross-dataset covariance (sweep-profile, u-grid)", y=1.02)
            fig.tight_layout()
            fig.savefig(cross_cov_png, bbox_inches="tight")
            plt.close(fig)
            cross_png_written = True
    else:
        cross_cov_obj["reason"] = "need >=2 datasets with valid sweep profile means"

    _write_json(cross_cov_path, cross_cov_obj)

    cross_cov_summary: dict[str, Any] = {"supported": bool(cross_cov_obj.get("supported"))}
    # 条件分岐: `bool(cross_cov_obj.get("supported"))` を満たす経路を評価する。
    if bool(cross_cov_obj.get("supported")):
        mat = cross_cov_obj.get("matrices") if isinstance(cross_cov_obj.get("matrices"), dict) else {}
        eig = mat.get("profile_cov_eigen") if isinstance(mat, dict) and isinstance(mat.get("profile_cov_eigen"), dict) else {}
        eig_vals = [
            float(v)
            for v in (eig.get("eigenvalues_desc") if isinstance(eig, dict) else []) or []
            if v is not None and math.isfinite(float(v))
        ]
        explained = [
            float(v)
            for v in (eig.get("explained_ratio_positive") if isinstance(eig, dict) else []) or []
            if v is not None and math.isfinite(float(v))
        ]
        corr = _matrix_from_json(mat.get("profile_corr") if isinstance(mat, dict) else None)
        ds_order = mat.get("dataset_order") if isinstance(mat, dict) and isinstance(mat.get("dataset_order"), list) else []
        strongest_pair = None
        # 条件分岐: `corr.ndim == 2 and corr.shape[0] == corr.shape[1] and corr.shape[0] >= 2` を満たす経路を評価する。
        if corr.ndim == 2 and corr.shape[0] == corr.shape[1] and corr.shape[0] >= 2:
            best_abs = -1.0
            best_i = -1
            best_j = -1
            best_v = None
            for i in range(corr.shape[0]):
                for j in range(i + 1, corr.shape[1]):
                    v = float(corr[i, j])
                    # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。
                    if not math.isfinite(v):
                        continue

                    av = abs(v)
                    # 条件分岐: `av > best_abs` を満たす経路を評価する。
                    if av > best_abs:
                        best_abs = av
                        best_i = i
                        best_j = j
                        best_v = v

            # 条件分岐: `best_i >= 0 and best_j >= 0 and best_v is not None` を満たす経路を評価する。

            if best_i >= 0 and best_j >= 0 and best_v is not None:
                lhs = str(ds_order[best_i]) if best_i < len(ds_order) else str(best_i)
                rhs = str(ds_order[best_j]) if best_j < len(ds_order) else str(best_j)
                strongest_pair = {
                    "dataset_i": lhs,
                    "dataset_j": rhs,
                    "corr": float(best_v),
                    "abs_corr": float(abs(best_v)),
                }

        cross_cov_summary.update(
            {
                "rank_eps_1e-10": eig.get("rank_eps_1e-10") if isinstance(eig, dict) else None,
                "condition_number_abs": eig.get("condition_number_abs") if isinstance(eig, dict) else None,
                "largest_eigenvalue": float(eig_vals[0]) if eig_vals else None,
                "largest_mode_explained_ratio": float(explained[0]) if explained else None,
                "strongest_profile_corr_pair": strongest_pair,
            }
        )
    else:
        cross_cov_summary["reason"] = cross_cov_obj.get("reason")

    longterm: dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 7, "step": "7.16.9", "name": "Bell: longterm cross-dataset consistency quantification"},
        "inputs": {
            "falsification_pack_json": "output/public/quantum/bell/falsification_pack.json" if fc_path.exists() else None,
            "systematics_templates_json": "output/public/quantum/bell/systematics_templates.json",
            "systematics_15items_json": "output/public/quantum/bell/systematics_decomposition_15items.json",
            "cross_dataset_covariance_json": "output/public/quantum/bell/cross_dataset_covariance.json",
        },
        "thresholds": thresholds,
        "datasets": datasets_longterm,
        "summary": {
            "n_datasets": int(len(datasets_longterm)),
            "selection_ratio_min": ratio_min,
            "selection_ratio_median": ratio_med,
            "delay_z_fast_min": delay_z_fast_min,
            "note": "Pass/fail flags are operational and only mean consistency with the repository thresholds (not universal physics thresholds).",
        },
        "longterm_trends": {
            "selection_ratio_vs_year": ratio_trend,
            "delay_z_vs_year": delay_trend,
        },
        "condition_dependence": {
            "selection_ratio_by_selection_class": ratio_by_selection_class,
            "selection_ratio_by_statistic_family": ratio_by_stat_family,
            "delay_z_by_selection_class": delay_by_selection_class,
            "delay_z_by_statistic_family": delay_by_stat_family,
        },
        "cross_dataset_covariance_summary": cross_cov_summary,
        "covariance_index": cov_index,
        "outputs": {
            "longterm_consistency_json": "output/public/quantum/bell/longterm_consistency.json",
            "longterm_consistency_png": "output/public/quantum/bell/longterm_consistency.png",
            "systematics_15items_json": "output/public/quantum/bell/systematics_decomposition_15items.json",
            "systematics_15items_csv": "output/public/quantum/bell/systematics_decomposition_15items.csv",
            "systematics_15items_png": "output/public/quantum/bell/systematics_decomposition_15items.png"
            if bool(sys15.get("plot_written"))
            else None,
            "cross_dataset_covariance_json": "output/public/quantum/bell/cross_dataset_covariance.json",
            "cross_dataset_covariance_png": (
                "output/public/quantum/bell/cross_dataset_covariance.png" if bool(cross_png_written) else None
            ),
        },
    }
    loophole10 = _write_selection_loophole_quantification(longterm=longterm)
    longterm["outputs"]["selection_loophole_quantification_json"] = (
        "output/public/quantum/bell/selection_loophole_quantification.json"
    )
    longterm["outputs"]["selection_loophole_quantification_csv"] = (
        "output/public/quantum/bell/selection_loophole_quantification.csv"
    )
    longterm["outputs"]["selection_loophole_quantification_png"] = (
        "output/public/quantum/bell/selection_loophole_quantification.png" if bool(loophole10.get("plot_written")) else None
    )
    _write_json(OUT_BASE / "longterm_consistency.json", longterm)

    # Optional plot (ratio + delay z)
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        return

    # 条件分岐: `not datasets_longterm` を満たす経路を評価する。

    if not datasets_longterm:
        return

    labels = [str(d.get("display_name") or d.get("dataset_id") or "") for d in datasets_longterm]
    ratios = [float(d.get("ratio")) if d.get("ratio") is not None else float("nan") for d in datasets_longterm]
    zvals: list[float] = []
    zcolors: list[str] = []
    for d in datasets_longterm:
        z = d.get("delay_z_max")
        # 条件分岐: `z is None` を満たす経路を評価する。
        if z is None:
            zvals.append(0.0)
            zcolors.append("0.85")
        else:
            zvals.append(float(z))
            zcolors.append("tab:orange")

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.2), dpi=170)
    x = np.arange(len(labels))
    ax[0].bar(x, ratios, color="tab:blue", alpha=0.85)
    ax[0].axhline(ratio_th, color="0.2", ls="--", lw=1.0)
    ax[0].set_xticks(x, labels, rotation=0, ha="center")
    ax[0].set_ylabel("Δ(stat) / σ_stat (median)")
    ax[0].set_title("Selection sensitivity (ratio)")
    ax[0].grid(True, axis="y", alpha=0.3, ls=":")

    ax[1].bar(x, zvals, color=zcolors, alpha=0.9)
    ax[1].axhline(delay_z_th, color="0.2", ls="--", lw=1.0)
    ax[1].set_xticks(x, labels, rotation=0, ha="center")
    ax[1].set_ylabel("z = ∣Δmedian∣ / σ(Δmedian)")
    ax[1].set_title("Delay setting-dependence (Δmedian; z)")
    ax[1].grid(True, axis="y", alpha=0.3, ls=":")

    for i, d in enumerate(datasets_longterm):
        # 条件分岐: `d.get("delay_z_max") is None` を満たす経路を評価する。
        if d.get("delay_z_max") is None:
            ax[1].text(float(i), 0.15, "n/a", ha="center", va="bottom", fontsize=9, color="0.35")

    fig.suptitle("Bell longterm consistency (cross-dataset)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_BASE / "longterm_consistency.png", bbox_inches="tight")
    plt.close(fig)


# 関数: `_write_freeze_policy` の入出力契約と処理意図を定義する。

def _write_freeze_policy(*, results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Phase 7 / Step 7.20.2:
    - output/public/quantum/bell/freeze_policy.json

    Freeze policy must be "blind": it must not inspect the Bell statistic values (S/J) to pick window/offset.
    """

    # 関数: `_sha256_or_none` の入出力契約と処理意図を定義する。
    def _sha256_or_none(p: Path) -> str | None:
        return _sha256(p) if p.exists() else None

    pack_path = OUT_BASE / "falsification_pack.json"
    pack = _read_json_or_none(pack_path) if pack_path.exists() else None
    pack_nw = (pack.get("natural_window") if isinstance(pack, dict) else None) if isinstance(pack, dict) else None
    pack_nw = pack_nw if isinstance(pack_nw, dict) else {}
    plateau_fraction_pairs = float(pack_nw.get("plateau_fraction_pairs", 0.99))
    overrides = pack_nw.get("overrides") if isinstance(pack_nw.get("overrides"), dict) else {}

    rules = [
        {
            "rule_id": "event_ready_protocol_offset_nearest_grid",
            "applies_to": "delft_*",
            "recommended": {"start_offset_ps": 0, "method": {"name": "event_ready_protocol"}},
            "freeze": {"snap_rule": "nearest_grid_to_recommended_start_offset_ps"},
            "note": "Use the published event-ready protocol baseline; treat offset sweep as systematic only.",
        },
        {
            "rule_id": "time_tag_window_from_dt_peak_then_snap_min_ge",
            "applies_to": "time-tag datasets (e.g., weihs1998_*)",
            "recommended": {"method": {"name": "dt_peak_hist_signal_fraction", "signal_fraction": 0.99}},
            "freeze": {"snap_rule": "min(window in grid where window >= recommended_window_ns); fallback: min grid"},
            "note": "Derive recommended window from dt peak width/drift (pairing structure), not from maximizing S/J.",
        },
        {
            "rule_id": "nist_trial_match_pairs_total_then_snap_min_ge",
            "applies_to": "nist_*",
            "recommended": {"method": {"name": "trial_match_pairs_total"}},
            "freeze": {"snap_rule": "min(window in grid where window >= recommended_window_ns); fallback: min grid"},
            "note": "Use independent trial-based coincidence total to pick a matching window (blind to S/J).",
        },
        {
            "rule_id": "kwiat_dataset_recommendation_then_snap_min_ge",
            "applies_to": "kwiat2013_*",
            "recommended": {"method": {"name": "dataset_recommendation", "recommended_window_bins": 18000}},
            "freeze": {"snap_rule": "min(window in grid where window >= recommended_window_ns); fallback: min grid"},
            "note": "Use public dataset guidance for the coincidence window; do not optimize J over the sweep.",
        },
        {
            "rule_id": "fallback_pairs_plateau_then_snap_min_ge",
            "applies_to": "other",
            "recommended": {"method": {"name": "pairs_plateau", "plateau_fraction_pairs": plateau_fraction_pairs}},
            "freeze": {"snap_rule": "min(window in grid where window >= recommended_window_ns); fallback: min grid"},
            "note": "Fallback: use a pairs-total plateau (not a statistic plateau).",
        },
    ]

    datasets_out: list[dict[str, Any]] = []
    # 条件分岐: `isinstance(pack, dict)` を満たす経路を評価する。
    if isinstance(pack, dict):
        ds_list = pack.get("datasets") if isinstance(pack.get("datasets"), list) else []
        for d in ds_list:
            # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
            if not isinstance(d, dict):
                continue

            ds_id = str(d.get("dataset_id") or "")
            # 条件分岐: `not ds_id` を満たす経路を評価する。
            if not ds_id:
                continue

            ds_dir = OUT_BASE / ds_id
            nwin_path = ds_dir / "natural_window_frozen.json"
            nwin = _read_json_or_none(nwin_path) if nwin_path.exists() else None
            nat = (nwin.get("natural_window") if isinstance(nwin, dict) else None) if isinstance(nwin, dict) else None
            nat = nat if isinstance(nat, dict) else {}

            rec_method = d.get("recommended_window_method")
            rec_method_name = str(rec_method.get("name") or "") if isinstance(rec_method, dict) else None
            # 条件分岐: `ds_id.startswith("delft_")` を満たす経路を評価する。
            if ds_id.startswith("delft_"):
                applied_rule = "event_ready_protocol_offset_nearest_grid"
            # 条件分岐: 前段条件が不成立で、`ds_id.startswith("nist_")` を追加評価する。
            elif ds_id.startswith("nist_"):
                applied_rule = "nist_trial_match_pairs_total_then_snap_min_ge"
            # 条件分岐: 前段条件が不成立で、`ds_id.startswith("kwiat2013_")` を追加評価する。
            elif ds_id.startswith("kwiat2013_"):
                applied_rule = "kwiat_dataset_recommendation_then_snap_min_ge"
            # 条件分岐: 前段条件が不成立で、`rec_method_name == "dt_peak_hist_signal_fraction"` を追加評価する。
            elif rec_method_name == "dt_peak_hist_signal_fraction":
                applied_rule = "time_tag_window_from_dt_peak_then_snap_min_ge"
            else:
                applied_rule = "fallback_pairs_plateau_then_snap_min_ge"

            evidence: dict[str, Any] = {
                "natural_window_frozen_json": {"path": _relpath_from_root(nwin_path), "sha256": _sha256_or_none(nwin_path)},
            }
            for extra in ("window_sweep_metrics.json", "offset_sweep_metrics.json", "trial_based_counts.json"):
                p = ds_dir / extra
                # 条件分岐: `p.exists()` を満たす経路を評価する。
                if p.exists():
                    evidence[extra.replace(".", "_")] = {"path": _relpath_from_root(p), "sha256": _sha256_or_none(p)}

            datasets_out.append(
                {
                    "dataset_id": ds_id,
                    "display_name": _dataset_display_name(ds_id),
                    "selection_knob": d.get("selection_knob"),
                    "recommended": {
                        "recommended_window_ns": d.get("recommended_window_ns"),
                        "recommended_start_offset_ps": d.get("recommended_start_offset_ps"),
                        "recommended_window_method": rec_method,
                    },
                    "frozen": {
                        "frozen_window_ns": nat.get("frozen_window_ns"),
                        "frozen_start_offset_ps": nat.get("frozen_start_offset_ps"),
                        "snap_rule": nat.get("snap_rule") or ("nearest_grid_to_recommended_start_offset_ps" if ds_id.startswith("delft_") else None),
                    },
                    "applied_rule_id": applied_rule,
                    "natural_window_overrides": overrides if isinstance(overrides, dict) else {},
                    "evidence": evidence,
                }
            )

    out_path = OUT_BASE / "freeze_policy.json"
    payload = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 7, "step": "7.20.2", "name": "Bell blind freeze policy (window/offset)"},
        "main_script": {"path": "scripts/quantum/bell_primary_products.py", "repro": "python -B scripts/quantum/bell_primary_products.py"},
        "policy_id": "bell_freeze_policy_v1",
        "policy": {
            "blind_freeze_requirement": "Do not inspect Bell statistics (S/J) when selecting natural window/offset.",
            "plateau_fraction_pairs": plateau_fraction_pairs,
            "overrides": overrides,
            "note": "This policy fixes the operational freeze point for window/offset selection and reduces p-hacking degrees of freedom.",
        },
        "rules": rules,
        "datasets": datasets_out,
        "inputs": {
            "falsification_pack_json": {"path": _relpath_from_root(pack_path), "sha256": _sha256_or_none(pack_path)},
        },
        "outputs": {"freeze_policy_json": _relpath_from_root(out_path)},
    }
    _write_json(out_path, payload)
    return payload


# 関数: `_write_null_tests` の入出力契約と処理意図を定義する。

def _write_null_tests(*, results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Phase 7 / Step 7.20.1:
    - output/public/quantum/bell/<dataset>/null_tests.json
    - output/public/quantum/bell/null_tests_summary.json
    """

    # 関数: `_safe_float` の入出力契約と処理意図を定義する。
    def _safe_float(x: Any) -> float | None:
        try:
            v = float(x)
        except Exception:
            return None

        return v if math.isfinite(v) else None

    # 関数: `_abs_delta_over_sigma` の入出力契約と処理意図を定義する。

    def _abs_delta_over_sigma(*, x: float | None, y: float | None, sigma: float | None) -> float | None:
        xv = _safe_float(x)
        yv = _safe_float(y)
        sv = _safe_float(sigma)
        # 条件分岐: `xv is None or yv is None or sv is None or sv <= 0.0` を満たす経路を評価する。
        if xv is None or yv is None or sv is None or sv <= 0.0:
            return None

        return abs(float(xv) - float(yv)) / float(sv)

    summary_path = OUT_BASE / "null_tests_summary.json"
    dataset_summaries: list[dict[str, Any]] = []

    for r in results:
        ds = str(r.get("dataset_id") or "")
        # 条件分岐: `not ds` を満たす経路を評価する。
        if not ds:
            continue

        ds_dir = OUT_BASE / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        out_path = ds_dir / "null_tests.json"
        nw_path = ds_dir / "natural_window_frozen.json"

        nw = _read_json_or_none(nw_path) if nw_path.exists() else None
        baseline_obj = (nw.get("baseline") if isinstance(nw, dict) else None) if isinstance(nw, dict) else None
        baseline_obj = baseline_obj if isinstance(baseline_obj, dict) else {}

        baseline_stat = _safe_float(baseline_obj.get("statistic"))
        baseline_sigma = _safe_float(baseline_obj.get("sigma_boot"))
        baseline_delay_z = _delay_signature_z_max(r.get("delay_signature"))

        dataset_out: dict[str, Any] = {
            "generated_utc": _utc_now(),
            "dataset_id": ds,
            "supported": False,
            "baseline": {
                "param_name": str(baseline_obj.get("param_name") or ""),
                "value": _safe_float(baseline_obj.get("value")),
                "statistic_name": str(baseline_obj.get("statistic_name") or ""),
                "statistic": baseline_stat,
                "sigma_boot": baseline_sigma,
                "delay_signature_z_max": float(baseline_delay_z) if baseline_delay_z is not None else None,
            },
            "tests": [],
            "inputs": {
                "natural_window_frozen_json": {
                    "path": _relpath_from_root(nw_path),
                    "sha256": _sha256(nw_path) if nw_path.exists() else None,
                },
            },
        }

        tests: list[dict[str, Any]] = []

        # --- Weihs 1998 (time-tag; CHSH |S|)
        if ds.startswith("weihs1998_"):
            try:
                mod = _load_script_module(rel_path="scripts/quantum/weihs1998_time_tag_reanalysis.py", name="_weihs1998_null")
                data = np.load(ds_dir / "normalized_events.npz")
                meta = _read_json(ds_dir / "normalized_events.json")
                encoding = str((meta.get("schema") or {}).get("encoding") or "bit0-setting")
                offset_s = float((meta.get("offset_estimate") or {}).get("offset_s"))

                t_a = data["a_t_s"].astype(np.float64, copy=False)
                c_a = data["a_c"].astype(np.uint16, copy=False)
                t_b = data["b_t_s"].astype(np.float64, copy=False)
                c_b = data["b_c"].astype(np.uint16, copy=False)

                ref_window_ns = 1.0
                frozen_window_ns = _safe_float((nw.get("natural_window") or {}).get("frozen_window_ns")) if isinstance(nw, dict) else None
                # 条件分岐: `frozen_window_ns is None` を満たす経路を評価する。
                if frozen_window_ns is None:
                    frozen_window_ns = 1.0

                # Fixed CHSH variant derived at ref window on original data.

                n_ref, sum_prod_ref, _pairs_ref = mod._pair_and_accumulate(
                    t_a,
                    c_a,
                    t_b,
                    c_b,
                    offset_s=offset_s,
                    window_s=float(ref_window_ns) * 1e-9,
                    encoding=encoding,
                )
                E_ref = mod._safe_div(sum_prod_ref, n_ref)
                variant_fixed, _s_ref_best = _best_chsh_variant(E_ref)

                # 関数: `_compute_s_abs_and_pairs` の入出力契約と処理意図を定義する。
                def _compute_s_abs_and_pairs(tt_a: np.ndarray, cc_a: np.ndarray, tt_b: np.ndarray, cc_b: np.ndarray) -> tuple[float | None, int | None]:
                    n, sum_prod, pairs_total = mod._pair_and_accumulate(
                        tt_a,
                        cc_a,
                        tt_b,
                        cc_b,
                        offset_s=offset_s,
                        window_s=float(frozen_window_ns) * 1e-9,
                        encoding=encoding,
                    )
                    E = mod._safe_div(sum_prod, n)
                    # 条件分岐: `not np.isfinite(E).all()` を満たす経路を評価する。
                    if not np.isfinite(E).all():
                        return None, int(pairs_total)

                    s = _apply_chsh_variant(E, variant_fixed)
                    # 条件分岐: `not math.isfinite(float(s))` を満たす経路を評価する。
                    if not math.isfinite(float(s)):
                        return None, int(pairs_total)

                    return float(abs(float(s))), int(pairs_total)

                # 関数: `_compute_delay_z_max` の入出力契約と処理意図を定義する。

                def _compute_delay_z_max(tt_a: np.ndarray, cc_a: np.ndarray, tt_b: np.ndarray, cc_b: np.ndarray) -> float | None:
                    dt_by_ab: dict[tuple[int, int], list[float]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
                    i = 0
                    j = 0
                    na = int(tt_a.size)
                    nb = int(tt_b.size)
                    w_s = float(ref_window_ns) * 1e-9
                    while i < na and j < nb:
                        dt = float(tt_b[j] - tt_a[i] - offset_s)
                        # 条件分岐: `dt < -w_s` を満たす経路を評価する。
                        if dt < -w_s:
                            j += 1
                            continue

                        # 条件分岐: `dt > w_s` を満たす経路を評価する。

                        if dt > w_s:
                            i += 1
                            continue

                        a_set, _ = mod._extract_setting_and_outcome(int(cc_a[i]), encoding=encoding)
                        b_set, _ = mod._extract_setting_and_outcome(int(cc_b[j]), encoding=encoding)
                        dt_by_ab[(int(a_set), int(b_set))].append(dt * 1e9)
                        i += 1
                        j += 1

                    eps_sig = 0.1
                    sig_a_best: dict[str, Any] | None = None
                    for b in (0, 1):
                        sig = _delay_signature_delta_median(dt_by_ab[(0, b)], dt_by_ab[(1, b)], epsilon=eps_sig)
                        z = sig.get("z_delta_median")
                        # 条件分岐: `z is None or not math.isfinite(float(z))` を満たす経路を評価する。
                        if z is None or not math.isfinite(float(z)):
                            continue

                        # 条件分岐: `sig_a_best is None or abs(float(z)) > abs(float(sig_a_best.get("z_delta_media...` を満たす経路を評価する。

                        if sig_a_best is None or abs(float(z)) > abs(float(sig_a_best.get("z_delta_median") or 0.0)):
                            sig_a_best = sig

                    sig_b_best: dict[str, Any] | None = None
                    for a in (0, 1):
                        sig = _delay_signature_delta_median(dt_by_ab[(a, 0)], dt_by_ab[(a, 1)], epsilon=eps_sig)
                        z = sig.get("z_delta_median")
                        # 条件分岐: `z is None or not math.isfinite(float(z))` を満たす経路を評価する。
                        if z is None or not math.isfinite(float(z)):
                            continue

                        # 条件分岐: `sig_b_best is None or abs(float(z)) > abs(float(sig_b_best.get("z_delta_media...` を満たす経路を評価する。

                        if sig_b_best is None or abs(float(z)) > abs(float(sig_b_best.get("z_delta_median") or 0.0)):
                            sig_b_best = sig

                    delay_signature = {
                        "Alice": sig_a_best,
                        "Bob": sig_b_best,
                    }
                    return _delay_signature_z_max(delay_signature)

                # Baseline (recomputed for sanity; should be close to natural_window_frozen baseline).

                base_s_abs, base_pairs = _compute_s_abs_and_pairs(t_a, c_a, t_b, c_b)

                # Null 1: shuffle setting bits (keep outcome bits).
                rng_a = np.random.default_rng(_stable_seed("bell", "null", ds, "shuffle_settings", "A"))
                rng_b = np.random.default_rng(_stable_seed("bell", "null", ds, "shuffle_settings", "B"))
                c_a_shuf = _shuffle_setting_bits_codes(c_a, encoding=encoding, rng=rng_a)
                c_b_shuf = _shuffle_setting_bits_codes(c_b, encoding=encoding, rng=rng_b)
                s_abs_shuf, pairs_shuf = _compute_s_abs_and_pairs(t_a, c_a_shuf, t_b, c_b_shuf)
                z_shuf = _compute_delay_z_max(t_a, c_a_shuf, t_b, c_b_shuf)
                tests.append(
                    {
                        "id": "shuffle_settings",
                        "supported": True,
                        "method": {
                            "name": "shuffle_settings_bits",
                            "encoding": encoding,
                            "seed_a": int(_stable_seed("bell", "null", ds, "shuffle_settings", "A")),
                            "seed_b": int(_stable_seed("bell", "null", ds, "shuffle_settings", "B")),
                            "note": "Shuffle only setting labels in the code stream; keep outcome/detector bit intact.",
                        },
                        "result": {
                            "statistic_name": "CHSH |S| (fixed variant; frozen window)",
                            "statistic_abs": s_abs_shuf,
                            "pairs_total": pairs_shuf,
                            "abs_delta_over_sigma_boot": _abs_delta_over_sigma(x=s_abs_shuf, y=baseline_stat, sigma=baseline_sigma),
                            "delay_signature_z_max": z_shuf,
                        },
                    }
                )

                # Null 2: circular time shift on Bob stream (break true pairing while keeping marginals).
                span_s = float(np.max(t_b) - np.min(t_b)) if int(t_b.size) else 0.0
                shift_s = 0.5 * span_s if math.isfinite(span_s) and span_s > 0.0 else 0.0
                t_b_shift, c_b_shift, shift_info = _circular_time_shift_sorted_pair(t_b, c_b, shift_s=shift_s)
                s_abs_shift, pairs_shift = _compute_s_abs_and_pairs(t_a, c_a, t_b_shift, c_b_shift)
                z_shift = _compute_delay_z_max(t_a, c_a, t_b_shift, c_b_shift)
                tests.append(
                    {
                        "id": "time_shift_bob",
                        "supported": bool(shift_info.get("supported")),
                        "method": {"name": "circular_time_shift_sorted_pair", **shift_info},
                        "result": {
                            "statistic_name": "CHSH |S| (fixed variant; frozen window)",
                            "statistic_abs": s_abs_shift,
                            "pairs_total": pairs_shift,
                            "abs_delta_over_sigma_boot": _abs_delta_over_sigma(x=s_abs_shift, y=baseline_stat, sigma=baseline_sigma),
                            "delay_signature_z_max": z_shift,
                        },
                    }
                )

                dataset_out["supported"] = True
                dataset_out["baseline"]["statistic_abs_recomputed"] = base_s_abs
                dataset_out["baseline"]["pairs_total_recomputed"] = base_pairs
            except Exception as exc:
                dataset_out["supported"] = False
                dataset_out["reason"] = str(exc)

        # --- Delft (event-ready trials; CHSH S / S_combined)
        elif ds.startswith("delft_"):
            try:
                mod = _load_script_module(rel_path="scripts/quantum/delft_hensen2015_chsh_reanalysis.py", name="_delft_null")
                frozen_off = _safe_float((nw.get("natural_window") or {}).get("frozen_start_offset_ps")) if isinstance(nw, dict) else None
                # 条件分岐: `frozen_off is None` を満たす経路を評価する。
                if frozen_off is None:
                    frozen_off = 0.0

                # 関数: `_shuffle_settings_column` の入出力契約と処理意図を定義する。

                def _shuffle_settings_column(data: np.ndarray, *, col: int, seed: int) -> np.ndarray:
                    rng = np.random.default_rng(int(seed))
                    perm = rng.permutation(int(data.shape[0]))
                    out = np.asarray(data).copy()
                    out[:, col] = np.asarray(data)[perm, col]
                    return out

                # 条件分岐: `ds == "delft_hensen2015"` を満たす経路を評価する。

                if ds == "delft_hensen2015":
                    zip_path = ROOT / "data" / "quantum" / "sources" / ds / "data.zip"
                    _ts, data = mod._load_table_from_zip(zip_path, member="bell_open_data.txt")
                    p = mod.Params()

                    base = mod._analyze(data, p=p, start_offset_ps=int(round(frozen_off)))
                    data2 = _shuffle_settings_column(data, col=6, seed=_stable_seed("bell", "null", ds, "shuffle_settings", "A"))
                    data2 = _shuffle_settings_column(data2, col=7, seed=_stable_seed("bell", "null", ds, "shuffle_settings", "B"))
                    res = mod._analyze(data2, p=p, start_offset_ps=int(round(frozen_off)))

                    tests.append(
                        {
                            "id": "shuffle_settings",
                            "supported": True,
                            "method": {
                                "name": "shuffle_trial_settings",
                                "seed_a": int(_stable_seed("bell", "null", ds, "shuffle_settings", "A")),
                                "seed_b": int(_stable_seed("bell", "null", ds, "shuffle_settings", "B")),
                                "note": "Shuffle random_number_a/b across trials; keep trial outcomes unchanged.",
                            },
                            "result": {
                                "statistic_name": "CHSH S (event-ready trials; frozen start offset)",
                                "statistic": float(res.s),
                                "statistic_abs": float(abs(float(res.s))) if math.isfinite(float(res.s)) else None,
                                "abs_delta_over_sigma_boot": _abs_delta_over_sigma(x=float(res.s), y=baseline_stat, sigma=baseline_sigma),
                            },
                        }
                    )
                    dataset_out["supported"] = True
                    dataset_out["baseline"]["statistic_recomputed"] = float(base.s)

                # 条件分岐: 前段条件が不成立で、`ds == "delft_hensen2016_srep30289"` を追加評価する。
                elif ds == "delft_hensen2016_srep30289":
                    zip_path = ROOT / "data" / "quantum" / "sources" / ds / "data.zip"
                    old_member = "bell_open_data_2_old_detector.txt"
                    new_member = "bell_open_data_2_new_detector.txt"
                    _ts_old, data_old = mod._load_table_from_zip(zip_path, member=old_member)
                    _ts_new, data_new = mod._load_table_from_zip(zip_path, member=new_member)
                    p2 = mod.Hensen2016Params()

                    base = mod._analyze_hensen2016(data_old, data_new, p=p2, start_offset_ps=int(round(frozen_off)))
                    data_old2 = _shuffle_settings_column(data_old, col=6, seed=_stable_seed("bell", "null", ds, "shuffle_settings", "A", "old"))
                    data_old2 = _shuffle_settings_column(data_old2, col=7, seed=_stable_seed("bell", "null", ds, "shuffle_settings", "B", "old"))
                    data_new2 = _shuffle_settings_column(data_new, col=6, seed=_stable_seed("bell", "null", ds, "shuffle_settings", "A", "new"))
                    data_new2 = _shuffle_settings_column(data_new2, col=7, seed=_stable_seed("bell", "null", ds, "shuffle_settings", "B", "new"))
                    res = mod._analyze_hensen2016(data_old2, data_new2, p=p2, start_offset_ps=int(round(frozen_off)))

                    tests.append(
                        {
                            "id": "shuffle_settings",
                            "supported": True,
                            "method": {
                                "name": "shuffle_trial_settings",
                                "seed_a_old": int(_stable_seed("bell", "null", ds, "shuffle_settings", "A", "old")),
                                "seed_b_old": int(_stable_seed("bell", "null", ds, "shuffle_settings", "B", "old")),
                                "seed_a_new": int(_stable_seed("bell", "null", ds, "shuffle_settings", "A", "new")),
                                "seed_b_new": int(_stable_seed("bell", "null", ds, "shuffle_settings", "B", "new")),
                                "note": "Shuffle random_number_a/b across trials within each run (old/new detector); keep outcomes unchanged.",
                            },
                            "result": {
                                "statistic_name": "CHSH S_combined (event-ready trials; frozen start offset)",
                                "statistic": float(res.s_combined),
                                "statistic_abs": float(abs(float(res.s_combined))) if math.isfinite(float(res.s_combined)) else None,
                                "abs_delta_over_sigma_boot": _abs_delta_over_sigma(
                                    x=float(res.s_combined), y=baseline_stat, sigma=baseline_sigma
                                ),
                            },
                        }
                    )
                    dataset_out["supported"] = True
                    dataset_out["baseline"]["statistic_recomputed"] = float(base.s_combined)

                else:
                    dataset_out["supported"] = False
                    dataset_out["reason"] = f"unknown delft dataset: {ds}"
            except Exception as exc:
                dataset_out["supported"] = False
                dataset_out["reason"] = str(exc)

        # --- NIST (time-tag; delay signature on click_delay)
        elif ds.startswith("nist_"):
            try:
                data = np.load(ds_dir / "normalized_events.npz")
                seconds_per_timetag = float(data["seconds_per_timetag"][0])
                a_delay_ns = data["alice_click_delay"].astype(np.float64) * seconds_per_timetag * 1e9
                b_delay_ns = data["bob_click_delay"].astype(np.float64) * seconds_per_timetag * 1e9
                a_set = data["alice_click_setting"].astype(np.int8, copy=False)
                b_set = data["bob_click_setting"].astype(np.int8, copy=False)

                rng_a = np.random.default_rng(_stable_seed("bell", "null", ds, "shuffle_settings", "A"))
                rng_b = np.random.default_rng(_stable_seed("bell", "null", ds, "shuffle_settings", "B"))
                a_set_shuf = a_set[rng_a.permutation(int(a_set.size))] if int(a_set.size) else a_set.copy()
                b_set_shuf = b_set[rng_b.permutation(int(b_set.size))] if int(b_set.size) else b_set.copy()

                eps_sig = 0.1
                delay_sig = {
                    "Alice": _delay_signature_delta_median(a_delay_ns[a_set_shuf == 0], a_delay_ns[a_set_shuf == 1], epsilon=eps_sig),
                    "Bob": _delay_signature_delta_median(b_delay_ns[b_set_shuf == 0], b_delay_ns[b_set_shuf == 1], epsilon=eps_sig),
                }
                z_shuf = _delay_signature_z_max(delay_sig)
                tests.append(
                    {
                        "id": "shuffle_settings",
                        "supported": True,
                        "method": {
                            "name": "shuffle_click_settings",
                            "seed_a": int(_stable_seed("bell", "null", ds, "shuffle_settings", "A")),
                            "seed_b": int(_stable_seed("bell", "null", ds, "shuffle_settings", "B")),
                            "note": "Shuffle setting labels across time-tag clicks; keep click_delay marginals unchanged.",
                        },
                        "result": {
                            "delay_signature_z_max": float(z_shuf) if z_shuf is not None else None,
                        },
                    }
                )
                dataset_out["supported"] = True
            except Exception as exc:
                dataset_out["supported"] = False
                dataset_out["reason"] = str(exc)

        # --- Kwiat/Christensen 2013 (trial-based; delay signature on dt-from-trigger at ref window)
        elif ds.startswith("kwiat2013_"):
            try:
                ref_npz = ds_dir / "delay_signature_ref_samples.npz"
                # 条件分岐: `not ref_npz.exists()` を満たす経路を評価する。
                if not ref_npz.exists():
                    raise FileNotFoundError(ref_npz)

                dt = np.load(ref_npz)
                a0 = np.asarray(dt["alice_setting0_dt_ns"], dtype=float)
                a1v = np.asarray(dt["alice_setting1_dt_ns"], dtype=float)
                b0 = np.asarray(dt["bob_setting0_dt_ns"], dtype=float)
                b1v = np.asarray(dt["bob_setting1_dt_ns"], dtype=float)

                # 関数: `_shuffle_labels` の入出力契約と処理意図を定義する。
                def _shuffle_labels(x0: np.ndarray, x1: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
                    x = np.concatenate([np.asarray(x0, dtype=float).reshape(-1), np.asarray(x1, dtype=float).reshape(-1)], axis=0)
                    labels = np.concatenate(
                        [
                            np.zeros(int(np.asarray(x0).size), dtype=np.int8),
                            np.ones(int(np.asarray(x1).size), dtype=np.int8),
                        ],
                        axis=0,
                    )
                    rng = np.random.default_rng(int(seed))
                    labels2 = labels[rng.permutation(int(labels.size))] if int(labels.size) else labels
                    return x[labels2 == 0], x[labels2 == 1]

                a0s, a1s = _shuffle_labels(a0, a1v, seed=_stable_seed("bell", "null", ds, "shuffle_settings", "A"))
                b0s, b1s = _shuffle_labels(b0, b1v, seed=_stable_seed("bell", "null", ds, "shuffle_settings", "B"))

                eps_sig = 0.1
                delay_sig = {
                    "Alice": _delay_signature_delta_median(a0s, a1s, epsilon=eps_sig),
                    "Bob": _delay_signature_delta_median(b0s, b1s, epsilon=eps_sig),
                }
                z_shuf = _delay_signature_z_max(delay_sig)
                tests.append(
                    {
                        "id": "shuffle_settings",
                        "supported": True,
                        "method": {
                            "name": "shuffle_delay_samples_labels",
                            "seed_a": int(_stable_seed("bell", "null", ds, "shuffle_settings", "A")),
                            "seed_b": int(_stable_seed("bell", "null", ds, "shuffle_settings", "B")),
                            "note": "Shuffle setting labels for ref-window delay samples (dt from trigger) and recompute Δmedian z.",
                        },
                        "result": {
                            "delay_signature_z_max": float(z_shuf) if z_shuf is not None else None,
                        },
                    }
                )
                dataset_out["supported"] = True
                dataset_out["inputs"]["delay_signature_ref_samples_npz"] = {
                    "path": _relpath_from_root(ref_npz),
                    "sha256": _sha256(ref_npz),
                }
            except Exception as exc:
                dataset_out["supported"] = False
                dataset_out["reason"] = str(exc)

        dataset_out["tests"] = tests
        _write_json(out_path, dataset_out)

        dataset_summaries.append(
            {
                "dataset_id": ds,
                "display_name": _dataset_display_name(ds),
                "statistic_family": _dataset_statistic_family(ds),
                "baseline": dataset_out.get("baseline"),
                "tests": dataset_out.get("tests"),
                "null_tests_json": {
                    "path": _relpath_from_root(out_path),
                    "sha256": _sha256(out_path) if out_path.exists() else None,
                },
            }
        )

    summary = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 7, "step": "7.20.1", "name": "Bell null tests (shuffle settings / time shift)"},
        "main_script": {"path": "scripts/quantum/bell_primary_products.py", "repro": "python -B scripts/quantum/bell_primary_products.py"},
        "datasets": dataset_summaries,
        "outputs": {
            "null_tests_summary_json": _relpath_from_root(summary_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


# 関数: `_write_pairing_crosschecks` の入出力契約と処理意図を定義する。

def _write_pairing_crosschecks(*, results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Phase 7 / Step 7.20.3:
    - output/public/quantum/bell/<dataset>/crosscheck_pairing.json
    - output/public/quantum/bell/crosscheck_pairing_summary.json

    Quantify "implementation/pairing choice" differences as a systematic (for supported datasets).
    """

    # 関数: `_sha256_or_none` の入出力契約と処理意図を定義する。
    def _sha256_or_none(p: Path) -> str | None:
        return _sha256(p) if p.exists() else None

    # 関数: `_extract_setting_and_outcome_vec` の入出力契約と処理意図を定義する。

    def _extract_setting_and_outcome_vec(c: np.ndarray, *, encoding: str) -> tuple[np.ndarray, np.ndarray]:
        cc = np.asarray(c, dtype=np.uint16).reshape(-1)
        # 条件分岐: `cc.size == 0` を満たす経路を評価する。
        if cc.size == 0:
            return np.asarray([], dtype=np.int8), np.asarray([], dtype=np.int8)

        # 条件分岐: `encoding == "bit0-setting"` を満たす経路を評価する。

        if encoding == "bit0-setting":
            setting = (cc & np.uint16(1)).astype(np.int8, copy=False)
            det = ((cc >> np.uint16(1)) & np.uint16(1)).astype(np.int8, copy=False)
        # 条件分岐: 前段条件が不成立で、`encoding == "bit0-detector"` を追加評価する。
        elif encoding == "bit0-detector":
            det = (cc & np.uint16(1)).astype(np.int8, copy=False)
            setting = ((cc >> np.uint16(1)) & np.uint16(1)).astype(np.int8, copy=False)
        else:
            raise ValueError(f"unknown encoding: {encoding}")

        outcome = (np.int8(1) - np.int8(2) * det).astype(np.int8, copy=False)  # det 0->+1, det 1->-1
        return setting, outcome

    # 関数: `_compute_ch_j_prob` の入出力契約と処理意図を定義する。

    def _compute_ch_j_prob(
        *,
        n_trials: np.ndarray,
        n_coinc: np.ndarray,
        n_trials_a: np.ndarray,
        n_trials_b: np.ndarray,
        n_click_a: np.ndarray,
        n_click_b: np.ndarray,
        a1: int,
        b1: int,
    ) -> float:
        n_trials0 = np.asarray(n_trials, dtype=np.int64)
        n_coinc0 = np.asarray(n_coinc, dtype=np.int64)
        n_trials_a0 = np.asarray(n_trials_a, dtype=np.int64).reshape(-1)
        n_trials_b0 = np.asarray(n_trials_b, dtype=np.int64).reshape(-1)
        n_click_a0 = np.asarray(n_click_a, dtype=np.int64).reshape(-1)
        n_click_b0 = np.asarray(n_click_b, dtype=np.int64).reshape(-1)

        # 条件分岐: `n_trials0.shape != (2, 2) or n_coinc0.shape != (2, 2)` を満たす経路を評価する。
        if n_trials0.shape != (2, 2) or n_coinc0.shape != (2, 2):
            raise ValueError("CH J_prob expects 2x2 trial and coincidence matrices")

        # 条件分岐: `n_trials_a0.size < 2 or n_trials_b0.size < 2 or n_click_a0.size < 2 or n_clic...` を満たす経路を評価する。

        if n_trials_a0.size < 2 or n_trials_b0.size < 2 or n_click_a0.size < 2 or n_click_b0.size < 2:
            raise ValueError("CH J_prob expects 2-element singles/trials arrays")

        # 条件分岐: `a1 not in (0, 1) or b1 not in (0, 1)` を満たす経路を評価する。

        if a1 not in (0, 1) or b1 not in (0, 1):
            raise ValueError("a1/b1 must be in {0,1}")

        p_a1 = float(n_click_a0[int(a1)] / max(1, int(n_trials_a0[int(a1)])))
        p_b1 = float(n_click_b0[int(b1)] / max(1, int(n_trials_b0[int(b1)])))
        c = n_coinc0.astype(np.float64, copy=False)
        t = n_trials0.astype(np.float64, copy=False)
        return float(
            c[0, 0] / max(1.0, float(t[0, 0]))
            + c[0, 1] / max(1.0, float(t[0, 1]))
            + c[1, 0] / max(1.0, float(t[1, 0]))
            - c[1, 1] / max(1.0, float(t[1, 1]))
            - p_a1
            - p_b1
        )

    # 関数: `_mutual_nearest_pairs` の入出力契約と処理意図を定義する。

    def _mutual_nearest_pairs(
        t_a: np.ndarray,
        t_b: np.ndarray,
        *,
        offset_s: float,
        window_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        ta = np.asarray(t_a, dtype=np.float64).reshape(-1)
        tb = np.asarray(t_b, dtype=np.float64).reshape(-1)
        na = int(ta.size)
        nb = int(tb.size)
        # 条件分岐: `na == 0 or nb == 0` を満たす経路を評価する。
        if na == 0 or nb == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

        # A -> nearest B to (t_a + offset)

        target_b = ta + float(offset_s)
        pos = np.searchsorted(tb, target_b, side="left")
        cand0 = np.clip(pos, 0, nb - 1).astype(np.int64, copy=False)
        cand1 = np.clip(pos - 1, 0, nb - 1).astype(np.int64, copy=False)
        dt0 = tb[cand0] - target_b
        dt1 = tb[cand1] - target_b
        use0 = np.abs(dt0) <= np.abs(dt1)
        j_near = np.where(use0, cand0, cand1).astype(np.int64, copy=False)
        dt = tb[j_near] - target_b
        valid_a = np.abs(dt) <= float(window_s)
        j_near = np.where(valid_a, j_near, -1).astype(np.int64, copy=False)

        # B -> nearest A to (t_b - offset)
        target_a = tb - float(offset_s)
        pos2 = np.searchsorted(ta, target_a, side="left")
        cand0b = np.clip(pos2, 0, na - 1).astype(np.int64, copy=False)
        cand1b = np.clip(pos2 - 1, 0, na - 1).astype(np.int64, copy=False)
        dt0b = tb - (ta[cand0b] + float(offset_s))
        dt1b = tb - (ta[cand1b] + float(offset_s))
        use0b = np.abs(dt0b) <= np.abs(dt1b)
        i_near = np.where(use0b, cand0b, cand1b).astype(np.int64, copy=False)
        dtb = tb - (ta[i_near] + float(offset_s))
        valid_b = np.abs(dtb) <= float(window_s)
        i_near = np.where(valid_b, i_near, -1).astype(np.int64, copy=False)

        # Mutual nearest neighbor pairing.
        m = j_near >= 0
        # 条件分岐: `int(np.sum(m)) == 0` を満たす経路を評価する。
        if int(np.sum(m)) == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

        i_valid = np.nonzero(m)[0].astype(np.int64, copy=False)
        j_valid = j_near[m].astype(np.int64, copy=False)
        mutual = i_near[j_valid] == i_valid
        return i_valid[mutual], j_valid[mutual]

    # 関数: `_mutual_nearest_pairs_counts_chunked` の入出力契約と処理意図を定義する。

    def _mutual_nearest_pairs_counts_chunked(
        t_a: np.ndarray,
        t_b: np.ndarray,
        *,
        offset_counts: int,
        window_counts: int,
        chunk_size: int = 1_000_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mutual nearest-neighbor pairing for large integer time-tags (counts).
        Returns index pairs (i,j) into (t_a, t_b).

        Implemented in chunks over A to control peak memory.
        """
        ta = np.asarray(t_a, dtype=np.int64).reshape(-1)
        tb = np.asarray(t_b, dtype=np.int64).reshape(-1)
        na = int(ta.size)
        nb = int(tb.size)
        # 条件分岐: `na == 0 or nb == 0` を満たす経路を評価する。
        if na == 0 or nb == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

        off = int(offset_counts)
        w = int(window_counts)
        # 条件分岐: `w < 0` を満たす経路を評価する。
        if w < 0:
            raise ValueError("window_counts must be non-negative")

        # B -> nearest A to (t_b - offset)

        target_a = (tb - off).astype(np.int64, copy=False)
        pos2 = np.searchsorted(ta, target_a, side="left")
        cand0b = np.clip(pos2, 0, na - 1).astype(np.int64, copy=False)
        cand1b = np.clip(pos2 - 1, 0, na - 1).astype(np.int64, copy=False)
        dt0b = tb - (ta[cand0b] + off)
        dt1b = tb - (ta[cand1b] + off)
        use0b = np.abs(dt0b) <= np.abs(dt1b)
        i_near = np.where(use0b, cand0b, cand1b).astype(np.int64, copy=False)
        dtb = tb - (ta[i_near] + off)
        valid_b = np.abs(dtb) <= w
        # Store as int32 to cut memory; safe for na<=~2e9 (here na ~ 5e6).
        i_near32 = np.where(valid_b, i_near, -1).astype(np.int32, copy=False)

        # A -> nearest B to (t_a + offset), chunked
        i_all: list[np.ndarray] = []
        j_all: list[np.ndarray] = []
        cs = int(chunk_size)
        # 条件分岐: `cs <= 0` を満たす経路を評価する。
        if cs <= 0:
            cs = na

        for start in range(0, na, cs):
            end = min(na, start + cs)
            ta_block = ta[start:end]
            target_b = (ta_block + off).astype(np.int64, copy=False)
            pos = np.searchsorted(tb, target_b, side="left")
            cand0 = np.clip(pos, 0, nb - 1).astype(np.int64, copy=False)
            cand1 = np.clip(pos - 1, 0, nb - 1).astype(np.int64, copy=False)
            dt0 = tb[cand0] - target_b
            dt1 = tb[cand1] - target_b
            use0 = np.abs(dt0) <= np.abs(dt1)
            j_near = np.where(use0, cand0, cand1).astype(np.int64, copy=False)
            dt = tb[j_near] - target_b
            valid_a = np.abs(dt) <= w
            # 条件分岐: `int(np.sum(valid_a)) == 0` を満たす経路を評価する。
            if int(np.sum(valid_a)) == 0:
                continue

            i_valid = (np.nonzero(valid_a)[0] + start).astype(np.int64, copy=False)
            j_valid = j_near[valid_a].astype(np.int64, copy=False)
            mutual = i_near32[j_valid] == i_valid.astype(np.int32, copy=False)
            # 条件分岐: `int(np.sum(mutual)) == 0` を満たす経路を評価する。
            if int(np.sum(mutual)) == 0:
                continue

            i_all.append(i_valid[mutual])
            j_all.append(j_valid[mutual])

        # 条件分岐: `not i_all` を満たす経路を評価する。

        if not i_all:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

        return np.concatenate(i_all).astype(np.int64, copy=False), np.concatenate(j_all).astype(np.int64, copy=False)

    summary_path = OUT_BASE / "crosscheck_pairing_summary.json"
    ds_summaries: list[dict[str, Any]] = []

    for r in results:
        ds = str(r.get("dataset_id") or "")
        # 条件分岐: `not ds` を満たす経路を評価する。
        if not ds:
            continue

        ds_dir = OUT_BASE / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        out_path = ds_dir / "crosscheck_pairing.json"

        nw_path = ds_dir / "natural_window_frozen.json"
        nw = _read_json_or_none(nw_path) if nw_path.exists() else None
        base = (nw.get("baseline") if isinstance(nw, dict) else None) if isinstance(nw, dict) else None
        base = base if isinstance(base, dict) else {}
        sigma_boot = base.get("sigma_boot")
        frozen_window_ns = None
        # 条件分岐: `isinstance(nw, dict)` を満たす経路を評価する。
        if isinstance(nw, dict):
            frozen_window_ns = _safe_float((nw.get("natural_window") or {}).get("frozen_window_ns"))
            # 条件分岐: `frozen_window_ns is None` を満たす経路を評価する。
            if frozen_window_ns is None:
                frozen_window_ns = _safe_float((nw.get("natural_window") or {}).get("recommended_window_ns"))

        # 条件分岐: `frozen_window_ns is None` を満たす経路を評価する。

        if frozen_window_ns is None:
            frozen_window_ns = _safe_float(base.get("value"))

        # 条件分岐: `frozen_window_ns is None` を満たす経路を評価する。

        if frozen_window_ns is None:
            frozen_window_ns = 1.0

        payload: dict[str, Any] = {
            "generated_utc": _utc_now(),
            "dataset_id": ds,
            "supported": False,
            "baseline": {
                "param_name": base.get("param_name"),
                "value": base.get("value"),
                "statistic_name": base.get("statistic_name"),
                "statistic": base.get("statistic"),
                "sigma_boot": sigma_boot,
            },
            "methods": [],
            "inputs": {
                "natural_window_frozen_json": {"path": _relpath_from_root(nw_path), "sha256": _sha256_or_none(nw_path)},
            },
        }

        # --- Support: Weihs 1998 time-tag CHSH (pairing matters).
        if ds.startswith("weihs1998_"):
            try:
                mod = _load_script_module(rel_path="scripts/quantum/weihs1998_time_tag_reanalysis.py", name="_weihs1998_xc")
                data = np.load(ds_dir / "normalized_events.npz")
                meta = _read_json(ds_dir / "normalized_events.json")
                encoding = str((meta.get("schema") or {}).get("encoding") or "bit0-setting")
                offset_s = float((meta.get("offset_estimate") or {}).get("offset_s"))

                t_a = data["a_t_s"].astype(np.float64, copy=False)
                c_a = data["a_c"].astype(np.uint16, copy=False)
                t_b = data["b_t_s"].astype(np.float64, copy=False)
                c_b = data["b_c"].astype(np.uint16, copy=False)

                window_s = float(frozen_window_ns) * 1e-9

                # Fixed CHSH variant derived at a small ref window on the original pipeline.
                ref_window_ns = 1.0
                n_ref, sum_prod_ref, _pairs_ref = mod._pair_and_accumulate(
                    t_a,
                    c_a,
                    t_b,
                    c_b,
                    offset_s=offset_s,
                    window_s=float(ref_window_ns) * 1e-9,
                    encoding=encoding,
                )
                E_ref = mod._safe_div(sum_prod_ref, n_ref)
                variant_fixed, _s_ref_best = _best_chsh_variant(E_ref)

                # Method 1: baseline greedy pairing (should match existing pipeline).
                n_g, sum_prod_g, pairs_g = mod._pair_and_accumulate(
                    t_a,
                    c_a,
                    t_b,
                    c_b,
                    offset_s=offset_s,
                    window_s=window_s,
                    encoding=encoding,
                )
                E_g = mod._safe_div(sum_prod_g, n_g)
                s_g = _apply_chsh_variant(E_g, variant_fixed) if np.isfinite(E_g).all() else float("nan")
                s_g_abs = float(abs(float(s_g))) if math.isfinite(float(s_g)) else None

                # Method 2: mutual nearest-neighbor pairing (different convention; one-to-one).
                i_pairs, j_pairs = _mutual_nearest_pairs(t_a, t_b, offset_s=offset_s, window_s=window_s)
                a_set, a_out = _extract_setting_and_outcome_vec(c_a[i_pairs], encoding=encoding)
                b_set, b_out = _extract_setting_and_outcome_vec(c_b[j_pairs], encoding=encoding)
                prod = (a_out.astype(np.int16) * b_out.astype(np.int16)).astype(np.int16, copy=False)

                n_m = np.zeros((2, 2), dtype=np.int64)
                sum_prod_m = np.zeros((2, 2), dtype=np.int64)
                # 条件分岐: `int(prod.size)` を満たす経路を評価する。
                if int(prod.size):
                    np.add.at(n_m, (a_set, b_set), 1)
                    np.add.at(sum_prod_m, (a_set, b_set), prod.astype(np.int64, copy=False))

                E_m = mod._safe_div(sum_prod_m, n_m)
                s_m = _apply_chsh_variant(E_m, variant_fixed) if np.isfinite(E_m).all() else float("nan")
                s_m_abs = float(abs(float(s_m))) if math.isfinite(float(s_m)) else None

                delta_over_sigma = None
                try:
                    sigma_v = float(sigma_boot)
                    # 条件分岐: `sigma_v > 0 and s_g_abs is not None and s_m_abs is not None` を満たす経路を評価する。
                    if sigma_v > 0 and s_g_abs is not None and s_m_abs is not None:
                        delta_over_sigma = abs(float(s_m_abs) - float(s_g_abs)) / sigma_v
                except Exception:
                    delta_over_sigma = None

                payload["supported"] = True
                payload["methods"] = [
                    {
                        "id": "greedy_time_order_1to1",
                        "pairing": "greedy 1:1 pairing by time order within |dt-offset|<=window",
                        "window_ns": float(frozen_window_ns),
                        "statistic_name": "CHSH |S| (fixed variant)",
                        "statistic_abs": s_g_abs,
                        "pairs_total": int(pairs_g),
                    },
                    {
                        "id": "mutual_nearest_neighbor_1to1",
                        "pairing": "mutual nearest-neighbor pairing within |dt-offset|<=window",
                        "window_ns": float(frozen_window_ns),
                        "statistic_name": "CHSH |S| (fixed variant)",
                        "statistic_abs": s_m_abs,
                        "pairs_total": int(i_pairs.size),
                    },
                ]
                payload["delta"] = {
                    "abs_delta_statistic_abs": abs(float(s_m_abs) - float(s_g_abs)) if (s_g_abs is not None and s_m_abs is not None) else None,
                    "delta_over_sigma_boot": delta_over_sigma,
                    "note": "Treat this pairing-choice delta as an implementation/systematic sensitivity (not an optimizer).",
                }
                payload["inputs"]["normalized_events_npz"] = {
                    "path": _relpath_from_root(ds_dir / "normalized_events.npz"),
                    "sha256": _sha256_or_none(ds_dir / "normalized_events.npz"),
                }
                payload["inputs"]["normalized_events_json"] = {
                    "path": _relpath_from_root(ds_dir / "normalized_events.json"),
                    "sha256": _sha256_or_none(ds_dir / "normalized_events.json"),
                }
            except Exception as exc:
                payload["supported"] = False
                payload["reason"] = str(exc)
        # 条件分岐: 前段条件が不成立で、`ds.startswith("nist_")` を追加評価する。
        elif ds.startswith("nist_"):
            # NIST time-tag: CH J_prob depends on coincidence pairing convention (greedy vs mutual nearest).
            try:
                tb_path = ds_dir / "trial_based_counts.json"
                w_path = ds_dir / "window_sweep_metrics.json"
                norm_npz = ds_dir / "normalized_events.npz"
                norm_json = ds_dir / "normalized_events.json"
                # 条件分岐: `not tb_path.exists()` を満たす経路を評価する。
                if not tb_path.exists():
                    raise FileNotFoundError(tb_path)

                # 条件分岐: `not w_path.exists()` を満たす経路を評価する。

                if not w_path.exists():
                    raise FileNotFoundError(w_path)

                # 条件分岐: `not norm_npz.exists()` を満たす経路を評価する。

                if not norm_npz.exists():
                    raise FileNotFoundError(norm_npz)

                tb = _read_json(tb_path)
                tb_counts = tb.get("counts") if isinstance(tb.get("counts"), dict) else {}
                n_trials = np.asarray(tb_counts.get("trials_by_setting_pair"), dtype=np.int64)
                n_trials_a = np.asarray(tb_counts.get("alice_trials_by_setting"), dtype=np.int64)
                n_trials_b = np.asarray(tb_counts.get("bob_trials_by_setting"), dtype=np.int64)
                n_click_a = np.asarray(tb_counts.get("alice_clicks_by_setting"), dtype=np.int64)
                n_click_b = np.asarray(tb_counts.get("bob_clicks_by_setting"), dtype=np.int64)
                a1, b1 = 0, 0

                wj = _read_json(w_path)
                rows = wj.get("rows") if isinstance(wj.get("rows"), list) else []
                row0 = None
                # 条件分岐: `frozen_window_ns is not None` を満たす経路を評価する。
                if frozen_window_ns is not None:
                    for row in rows:
                        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
                        if not isinstance(row, dict):
                            continue

                        w_ns = _safe_float(row.get("window_ns"))
                        # 条件分岐: `w_ns is None` を満たす経路を評価する。
                        if w_ns is None:
                            continue

                        # 条件分岐: `abs(float(w_ns) - float(frozen_window_ns)) < 1e-9` を満たす経路を評価する。

                        if abs(float(w_ns) - float(frozen_window_ns)) < 1e-9:
                            row0 = row
                            break

                # 条件分岐: `row0 is None` を満たす経路を評価する。

                if row0 is None:
                    raise RuntimeError(f"window_sweep_metrics missing frozen window row: window_ns={frozen_window_ns}")

                c_g = np.asarray(row0.get("coinc_by_setting_pair"), dtype=np.int64)
                pairs_g = int(row0.get("pairs_total")) if row0.get("pairs_total") is not None else int(np.sum(c_g))
                j_g = _safe_float(row0.get("J_prob"))
                # 条件分岐: `j_g is None` を満たす経路を評価する。
                if j_g is None:
                    j_g = _compute_ch_j_prob(
                        n_trials=n_trials,
                        n_coinc=c_g,
                        n_trials_a=n_trials_a,
                        n_trials_b=n_trials_b,
                        n_click_a=n_click_a,
                        n_click_b=n_click_b,
                        a1=a1,
                        b1=b1,
                    )

                j_g = float(j_g)

                data = np.load(norm_npz)
                seconds_per_timetag = float(np.asarray(data["seconds_per_timetag"]).reshape(-1)[0])
                window_counts = int(float(frozen_window_ns) * 1e-9 / seconds_per_timetag)
                ta = data["alice_click_t"].astype(np.int64, copy=False)
                tb2 = data["bob_click_t"].astype(np.int64, copy=False)
                a_set = data["alice_click_setting"].astype(np.int8, copy=False)
                b_set = data["bob_click_setting"].astype(np.int8, copy=False)
                i_pairs, j_pairs = _mutual_nearest_pairs_counts_chunked(
                    ta,
                    tb2,
                    offset_counts=0,
                    window_counts=window_counts,
                    chunk_size=1_000_000,
                )

                c_m = np.zeros((2, 2), dtype=np.int64)
                # 条件分岐: `int(i_pairs.size)` を満たす経路を評価する。
                if int(i_pairs.size):
                    np.add.at(c_m, (a_set[i_pairs], b_set[j_pairs]), 1)

                pairs_m = int(i_pairs.size)
                j_m = _compute_ch_j_prob(
                    n_trials=n_trials,
                    n_coinc=c_m,
                    n_trials_a=n_trials_a,
                    n_trials_b=n_trials_b,
                    n_click_a=n_click_a,
                    n_click_b=n_click_b,
                    a1=a1,
                    b1=b1,
                )

                delta_over_sigma = None
                try:
                    sigma_v = float(sigma_boot)
                    # 条件分岐: `sigma_v > 0 and math.isfinite(float(j_g)) and math.isfinite(float(j_m))` を満たす経路を評価する。
                    if sigma_v > 0 and math.isfinite(float(j_g)) and math.isfinite(float(j_m)):
                        delta_over_sigma = abs(float(j_m) - float(j_g)) / sigma_v
                except Exception:
                    delta_over_sigma = None

                payload["supported"] = True
                payload["methods"] = [
                    {
                        "id": "greedy_time_order_1to1",
                        "pairing": "greedy 1:1 pairing by time order within |dt|<=window",
                        "window_ns": float(frozen_window_ns),
                        "window_counts": int(window_counts),
                        "statistic_name": "CH J_prob (A1=0,B1=0)",
                        "statistic": float(j_g),
                        "pairs_total": int(pairs_g),
                        "coinc_by_setting_pair": c_g.astype(int).tolist(),
                    },
                    {
                        "id": "mutual_nearest_neighbor_1to1",
                        "pairing": "mutual nearest-neighbor pairing within |dt|<=window",
                        "window_ns": float(frozen_window_ns),
                        "window_counts": int(window_counts),
                        "statistic_name": "CH J_prob (A1=0,B1=0)",
                        "statistic": float(j_m),
                        "pairs_total": int(pairs_m),
                        "coinc_by_setting_pair": c_m.astype(int).tolist(),
                    },
                ]
                payload["delta"] = {
                    "abs_delta_statistic": abs(float(j_m) - float(j_g)),
                    "delta_over_sigma_boot": delta_over_sigma,
                    "note": "Treat this pairing-choice delta as an implementation/systematic sensitivity (not an optimizer).",
                }
                payload["inputs"]["trial_based_counts_json"] = {"path": _relpath_from_root(tb_path), "sha256": _sha256_or_none(tb_path)}
                payload["inputs"]["window_sweep_metrics_json"] = {"path": _relpath_from_root(w_path), "sha256": _sha256_or_none(w_path)}
                payload["inputs"]["normalized_events_npz"] = {"path": _relpath_from_root(norm_npz), "sha256": _sha256_or_none(norm_npz)}
                # 条件分岐: `norm_json.exists()` を満たす経路を評価する。
                if norm_json.exists():
                    payload["inputs"]["normalized_events_json"] = {"path": _relpath_from_root(norm_json), "sha256": _sha256_or_none(norm_json)}
            except Exception as exc:
                payload["supported"] = False
                payload["reason"] = str(exc)
        # 条件分岐: 前段条件が不成立で、`ds.startswith("kwiat2013_")` を追加評価する。
        elif ds.startswith("kwiat2013_"):
            # Kwiat/Christensen 2013: trial-based CH J_prob; pairing is not a coincidence-window problem.
            # Here we at least cross-check internal consistency of the frozen ref-window counts vs sweep row.
            try:
                tb_path = ds_dir / "trial_based_counts.json"
                w_path = ds_dir / "window_sweep_metrics.json"
                # 条件分岐: `not tb_path.exists()` を満たす経路を評価する。
                if not tb_path.exists():
                    raise FileNotFoundError(tb_path)

                # 条件分岐: `not w_path.exists()` を満たす経路を評価する。

                if not w_path.exists():
                    raise FileNotFoundError(w_path)

                tb = _read_json(tb_path)
                tb_counts = tb.get("counts") if isinstance(tb.get("counts"), dict) else {}
                n_trials = np.asarray(tb_counts.get("trials_by_setting_pair"), dtype=np.int64)
                n_trials_a = np.asarray(tb_counts.get("alice_trials_by_setting"), dtype=np.int64)
                n_trials_b = np.asarray(tb_counts.get("bob_trials_by_setting"), dtype=np.int64)
                n_click_a = np.asarray(tb_counts.get("alice_clicks_by_setting"), dtype=np.int64)
                n_click_b = np.asarray(tb_counts.get("bob_clicks_by_setting"), dtype=np.int64)
                c_tb = np.asarray(tb_counts.get("coinc_by_setting_pair"), dtype=np.int64)
                a1, b1 = 0, 0
                j_tb = _compute_ch_j_prob(
                    n_trials=n_trials,
                    n_coinc=c_tb,
                    n_trials_a=n_trials_a,
                    n_trials_b=n_trials_b,
                    n_click_a=n_click_a,
                    n_click_b=n_click_b,
                    a1=a1,
                    b1=b1,
                )

                wj = _read_json(w_path)
                rows = wj.get("rows") if isinstance(wj.get("rows"), list) else []
                row0 = None
                # 条件分岐: `frozen_window_ns is not None` を満たす経路を評価する。
                if frozen_window_ns is not None:
                    for row in rows:
                        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
                        if not isinstance(row, dict):
                            continue

                        w_ns = _safe_float(row.get("window_ns"))
                        # 条件分岐: `w_ns is None` を満たす経路を評価する。
                        if w_ns is None:
                            continue

                        # 条件分岐: `abs(float(w_ns) - float(frozen_window_ns)) < 1e-9` を満たす経路を評価する。

                        if abs(float(w_ns) - float(frozen_window_ns)) < 1e-9:
                            row0 = row
                            break

                # 条件分岐: `row0 is None` を満たす経路を評価する。

                if row0 is None:
                    raise RuntimeError(f"window_sweep_metrics missing frozen window row: window_ns={frozen_window_ns}")

                c_row = np.asarray(row0.get("coinc_by_setting_pair"), dtype=np.int64)
                pairs_row = int(row0.get("pairs_total")) if row0.get("pairs_total") is not None else int(np.sum(c_row))
                j_row = _safe_float(row0.get("J_prob"))
                # 条件分岐: `j_row is None` を満たす経路を評価する。
                if j_row is None:
                    j_row = _compute_ch_j_prob(
                        n_trials=n_trials,
                        n_coinc=c_row,
                        n_trials_a=n_trials_a,
                        n_trials_b=n_trials_b,
                        n_click_a=n_click_a,
                        n_click_b=n_click_b,
                        a1=a1,
                        b1=b1,
                    )

                j_row = float(j_row)

                delta_over_sigma = None
                try:
                    sigma_v = float(sigma_boot)
                    # 条件分岐: `sigma_v > 0 and math.isfinite(float(j_tb)) and math.isfinite(float(j_row))` を満たす経路を評価する。
                    if sigma_v > 0 and math.isfinite(float(j_tb)) and math.isfinite(float(j_row)):
                        delta_over_sigma = abs(float(j_row) - float(j_tb)) / sigma_v
                except Exception:
                    delta_over_sigma = None

                payload["supported"] = True
                payload["methods"] = [
                    {
                        "id": "trial_based_counts_ref_window",
                        "pairing": "trial-based hit counting at ref window (as parsed from raw .mat)",
                        "window_ns": float(frozen_window_ns),
                        "statistic_name": "CH J_prob (A1=0,B1=0)",
                        "statistic": float(j_tb),
                        "pairs_total": int(np.sum(c_tb)),
                        "coinc_by_setting_pair": c_tb.astype(int).tolist(),
                    },
                    {
                        "id": "window_sweep_row_frozen_window",
                        "pairing": "window sweep row at frozen window (consistency check)",
                        "window_ns": float(frozen_window_ns),
                        "statistic_name": "CH J_prob (A1=0,B1=0)",
                        "statistic": float(j_row),
                        "pairs_total": int(pairs_row),
                        "coinc_by_setting_pair": c_row.astype(int).tolist(),
                    },
                ]
                payload["delta"] = {
                    "abs_delta_statistic": abs(float(j_row) - float(j_tb)),
                    "delta_over_sigma_boot": delta_over_sigma,
                    "note": "Internal consistency check for trial-based CH J_prob at the frozen window (not a coincidence-pairing convention).",
                }
                payload["inputs"]["trial_based_counts_json"] = {"path": _relpath_from_root(tb_path), "sha256": _sha256_or_none(tb_path)}
                payload["inputs"]["window_sweep_metrics_json"] = {"path": _relpath_from_root(w_path), "sha256": _sha256_or_none(w_path)}
            except Exception as exc:
                payload["supported"] = False
                payload["reason"] = str(exc)
        else:
            payload["supported"] = False
            payload["reason"] = "pairing crosscheck not implemented for this dataset type"

        _write_json(out_path, payload)
        ds_summaries.append(
            {
                "dataset_id": ds,
                "display_name": _dataset_display_name(ds),
                "supported": bool(payload.get("supported")),
                "crosscheck_pairing_json": {"path": _relpath_from_root(out_path), "sha256": _sha256_or_none(out_path)},
                "delta": payload.get("delta"),
            }
        )

    summary = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 7, "step": "7.20.3", "name": "Bell pairing/estimator cross-checks"},
        "main_script": {"path": "scripts/quantum/bell_primary_products.py", "repro": "python -B scripts/quantum/bell_primary_products.py"},
        "datasets": ds_summaries,
        "outputs": {"crosscheck_pairing_summary_json": _relpath_from_root(summary_path)},
    }
    _write_json(summary_path, summary)
    return summary


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 7 / Step 7.16.7-7.16.10: Bell-test primary products + covariance "
            "(bootstrap/jackknife/eigen), 15-item systematics decomposition, longterm "
            "cross-dataset quantification, loophole quantification, and falsification pack."
        )
    )
    ap.add_argument("--datasets", default="all", help="Comma-separated dataset ids to run (default: all).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    want = [s.strip() for s in str(args.datasets).split(",") if s.strip()]
    # 条件分岐: `want == ["all"]` を満たす経路を評価する。
    if want == ["all"]:
        want = [
            "weihs1998_longdist_longdist1",
            "delft_hensen2015",
            "delft_hensen2016_srep30289",
            "nist_03_43_afterfixingModeLocking_s3600",
            "kwiat2013_prl111_130406_05082013_15",
        ]

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    # 条件分岐: `"weihs1998_longdist_longdist1" in want` を満たす経路を評価する。
    if "weihs1998_longdist_longdist1" in want:
        print("[run] weihs1998_longdist_longdist1")
        results.append(_weihs1998_dataset(dataset_id="weihs1998_longdist_longdist1", overwrite=bool(args.overwrite)))

    # 条件分岐: `any(x.startswith("delft_") for x in want)` を満たす経路を評価する。

    if any(x.startswith("delft_") for x in want):
        print("[run] delft (2015/2016)")
        for r in _delft_datasets(overwrite=bool(args.overwrite)):
            # 条件分岐: `r.get("dataset_id") in want` を満たす経路を評価する。
            if r.get("dataset_id") in want:
                results.append(r)

    # 条件分岐: `"nist_03_43_afterfixingModeLocking_s3600" in want` を満たす経路を評価する。

    if "nist_03_43_afterfixingModeLocking_s3600" in want:
        print("[run] nist_03_43_afterfixingModeLocking_s3600")
        results.append(_nist_dataset(overwrite=bool(args.overwrite)))

    # 条件分岐: `"kwiat2013_prl111_130406_05082013_15" in want` を満たす経路を評価する。

    if "kwiat2013_prl111_130406_05082013_15" in want:
        print("[run] kwiat2013_prl111_130406_05082013_15")
        results.append(
            _kwiat2013_dataset(dataset_id="kwiat2013_prl111_130406_05082013_15", overwrite=bool(args.overwrite))
        )

    _write_json(OUT_BASE / "table1_row.json", _build_table1_row(results=results))
    falsification_pack = _build_falsification_pack(results=results)
    _write_json(OUT_BASE / "falsification_pack.json", falsification_pack)
    _write_covariance_products(results=results)
    _write_freeze_policy(results=results)
    _write_null_tests(results=results)
    _write_pairing_crosschecks(results=results)
    _enrich_falsification_pack_cross_dataset(pack_path=OUT_BASE / "falsification_pack.json")

    # Optional: also emit a quick aggregate plot (ratios + KS)
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        pass
    else:
        thresholds = falsification_pack.get("thresholds") if isinstance(falsification_pack.get("thresholds"), dict) else {}
        ratio_th = float(thresholds.get("selection_origin_ratio_min", 1.0))
        delay_z_th = float(thresholds.get("delay_signature_z_min", 3.0))

        ds_list = falsification_pack.get("datasets") if isinstance(falsification_pack.get("datasets"), list) else []
        labels = [_dataset_display_name(str(d.get("dataset_id") or "")) for d in ds_list if isinstance(d, dict)]
        ratios = [
            float(d["ratio"]) if d.get("ratio") is not None and math.isfinite(float(d["ratio"])) else float("nan")
            for d in ds_list
            if isinstance(d, dict)
        ]

        zvals: list[float] = []
        zcolors: list[str] = []
        z_is_na: list[bool] = []
        for d in ds_list:
            # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
            if not isinstance(d, dict):
                continue

            z = _delay_signature_z_max(d.get("delay_signature"))
            # 条件分岐: `z is None` を満たす経路を評価する。
            if z is None:
                zvals.append(0.0)
                zcolors.append("0.85")
                z_is_na.append(True)
            else:
                zvals.append(float(z))
                zcolors.append("tab:orange")
                z_is_na.append(False)

        fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.2), dpi=170)
        x = np.arange(len(labels))
        ax[0].bar(x, ratios, color="tab:blue", alpha=0.85)
        ax[0].axhline(ratio_th, color="0.2", ls="--", lw=1.0)
        ax[0].set_xticks(x, labels, rotation=0, ha="center")
        ax[0].set_ylabel("Δ(stat) / σ_stat (median)")
        ax[0].set_title("Selection sensitivity (ratio)")
        ax[0].grid(True, axis="y", alpha=0.3, ls=":")

        ax[1].bar(x, zvals, color=zcolors, alpha=0.9)
        ax[1].axhline(delay_z_th, color="0.2", ls="--", lw=1.0)
        ax[1].set_xticks(x, labels, rotation=0, ha="center")
        ax[1].set_ylabel("z = ∣Δmedian∣ / σ(Δmedian)")
        ax[1].set_title("Delay setting-dependence (Δmedian; z)")
        ax[1].grid(True, axis="y", alpha=0.3, ls=":")
        for i, is_na in enumerate(z_is_na):
            # 条件分岐: `is_na` を満たす経路を評価する。
            if is_na:
                ax[1].text(float(i), 0.15, "n/a", ha="center", va="bottom", fontsize=9, color="0.35")

        fig.suptitle("Bell falsification pack (operational thresholds)", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_BASE / "falsification_pack.png", bbox_inches="tight")
        plt.close(fig)

    print(f"[ok] wrote: {OUT_BASE / 'table1_row.json'}")
    print(f"[ok] wrote: {OUT_BASE / 'falsification_pack.json'}")
    print(f"[ok] wrote: {OUT_BASE / 'selection_loophole_quantification.json'}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
