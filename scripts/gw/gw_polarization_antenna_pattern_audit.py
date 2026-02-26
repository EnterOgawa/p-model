from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

_C = 299_792_458.0
_R_E = 6_378_137.0

_DETECTOR_SITES = {
    "H1": {
        "lat_deg": 46.4551466667,
        "lon_deg": -119.4076571389,
        "xarm_az_deg": 125.9994,
    },
    "L1": {
        "lat_deg": 30.5628943333,
        "lon_deg": -90.7742403889,
        "xarm_az_deg": 197.7165,
    },
    "V1": {
        "lat_deg": 43.6314144722,
        "lon_deg": 10.5044966111,
        "xarm_az_deg": 70.5674,
    },
}


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_slugify` の入出力契約と処理意図を定義する。

def _slugify(s: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "").strip())
    while "__" in out:
        out = out.replace("__", "_")

    return out.strip("_") or "event"


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")

    return v if math.isfinite(v) else float("nan")


# 関数: `_detector_order` の入出力契約と処理意図を定義する。

def _detector_order(det: str) -> int:
    order = {"H1": 0, "L1": 1, "V1": 2, "K1": 3}
    return int(order.get(str(det).upper(), 999))


# 関数: `_canonical_pair` の入出力契約と処理意図を定義する。

def _canonical_pair(det_a: str, det_b: str) -> Tuple[str, str]:
    da = str(det_a).upper().strip()
    db = str(det_b).upper().strip()
    # 条件分岐: `da == db` を満たす経路を評価する。
    if da == db:
        return da, db

    pair = sorted([da, db], key=lambda d: (_detector_order(d), d))
    return pair[0], pair[1]


# 関数: `_fmt` の入出力契約と処理意図を定義する。

def _fmt(v: float, digits: int = 7) -> str:
    # 条件分岐: `not math.isfinite(float(v))` を満たす経路を評価する。
    if not math.isfinite(float(v)):
        return ""

    x = float(v)
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_site_geometry` の入出力契約と処理意図を定義する。

def _site_geometry(lat_deg: float, lon_deg: float, xarm_az_deg: float) -> Dict[str, np.ndarray]:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    az = math.radians(float(xarm_az_deg))

    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)], dtype=np.float64)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64)
    north = np.array(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        dtype=np.float64,
    )
    xarm = math.cos(az) * north + math.sin(az) * east
    yarm = math.cos(az + 0.5 * math.pi) * north + math.sin(az + 0.5 * math.pi) * east

    xarm /= np.linalg.norm(xarm)
    yarm /= np.linalg.norm(yarm)
    pos = _R_E * up
    tensor = 0.5 * (np.outer(xarm, xarm) - np.outer(yarm, yarm))
    return {
        "position_m": pos,
        "xarm": xarm,
        "yarm": yarm,
        "tensor": tensor,
    }


# 関数: `_build_network_geometry` の入出力契約と処理意図を定義する。

def _build_network_geometry() -> Dict[str, Dict[str, np.ndarray]]:
    return {det: _site_geometry(**cfg) for det, cfg in _DETECTOR_SITES.items()}


# 関数: `_fibonacci_sphere` の入出力契約と処理意図を定義する。

def _fibonacci_sphere(n: int) -> np.ndarray:
    n_use = int(max(64, n))
    idx = np.arange(n_use, dtype=np.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_use)
    phi = (2.0 * math.pi * idx) / ((1.0 + math.sqrt(5.0)) / 2.0)
    r_xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    return np.stack([x, y, z], axis=1)


# 関数: `_load_metrics` の入出力契約と処理意図を定義する。

def _load_metrics(event: str, slug: str, det_first: str = "H1", det_second: str = "L1") -> Optional[Dict[str, Any]]:
    pair_stem = f"{_slugify(det_first)}_{_slugify(det_second)}"
    legacy_stem = "h1_l1" if (det_first, det_second) == ("H1", "L1") else None
    names = [f"{slug}_{pair_stem}_amplitude_ratio_metrics.json"]
    # 条件分岐: `legacy_stem is not None` を満たす経路を評価する。
    if legacy_stem is not None:
        names.append(f"{slug}_{legacy_stem}_amplitude_ratio_metrics.json")

    candidates: List[Path] = []
    for base in ["public", "private"]:
        for name in names:
            candidates.append(_ROOT / "output" / base / "gw" / name)

    for path in candidates:
        # 条件分岐: `path.exists()` を満たす経路を評価する。
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["_metrics_path"] = str(path).replace("\\", "/")
                payload["_event_requested"] = event
                payload["_pair"] = [det_first, det_second]
                return payload
            except Exception:
                continue

    return None


# 関数: `_load_lag_scan` の入出力契約と処理意図を定義する。

def _load_lag_scan(slug: str, det_first: str = "H1", det_second: str = "L1") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pair_stem = f"{_slugify(det_first)}_{_slugify(det_second)}"
    legacy_stem = "h1_l1" if (det_first, det_second) == ("H1", "L1") else None
    names = [f"{slug}_{pair_stem}_amplitude_ratio_lag_scan.csv"]
    # 条件分岐: `legacy_stem is not None` を満たす経路を評価する。
    if legacy_stem is not None:
        names.append(f"{slug}_{legacy_stem}_amplitude_ratio_lag_scan.csv")

    candidates: List[Path] = []
    for base in ["public", "private"]:
        for name in names:
            candidates.append(_ROOT / "output" / base / "gw" / name)

    for path in candidates:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            continue

        try:
            lags: List[float] = []
            corrs: List[float] = []
            with path.open("r", encoding="utf-8", newline="") as f:
                next(f, None)
                for line in f:
                    cells = [c.strip() for c in line.split(",")]
                    # 条件分岐: `len(cells) < 2` を満たす経路を評価する。
                    if len(cells) < 2:
                        continue

                    lag = _safe_float(cells[0])
                    corr = _safe_float(cells[1])
                    # 条件分岐: `math.isfinite(lag) and math.isfinite(corr)` を満たす経路を評価する。
                    if math.isfinite(lag) and math.isfinite(corr):
                        lags.append(lag)
                        corrs.append(corr)

            # 条件分岐: `lags` を満たす経路を評価する。

            if lags:
                return np.asarray(lags, dtype=np.float64), np.asarray(corrs, dtype=np.float64)
        except Exception:
            continue

    return None


# 関数: `_estimate_delay_tolerance_s` の入出力契約と処理意図を定義する。

def _estimate_delay_tolerance_s(
    *,
    best_lag_ms: float,
    best_abs_corr: float,
    fs_hz: float,
    lag_scan: Optional[Tuple[np.ndarray, np.ndarray]],
) -> float:
    tol_floor = max(0.25e-3, 2.0 / fs_hz if math.isfinite(fs_hz) and fs_hz > 0.0 else 0.25e-3)
    # 条件分岐: `lag_scan is None or not math.isfinite(best_abs_corr)` を満たす経路を評価する。
    if lag_scan is None or not math.isfinite(best_abs_corr):
        return float(tol_floor)

    lags_ms, corrs = lag_scan
    # 条件分岐: `lags_ms.size < 3` を満たす経路を評価する。
    if lags_ms.size < 3:
        return float(tol_floor)

    thr = 0.95 * float(best_abs_corr)
    mask = np.abs(corrs) >= float(thr)
    # 条件分岐: `int(np.sum(mask)) < 2` を満たす経路を評価する。
    if int(np.sum(mask)) < 2:
        return float(tol_floor)

    span_ms = float(np.max(lags_ms[mask]) - np.min(lags_ms[mask]))
    return float(max(tol_floor, 0.5 * span_ms * 1e-3))


# 関数: `_direction_basis` の入出力契約と処理意図を定義する。

def _direction_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    theta = math.acos(max(-1.0, min(1.0, nz)))
    phi = math.atan2(ny, nx)
    e_theta = np.array([math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), -math.sin(theta)], dtype=np.float64)
    e_phi = np.array([-math.sin(phi), math.cos(phi), 0.0], dtype=np.float64)
    # 条件分岐: `np.linalg.norm(e_phi) < 1e-12` を満たす経路を評価する。
    if np.linalg.norm(e_phi) < 1e-12:
        e_phi = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e_theta /= np.linalg.norm(e_theta)
    e_phi /= np.linalg.norm(e_phi)
    return e_theta, e_phi


# 関数: `_response_grid_for_direction` の入出力契約と処理意図を定義する。

def _response_grid_for_direction(
    *,
    n: np.ndarray,
    tensor_h: np.ndarray,
    tensor_l: np.ndarray,
    psi_grid: np.ndarray,
    cosi_grid: np.ndarray,
    response_floor_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    e_theta, e_phi = _direction_basis(n)
    cpsi = np.cos(psi_grid)
    spsi = np.sin(psi_grid)
    p = cpsi[:, None] * e_theta[None, :] + spsi[:, None] * e_phi[None, :]
    q = -spsi[:, None] * e_theta[None, :] + cpsi[:, None] * e_phi[None, :]

    eplus = np.einsum("ai,aj->aij", p, p) - np.einsum("ai,aj->aij", q, q)
    ecross = np.einsum("ai,aj->aij", p, q) + np.einsum("ai,aj->aij", q, p)
    eb = np.einsum("ai,aj->aij", p, p) + np.einsum("ai,aj->aij", q, q)

    fp_h = np.einsum("ij,aij->a", tensor_h, eplus)
    fx_h = np.einsum("ij,aij->a", tensor_h, ecross)
    fb_h = np.einsum("ij,aij->a", tensor_h, eb)
    fp_l = np.einsum("ij,aij->a", tensor_l, eplus)
    fx_l = np.einsum("ij,aij->a", tensor_l, ecross)
    fb_l = np.einsum("ij,aij->a", tensor_l, eb)

    floor_frac = float(max(0.0, min(0.5, response_floor_frac)))
    tensor_ratios: List[float] = []
    for cosi in cosi_grid:
        hp = 0.5 * (1.0 + float(cosi) * float(cosi))
        hx = float(cosi)
        amp_h = np.sqrt((fp_h * hp) ** 2 + (fx_h * hx) ** 2)
        amp_l = np.sqrt((fp_l * hp) ** 2 + (fx_l * hx) ** 2)
        floor_h = floor_frac * float(np.max(amp_h)) if amp_h.size else 0.0
        floor_l = floor_frac * float(np.max(amp_l)) if amp_l.size else 0.0
        den_ok = (amp_l > max(1e-10, floor_l)) & (amp_h > max(1e-10, floor_h))
        # 条件分岐: `np.any(den_ok)` を満たす経路を評価する。
        if np.any(den_ok):
            tensor_ratios.extend((amp_h[den_ok] / amp_l[den_ok]).tolist())

    abs_fbh = np.abs(fb_h)
    abs_fbl = np.abs(fb_l)
    floor_bh = floor_frac * float(np.max(abs_fbh)) if abs_fbh.size else 0.0
    floor_bl = floor_frac * float(np.max(abs_fbl)) if abs_fbl.size else 0.0
    scalar_ok = (abs_fbl > max(1e-10, floor_bl)) & (abs_fbh > max(1e-10, floor_bh))
    scalar_ratios = np.abs(fb_h[scalar_ok] / fb_l[scalar_ok]) if np.any(scalar_ok) else np.array([], dtype=np.float64)
    return np.asarray(tensor_ratios, dtype=np.float64), np.asarray(scalar_ratios, dtype=np.float64)


# 関数: `_range_clip` の入出力契約と処理意図を定義する。

def _range_clip(arr: np.ndarray, q_lo: float = 0.5, q_hi: float = 99.5) -> Tuple[float, float]:
    # 条件分岐: `arr.size == 0` を満たす経路を評価する。
    if arr.size == 0:
        return float("nan"), float("nan")

    return float(np.percentile(arr, q_lo)), float(np.percentile(arr, q_hi))


# 関数: `_interval_overlap` の入出力契約と処理意図を定義する。

def _interval_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    # 条件分岐: `not (math.isfinite(a_lo) and math.isfinite(a_hi) and math.isfinite(b_lo) and...` を満たす経路を評価する。
    if not (math.isfinite(a_lo) and math.isfinite(a_hi) and math.isfinite(b_lo) and math.isfinite(b_hi)):
        return False

    return not (a_hi < b_lo or b_hi < a_lo)


# 関数: `_min_rel_mismatch` の入出力契約と処理意図を定義する。

def _min_rel_mismatch(arr: np.ndarray, target: float) -> float:
    # 条件分岐: `arr.size == 0 or not math.isfinite(target)` を満たす経路を評価する。
    if arr.size == 0 or not math.isfinite(target):
        return float("nan")

    den = abs(float(target)) + 1e-15
    return float(np.min(np.abs(arr - float(target)) / den))


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "event",
        "slug",
        "quality",
        "detectors",
        "usable_pair_count",
        "pair_names",
        "pair_tensor_overlap_count",
        "pair_scalar_overlap_count",
        "status",
        "status_reason",
        "best_lag_ms",
        "delay_h1_minus_l1_ms",
        "delay_tolerance_ms",
        "abs_corr",
        "obs_ratio_p16",
        "obs_ratio_median",
        "obs_ratio_p84",
        "tensor_ratio_lo",
        "tensor_ratio_hi",
        "scalar_ratio_lo",
        "scalar_ratio_hi",
        "tensor_interval_overlap",
        "scalar_interval_overlap",
        "tensor_min_rel_mismatch",
        "scalar_min_rel_mismatch",
        "ring_directions_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            vals: List[Any] = []
            for h in headers:
                v = row.get(h, "")
                # 条件分岐: `isinstance(v, float)` を満たす経路を評価する。
                if isinstance(v, float):
                    vals.append(_fmt(v))
                else:
                    vals.append(v)

            w.writerow(vals)


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(rows: List[Dict[str, Any]], out_png: Path) -> None:
    _set_japanese_font()
    labels = [str(r.get("event", "")) for r in rows]
    y = np.arange(len(rows), dtype=float)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13.5, 8.6), gridspec_kw={"height_ratios": [2.2, 1.0]})

    for i, row in enumerate(rows):
        obs_lo = _safe_float(row.get("obs_ratio_p16"))
        obs_md = _safe_float(row.get("obs_ratio_median"))
        obs_hi = _safe_float(row.get("obs_ratio_p84"))
        t_lo = _safe_float(row.get("tensor_ratio_lo"))
        t_hi = _safe_float(row.get("tensor_ratio_hi"))
        s_lo = _safe_float(row.get("scalar_ratio_lo"))
        s_hi = _safe_float(row.get("scalar_ratio_hi"))
        status = str(row.get("status", "unknown"))

        # 条件分岐: `math.isfinite(t_lo) and math.isfinite(t_hi)` を満たす経路を評価する。
        if math.isfinite(t_lo) and math.isfinite(t_hi):
            ax0.hlines(y[i] + 0.16, t_lo, t_hi, color="#2ca02c", linewidth=6.0, alpha=0.8, label="tensor range" if i == 0 else None)

        # 条件分岐: `math.isfinite(s_lo) and math.isfinite(s_hi)` を満たす経路を評価する。

        if math.isfinite(s_lo) and math.isfinite(s_hi):
            ax0.hlines(y[i] - 0.16, s_lo, s_hi, color="#f58518", linewidth=6.0, alpha=0.8, label="scalar breathing range" if i == 0 else None)

        # 条件分岐: `math.isfinite(obs_lo) and math.isfinite(obs_hi)` を満たす経路を評価する。

        if math.isfinite(obs_lo) and math.isfinite(obs_hi):
            ax0.hlines(y[i], obs_lo, obs_hi, color="#1f77b4", linewidth=4.0, alpha=0.95, label="observed p16-p84" if i == 0 else None)

        # 条件分岐: `math.isfinite(obs_md)` を満たす経路を評価する。

        if math.isfinite(obs_md):
            ax0.plot([obs_md], [y[i]], marker="o", color="#0b3b8c", markersize=5.5)

        ax0.text(
            1.01,
            y[i],
            status,
            va="center",
            ha="left",
            transform=ax0.get_yaxis_transform(),
            fontsize=9,
            color="#222222",
        )

    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=9)
    ax0.invert_yaxis()
    ax0.set_xlabel("amplitude ratio |H1|/|L1|")
    ax0.set_title("Antenna-pattern compatibility: observed vs tensor/scalar response ranges")
    ax0.grid(True, axis="x", alpha=0.25)
    ax0.legend(loc="lower right", fontsize=9)

    t_mis = np.array([_safe_float(r.get("tensor_min_rel_mismatch")) for r in rows], dtype=float)
    s_mis = np.array([_safe_float(r.get("scalar_min_rel_mismatch")) for r in rows], dtype=float)
    x = np.arange(len(rows), dtype=float)
    w = 0.36
    ax1.bar(x - w / 2.0, t_mis, width=w, color="#2ca02c", alpha=0.85, label="tensor")
    ax1.bar(x + w / 2.0, s_mis, width=w, color="#f58518", alpha=0.85, label="scalar breathing")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=9)
    ax1.set_ylabel("min relative mismatch to ratio median")
    ax1.set_xlabel("event")
    ax1.set_title("Best-fit mismatch proxy (smaller is better)")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)

    fig.suptitle("GW polarization antenna-pattern audit (Step 8.7.19.2)", fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 8.7.19.2: detector antenna-pattern polarization audit.")
    ap.add_argument(
        "--events",
        type=str,
        default="GW150914,GW151226,GW170104,GW200129_065458",
        help="Comma-separated event names (must have H1/L1 amplitude metrics).",
    )
    ap.add_argument("--corr-use-min", type=float, default=0.6)
    ap.add_argument("--sky-samples", type=int, default=3000)
    ap.add_argument("--psi-samples", type=int, default=36)
    ap.add_argument("--cosi-samples", type=int, default=41)
    ap.add_argument(
        "--response-floor-frac",
        type=float,
        default=0.05,
        help="Minimum response floor fraction to avoid near-null singular ratio tails.",
    )
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    ap.add_argument("--prefix", type=str, default="gw_polarization_antenna_pattern_audit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    events = [s.strip() for s in str(args.events).split(",") if s.strip()]
    # 条件分岐: `not events` を満たす経路を評価する。
    if not events:
        print("[err] --events is empty")
        return 2

    geom = _build_network_geometry()
    pos_h = geom["H1"]["position_m"]
    pos_l = geom["L1"]["position_m"]
    tensor_h = geom["H1"]["tensor"]
    tensor_l = geom["L1"]["tensor"]
    baseline_hl = pos_h - pos_l
    baseline_norm = float(np.linalg.norm(baseline_hl))

    sky_dirs = _fibonacci_sphere(int(args.sky_samples))
    psi_grid = np.linspace(0.0, math.pi, int(max(8, args.psi_samples)), endpoint=False, dtype=np.float64)
    cosi_grid = np.linspace(-1.0, 1.0, int(max(9, args.cosi_samples)), dtype=np.float64)

    rows: List[Dict[str, Any]] = []
    for event in events:
        slug = _slugify(event)
        payload = _load_metrics(event, slug)
        # 条件分岐: `payload is None` を満たす経路を評価する。
        if payload is None:
            rows.append(
                {
                    "event": event,
                    "slug": slug,
                    "quality": "missing",
                    "status": "missing",
                    "status_reason": "metrics_not_found",
                }
            )
            continue

        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        ratio = metrics.get("ratio") if isinstance(metrics.get("ratio"), dict) else {}
        inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
        best_lag_ms = _safe_float(metrics.get("best_lag_ms_apply_to_h1"))
        abs_corr = _safe_float(metrics.get("abs_best_corr"))
        obs_p16 = _safe_float(ratio.get("p16"))
        obs_med = _safe_float(ratio.get("median"))
        obs_p84 = _safe_float(ratio.get("p84"))
        fs_hz = _safe_float(inputs.get("analysis_fs_hz"))
        delay_hl_s = -best_lag_ms * 1e-3 if math.isfinite(best_lag_ms) else float("nan")
        lag_scan = _load_lag_scan(slug)
        delay_tol_s = _estimate_delay_tolerance_s(
            best_lag_ms=best_lag_ms,
            best_abs_corr=abs_corr,
            fs_hz=fs_hz,
            lag_scan=lag_scan,
        )

        # 条件分岐: `not (math.isfinite(abs_corr) and abs_corr >= float(args.corr_use_min))` を満たす経路を評価する。
        if not (math.isfinite(abs_corr) and abs_corr >= float(args.corr_use_min)):
            rows.append(
                {
                    "event": event,
                    "slug": slug,
                    "quality": "low_corr",
                    "status": "inconclusive_low_corr",
                    "status_reason": "abs_corr_below_threshold",
                    "best_lag_ms": best_lag_ms,
                    "delay_h1_minus_l1_ms": delay_hl_s * 1e3 if math.isfinite(delay_hl_s) else float("nan"),
                    "delay_tolerance_ms": delay_tol_s * 1e3,
                    "abs_corr": abs_corr,
                    "obs_ratio_p16": obs_p16,
                    "obs_ratio_median": obs_med,
                    "obs_ratio_p84": obs_p84,
                }
            )
            continue

        # 条件分岐: `not (math.isfinite(delay_hl_s) and math.isfinite(obs_med) and math.isfinite(o...` を満たす経路を評価する。

        if not (math.isfinite(delay_hl_s) and math.isfinite(obs_med) and math.isfinite(obs_p16) and math.isfinite(obs_p84)):
            rows.append(
                {
                    "event": event,
                    "slug": slug,
                    "quality": "usable",
                    "status": "inconclusive_missing_fields",
                    "status_reason": "missing_delay_or_ratio",
                    "best_lag_ms": best_lag_ms,
                    "delay_h1_minus_l1_ms": delay_hl_s * 1e3 if math.isfinite(delay_hl_s) else float("nan"),
                    "delay_tolerance_ms": delay_tol_s * 1e3,
                    "abs_corr": abs_corr,
                    "obs_ratio_p16": obs_p16,
                    "obs_ratio_median": obs_med,
                    "obs_ratio_p84": obs_p84,
                }
            )
            continue

        dot_vals = sky_dirs @ baseline_hl
        dt_pred = -dot_vals / _C
        ring_mask = np.abs(dt_pred - delay_hl_s) <= float(delay_tol_s)
        ring_dirs = sky_dirs[ring_mask]
        # 条件分岐: `ring_dirs.shape[0] < 8` を満たす経路を評価する。
        if ring_dirs.shape[0] < 8:
            expand = max(delay_tol_s, 0.25e-3) * 2.0
            ring_mask = np.abs(dt_pred - delay_hl_s) <= float(expand)
            ring_dirs = sky_dirs[ring_mask]

        # 条件分岐: `ring_dirs.shape[0] < 8` を満たす経路を評価する。

        if ring_dirs.shape[0] < 8:
            rows.append(
                {
                    "event": event,
                    "slug": slug,
                    "quality": "usable",
                    "status": "inconclusive_geometry",
                    "status_reason": "insufficient_ring_directions",
                    "best_lag_ms": best_lag_ms,
                    "delay_h1_minus_l1_ms": delay_hl_s * 1e3,
                    "delay_tolerance_ms": delay_tol_s * 1e3,
                    "abs_corr": abs_corr,
                    "obs_ratio_p16": obs_p16,
                    "obs_ratio_median": obs_med,
                    "obs_ratio_p84": obs_p84,
                    "ring_directions_used": int(ring_dirs.shape[0]),
                }
            )
            continue

        tensor_ratios_all: List[float] = []
        scalar_ratios_all: List[float] = []
        for n in ring_dirs:
            tensor_r, scalar_r = _response_grid_for_direction(
                n=n,
                tensor_h=tensor_h,
                tensor_l=tensor_l,
                psi_grid=psi_grid,
                cosi_grid=cosi_grid,
                response_floor_frac=float(args.response_floor_frac),
            )
            # 条件分岐: `tensor_r.size > 0` を満たす経路を評価する。
            if tensor_r.size > 0:
                tensor_ratios_all.extend(tensor_r.tolist())

            # 条件分岐: `scalar_r.size > 0` を満たす経路を評価する。

            if scalar_r.size > 0:
                scalar_ratios_all.extend(scalar_r.tolist())

        tensor_arr = np.asarray(tensor_ratios_all, dtype=np.float64)
        scalar_arr = np.asarray(scalar_ratios_all, dtype=np.float64)
        t_lo, t_hi = _range_clip(tensor_arr, 0.5, 99.5)
        s_lo, s_hi = _range_clip(scalar_arr, 0.5, 99.5)
        obs_lo = min(obs_p16, obs_p84)
        obs_hi = max(obs_p16, obs_p84)
        tensor_overlap = _interval_overlap(obs_lo, obs_hi, t_lo, t_hi)
        scalar_overlap = _interval_overlap(obs_lo, obs_hi, s_lo, s_hi)
        t_mis = _min_rel_mismatch(tensor_arr, obs_med)
        s_mis = _min_rel_mismatch(scalar_arr, obs_med)

        # 条件分岐: `not tensor_overlap` を満たす経路を評価する。
        if not tensor_overlap:
            status = "reject_tensor_response"
            reason = "observed_ratio_outside_tensor_range"
        # 条件分岐: 前段条件が不成立で、`tensor_overlap and not scalar_overlap` を追加評価する。
        elif tensor_overlap and not scalar_overlap:
            status = "pass_scalar_only_disfavored"
            reason = "observed_ratio_outside_scalar_range"
        else:
            status = "watch_scalar_not_excluded"
            reason = "tensor_and_scalar_overlap"

        rows.append(
            {
                "event": event,
                "slug": slug,
                "quality": "usable",
                "status": status,
                "status_reason": reason,
                "best_lag_ms": best_lag_ms,
                "delay_h1_minus_l1_ms": delay_hl_s * 1e3,
                "delay_tolerance_ms": delay_tol_s * 1e3,
                "abs_corr": abs_corr,
                "obs_ratio_p16": obs_p16,
                "obs_ratio_median": obs_med,
                "obs_ratio_p84": obs_p84,
                "tensor_ratio_lo": t_lo,
                "tensor_ratio_hi": t_hi,
                "scalar_ratio_lo": s_lo,
                "scalar_ratio_hi": s_hi,
                "tensor_interval_overlap": bool(tensor_overlap),
                "scalar_interval_overlap": bool(scalar_overlap),
                "tensor_min_rel_mismatch": t_mis,
                "scalar_min_rel_mismatch": s_mis,
                "ring_directions_used": int(ring_dirs.shape[0]),
            }
        )

    usable_rows = [r for r in rows if str(r.get("quality")) == "usable" and str(r.get("status", "")).startswith(("reject_", "watch_", "pass_"))]
    has_tensor_reject = any(str(r.get("status")) == "reject_tensor_response" for r in usable_rows)
    all_scalar_disfavored = bool(usable_rows) and all(str(r.get("status")) == "pass_scalar_only_disfavored" for r in usable_rows)
    # 条件分岐: `not usable_rows` を満たす経路を評価する。
    if not usable_rows:
        overall_status = "inconclusive"
        overall_reason = "no_usable_events"
    # 条件分岐: 前段条件が不成立で、`has_tensor_reject` を追加評価する。
    elif has_tensor_reject:
        overall_status = "reject"
        overall_reason = "tensor_response_failed_for_some_events"
    # 条件分岐: 前段条件が不成立で、`all_scalar_disfavored` を追加評価する。
    elif all_scalar_disfavored:
        overall_status = "pass"
        overall_reason = "scalar_only_disfavored_in_all_usable_events"
    else:
        overall_status = "watch"
        overall_reason = "scalar_not_excluded_with_current_h1_l1_only_constraints"

    scalar_only_overlap_all = all(bool(r.get("scalar_interval_overlap")) for r in usable_rows) if usable_rows else False
    scalar_only_mode_global_upper_bound_proxy = 1.0 if scalar_only_overlap_all else 0.0

    outdir = Path(str(args.outdir))
    public_outdir = Path(str(args.public_outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"

    _write_csv(out_csv, rows)
    _plot(rows, out_png)

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.polarization.antenna_pattern_audit.v1",
        "phase": 8,
        "step": "8.7.19.2",
        "inputs": {
            "events": events,
            "corr_use_min": float(args.corr_use_min),
            "sky_samples": int(args.sky_samples),
            "psi_samples": int(args.psi_samples),
            "cosi_samples": int(args.cosi_samples),
            "response_floor_frac": float(args.response_floor_frac),
            "detectors": ["H1", "L1"],
            "detector_site_constants": _DETECTOR_SITES,
            "baseline_length_m": baseline_norm,
        },
        "summary": {
            "n_events_requested": int(len(events)),
            "n_rows": int(len(rows)),
            "n_usable_events": int(len(usable_rows)),
            "overall_status": overall_status,
            "overall_reason": overall_reason,
            "scalar_only_mode_global_upper_bound_proxy": scalar_only_mode_global_upper_bound_proxy,
            "notes": [
                "This audit uses H1/L1 amplitude-ratio and time-delay ring constraints.",
                "With two-detector data, scalar-only exclusion can remain weak; global upper bound proxy=1 means not excluded.",
            ],
        },
        "rows": rows,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    copied: List[str] = []
    for src in [out_json, out_csv, out_png]:
        dst = public_outdir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst).replace("\\", "/"))

    payload["outputs"]["public_copies"] = copied
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, public_outdir / out_json.name)

    try:
        worklog.append_event(
            {
                "event_type": "gw_polarization_antenna_pattern_audit",
                "argv": list(sys.argv),
                "outputs": {
                    "audit_json": out_json,
                    "audit_csv": out_csv,
                    "audit_png": out_png,
                },
                "metrics": {
                    "overall_status": overall_status,
                    "n_usable_events": int(len(usable_rows)),
                    "scalar_only_mode_global_upper_bound_proxy": scalar_only_mode_global_upper_bound_proxy,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
