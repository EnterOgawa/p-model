from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ENERGY_KEYS = ("e2plus", "e4plus", "e3minus")


# 関数: `_parse_float` の入出力契約と処理意図を定義する。
def _parse_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")

    return out if math.isfinite(out) else float("nan")


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_median_abs` の入出力契約と処理意図を定義する。

def _median_abs(values: list[float]) -> float:
    finite = [abs(float(v)) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return float(median(finite))


# 関数: `_rms` の入出力契約と処理意図を定義する。

def _rms(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return math.sqrt(sum(v * v for v in finite) / float(len(finite)))


# 関数: `_safe_median` の入出力契約と処理意図を定義する。

def _safe_median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return float(median(finite))


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        # 条件分岐: `not rows` を満たす経路を評価する。
        if not rows:
            f.write("")
            return

        headers = list(rows[0].keys())
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h) for h in headers])


# 関数: `_solve_linear_system` の入出力契約と処理意図を定義する。

def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    a = [row[:] for row in matrix]
    b = rhs[:]
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        # 条件分岐: `abs(a[pivot_row][i]) < 1.0e-15` を満たす経路を評価する。
        if abs(a[pivot_row][i]) < 1.0e-15:
            raise SystemExit("[fail] singular system in weighted linear fit")

        # 条件分岐: `pivot_row != i` を満たす経路を評価する。

        if pivot_row != i:
            a[i], a[pivot_row] = a[pivot_row], a[i]
            b[i], b[pivot_row] = b[pivot_row], b[i]

        pivot = a[i][i]
        inv_pivot = 1.0 / pivot
        for j in range(i, n):
            a[i][j] *= inv_pivot

        b[i] *= inv_pivot
        for r in range(n):
            # 条件分岐: `r == i` を満たす経路を評価する。
            if r == i:
                continue

            factor = a[r][i]
            # 条件分岐: `factor == 0.0` を満たす経路を評価する。
            if factor == 0.0:
                continue

            for c in range(i, n):
                a[r][c] -= factor * a[i][c]

            b[r] -= factor * b[i]

    return b


# 関数: `_weighted_linear_fit` の入出力契約と処理意図を定義する。

def _weighted_linear_fit(
    *,
    xs: list[list[float]],
    ys: list[float],
    ws: list[float],
    ridge: float = 1.0e-10,
) -> list[float]:
    # 条件分岐: `not xs or not ys or not ws` を満たす経路を評価する。
    if not xs or not ys or not ws:
        raise SystemExit("[fail] empty data in weighted linear fit")

    p = len(xs[0])
    xtwx = [[0.0 for _ in range(p)] for _ in range(p)]
    xtwy = [0.0 for _ in range(p)]
    for x, y, w in zip(xs, ys, ws):
        # 条件分岐: `not (math.isfinite(y) and math.isfinite(w) and w > 0.0)` を満たす経路を評価する。
        if not (math.isfinite(y) and math.isfinite(w) and w > 0.0):
            continue

        for i in range(p):
            xi = float(x[i])
            # 条件分岐: `not math.isfinite(xi)` を満たす経路を評価する。
            if not math.isfinite(xi):
                continue

            xtwy[i] += w * xi * y
            for j in range(p):
                xj = float(x[j])
                # 条件分岐: `not math.isfinite(xj)` を満たす経路を評価する。
                if not math.isfinite(xj):
                    continue

                xtwx[i][j] += w * xi * xj

    for i in range(p):
        xtwx[i][i] += ridge

    return _solve_linear_system(xtwx, xtwy)


# 関数: `_dot` の入出力契約と処理意図を定義する。

def _dot(x: list[float], beta: list[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(x, beta)))


# 関数: `_a_band` の入出力契約と処理意図を定義する。

def _a_band(a: int) -> str:
    # 条件分岐: `a <= 40` を満たす経路を評価する。
    if a <= 40:
        return "A_2_40"

    # 条件分岐: `a <= 100` を満たす経路を評価する。

    if a <= 100:
        return "A_41_100"

    return "A_101_plus"


# 関数: `_r42_class` の入出力契約と処理意図を定義する。

def _r42_class(value: float) -> str:
    # 条件分岐: `not math.isfinite(value)` を満たす経路を評価する。
    if not math.isfinite(value):
        return "unknown"

    # 条件分岐: `value < 2.0` を満たす経路を評価する。

    if value < 2.0:
        return "vibrational_like"

    # 条件分岐: `value < 2.8` を満たす経路を評価する。

    if value < 2.8:
        return "transitional"

    return "rotational_like"


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_json = root / "data" / "quantum" / "sources" / "nndc_nudat3_primary_secondary" / "extracted_spectroscopy.json"
    ame_csv = out_dir / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv"
    # 条件分岐: `not spec_json.exists()` を満たす経路を評価する。
    if not spec_json.exists():
        raise SystemExit(f"[fail] missing required input: {spec_json}")

    # 条件分岐: `not ame_csv.exists()` を満たす経路を評価する。

    if not ame_csv.exists():
        raise SystemExit(f"[fail] missing required input: {ame_csv}")

    ame_by_za: dict[tuple[int, int], dict[str, Any]] = {}
    with ame_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            a = int(row["A"])
            ame_by_za[(z, a)] = row

    payload = json.loads(spec_json.read_text(encoding="utf-8"))
    rows_in = payload.get("rows")
    # 条件分岐: `not isinstance(rows_in, list)` を満たす経路を評価する。
    if not isinstance(rows_in, list):
        raise SystemExit("[fail] invalid extracted_spectroscopy.json format")

    joined: list[dict[str, Any]] = []
    for item in rows_in:
        # 条件分岐: `not isinstance(item, dict)` を満たす経路を評価する。
        if not isinstance(item, dict):
            continue

        z = int(item["Z"])
        a = int(item["A"])
        n = int(item["N"])
        ame = ame_by_za.get((z, a))
        # 条件分岐: `ame is None` を満たす経路を評価する。
        if ame is None:
            continue

        i_asym = float((n - z) / a)
        log_a = float(math.log(float(a)))
        j_e = float(ame["J_E_MeV"])
        c_req = float(ame["C_required"])
        # 条件分岐: `not (math.isfinite(j_e) and j_e > 0.0 and math.isfinite(c_req) and c_req > 0.0)` を満たす経路を評価する。
        if not (math.isfinite(j_e) and j_e > 0.0 and math.isfinite(c_req) and c_req > 0.0):
            continue

        row: dict[str, Any] = {
            "Z": z,
            "N": n,
            "A": a,
            "name": str(item.get("name", "")),
            "symbol": str(ame.get("symbol", "")),
            "parity": str(ame.get("parity", "")),
            "is_magic_any": str(ame.get("is_magic_any", "False")).lower() == "true",
            "logA": log_a,
            "I_asym": i_asym,
            "J_E_MeV": j_e,
            "logJ": float(math.log(j_e)),
            "C_required": c_req,
            "logC": float(math.log(c_req)),
            "A_band": _a_band(a),
            "e2plus_obs_MeV": _parse_float(item.get("e2plus_keV")) / 1000.0,
            "e2plus_sigma_obs_MeV": _parse_float(item.get("e2plus_sigma_keV")) / 1000.0,
            "e4plus_obs_MeV": _parse_float(item.get("e4plus_keV")) / 1000.0,
            "e4plus_sigma_obs_MeV": _parse_float(item.get("e4plus_sigma_keV")) / 1000.0,
            "e3minus_obs_MeV": _parse_float(item.get("e3minus_keV")) / 1000.0,
            "e3minus_sigma_obs_MeV": _parse_float(item.get("e3minus_sigma_keV")) / 1000.0,
            "r42_obs": _parse_float(item.get("r42")),
            "r42_sigma_obs": _parse_float(item.get("r42_sigma")),
        }
        n_levels = 0
        for energy_key in ENERGY_KEYS:
            value = float(row[f"{energy_key}_obs_MeV"])
            # 条件分岐: `math.isfinite(value) and value > 0.0` を満たす経路を評価する。
            if math.isfinite(value) and value > 0.0:
                n_levels += 1

        row["n_observed_levels"] = n_levels
        joined.append(row)

    # 条件分岐: `len(joined) < 100` を満たす経路を評価する。

    if len(joined) < 100:
        raise SystemExit(f"[fail] insufficient spectroscopy-join rows: n={len(joined)}")

    features = [[1.0, float(r["logA"]), float(r["I_asym"]), 1.0 if bool(r["is_magic_any"]) else 0.0] for r in joined]

    # Fit energy models in log space.
    beta_by_energy: dict[str, list[float]] = {}
    fit_counts: dict[str, int] = {}
    for energy_key in ENERGY_KEYS:
        xs: list[list[float]] = []
        ys: list[float] = []
        ws: list[float] = []
        for feature, row in zip(features, joined):
            obs = float(row[f"{energy_key}_obs_MeV"])
            # 条件分岐: `not (math.isfinite(obs) and obs > 0.0)` を満たす経路を評価する。
            if not (math.isfinite(obs) and obs > 0.0):
                continue

            sigma = float(row.get(f"{energy_key}_sigma_obs_MeV", float("nan")))
            # 条件分岐: `math.isfinite(sigma) and sigma > 0.0` を満たす経路を評価する。
            if math.isfinite(sigma) and sigma > 0.0:
                weight = 1.0 / (sigma * sigma)
            else:
                weight = 1.0

            xs.append(feature)
            ys.append(float(math.log(obs)))
            ws.append(float(weight))

        # 条件分岐: `len(xs) < 20` を満たす経路を評価する。

        if len(xs) < 20:
            continue

        beta_by_energy[energy_key] = _weighted_linear_fit(xs=xs, ys=ys, ws=ws)
        fit_counts[energy_key] = len(xs)

    # Fit R4/2 proxy class model.

    r42_xs: list[list[float]] = []
    r42_ys: list[float] = []
    r42_ws: list[float] = []
    for row in joined:
        obs = float(row["r42_obs"])
        # 条件分岐: `not (math.isfinite(obs) and obs > 0.0)` を満たす経路を評価する。
        if not (math.isfinite(obs) and obs > 0.0):
            continue

        sigma = float(row.get("r42_sigma_obs", float("nan")))
        w = 1.0 / (sigma * sigma) if (math.isfinite(sigma) and sigma > 0.0) else 1.0
        r42_xs.append([1.0, float(row["logA"]), float(row["I_asym"]), 1.0 if bool(row["is_magic_any"]) else 0.0])
        r42_ys.append(obs)
        r42_ws.append(float(w))

    beta_r42 = _weighted_linear_fit(xs=r42_xs, ys=r42_ys, ws=r42_ws) if len(r42_xs) >= 20 else [float("nan")] * 4

    # Predict and summarize.
    summary_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    observables_for_summary: list[str] = []
    for energy_key in ENERGY_KEYS:
        # 条件分岐: `energy_key in beta_by_energy` を満たす経路を評価する。
        if energy_key in beta_by_energy:
            observables_for_summary.append(energy_key)

    # 条件分岐: `len(r42_xs) >= 20` を満たす経路を評価する。

    if len(r42_xs) >= 20:
        observables_for_summary.append("r42")

    # 関数: `_mad_sigma` の入出力契約と処理意図を定義する。

    def _mad_sigma(values: list[float]) -> float:
        finite = [float(v) for v in values if math.isfinite(float(v))]
        # 条件分岐: `len(finite) < 3` を満たす経路を評価する。
        if len(finite) < 3:
            return float("nan")

        center = _safe_median(finite)
        mad = _safe_median([abs(v - center) for v in finite])
        sigma = 1.4826 * mad
        return float(sigma) if (math.isfinite(sigma) and sigma > 0.0) else float("nan")

    for row, feature in zip(joined, features):
        out = dict(row)
        for energy_key in ENERGY_KEYS:
            beta = beta_by_energy.get(energy_key)
            obs = float(row[f"{energy_key}_obs_MeV"])
            # 条件分岐: `beta is None` を満たす経路を評価する。
            if beta is None:
                out[f"{energy_key}_pred_MeV"] = float("nan")
                out[f"{energy_key}_log10_ratio"] = float("nan")
                out[f"{energy_key}_z_obs"] = float("nan")
                continue

            pred = float(math.exp(_dot(feature, beta)))
            out[f"{energy_key}_pred_MeV"] = pred
            # 条件分岐: `math.isfinite(obs) and obs > 0.0` を満たす経路を評価する。
            if math.isfinite(obs) and obs > 0.0:
                out[f"{energy_key}_log10_ratio"] = float(math.log10(pred / obs))
                sigma = float(row.get(f"{energy_key}_sigma_obs_MeV", float("nan")))
                out[f"{energy_key}_z_obs"] = float((pred - obs) / sigma) if (math.isfinite(sigma) and sigma > 0.0) else float("nan")
            else:
                out[f"{energy_key}_log10_ratio"] = float("nan")
                out[f"{energy_key}_z_obs"] = float("nan")

        # 条件分岐: `len(r42_xs) >= 20` を満たす経路を評価する。

        if len(r42_xs) >= 20:
            r42_pred = _dot([1.0, float(row["logA"]), float(row["I_asym"]), 1.0 if bool(row["is_magic_any"]) else 0.0], beta_r42)
            # 条件分岐: `r42_pred < 0.2` を満たす経路を評価する。
            if r42_pred < 0.2:
                r42_pred = 0.2

            out["r42_pred"] = float(r42_pred)
            r42_obs = float(row["r42_obs"])
            r42_sigma = float(row["r42_sigma_obs"])
            out["r42_resid"] = float(r42_pred - r42_obs) if math.isfinite(r42_obs) else float("nan")
            out["r42_z_obs"] = float((r42_pred - r42_obs) / r42_sigma) if (math.isfinite(r42_obs) and math.isfinite(r42_sigma) and r42_sigma > 0.0) else float("nan")
            out["r42_class_obs"] = _r42_class(r42_obs)
            out["r42_class_pred"] = _r42_class(float(r42_pred))
            out["r42_class_match"] = str(out["r42_class_obs"]) == str(out["r42_class_pred"])
        else:
            out["r42_pred"] = float("nan")
            out["r42_resid"] = float("nan")
            out["r42_z_obs"] = float("nan")
            out["r42_class_obs"] = "unknown"
            out["r42_class_pred"] = "unknown"
            out["r42_class_match"] = False

        full_rows.append(out)

    # Robust z normalization for stable falsification-style summaries.

    for energy_key in ENERGY_KEYS:
        ratios = [float(r[f"{energy_key}_log10_ratio"]) for r in full_rows if math.isfinite(float(r[f"{energy_key}_log10_ratio"]))]
        center = _safe_median(ratios)
        sigma = _mad_sigma(ratios)
        for row in full_rows:
            val = float(row[f"{energy_key}_log10_ratio"])
            # 条件分岐: `math.isfinite(val) and math.isfinite(center) and math.isfinite(sigma) and sig...` を満たす経路を評価する。
            if math.isfinite(val) and math.isfinite(center) and math.isfinite(sigma) and sigma > 0.0:
                row[f"{energy_key}_z_robust"] = float((val - center) / sigma)
            else:
                row[f"{energy_key}_z_robust"] = float("nan")

    r42_resid_all = [float(r["r42_resid"]) for r in full_rows if math.isfinite(float(r["r42_resid"]))]
    r42_center = _safe_median(r42_resid_all)
    r42_sigma_robust = _mad_sigma(r42_resid_all)
    for row in full_rows:
        val = float(row["r42_resid"])
        # 条件分岐: `math.isfinite(val) and math.isfinite(r42_center) and math.isfinite(r42_sigma_...` を満たす経路を評価する。
        if math.isfinite(val) and math.isfinite(r42_center) and math.isfinite(r42_sigma_robust) and r42_sigma_robust > 0.0:
            row["r42_z_robust"] = float((val - r42_center) / r42_sigma_robust)
        else:
            row["r42_z_robust"] = float("nan")

    # 関数: `summarize_energy` の入出力契約と処理意図を定義する。

    def summarize_energy(key: str) -> dict[str, Any]:
        sub = [r for r in full_rows if math.isfinite(float(r[f"{key}_pred_MeV"])) and math.isfinite(float(r[f"{key}_obs_MeV"])) and float(r[f"{key}_obs_MeV"]) > 0.0]
        ratios = [float(r[f"{key}_log10_ratio"]) for r in sub if math.isfinite(float(r[f"{key}_log10_ratio"]))]
        zvals = [float(r[f"{key}_z_robust"]) for r in sub if math.isfinite(float(r[f"{key}_z_robust"]))]
        pass_z3 = sum(1 for z in zvals if abs(z) <= 3.0)
        return {
            "observable": key,
            "n": len(sub),
            "fit_n": int(fit_counts.get(key, 0)),
            "median_log10_ratio": _safe_median(ratios),
            "median_abs_log10_ratio": _median_abs(ratios),
            "rms_log10_ratio": _rms(ratios),
            "median_abs_z": _median_abs(zvals),
            "z_pass_fraction": float(pass_z3 / len(zvals)) if zvals else float("nan"),
        }

    for key in ENERGY_KEYS:
        # 条件分岐: `key in beta_by_energy` を満たす経路を評価する。
        if key in beta_by_energy:
            summary_rows.append(summarize_energy(key))

    # 条件分岐: `len(r42_xs) >= 20` を満たす経路を評価する。

    if len(r42_xs) >= 20:
        sub = [r for r in full_rows if math.isfinite(float(r["r42_pred"])) and math.isfinite(float(r["r42_obs"])) and float(r["r42_obs"]) > 0.0]
        resid = [float(r["r42_resid"]) for r in sub if math.isfinite(float(r["r42_resid"]))]
        zvals = [float(r["r42_z_robust"]) for r in sub if math.isfinite(float(r["r42_z_robust"]))]
        cls_n = sum(1 for r in sub if str(r["r42_class_obs"]) != "unknown")
        cls_hit = sum(1 for r in sub if bool(r["r42_class_match"]))
        summary_rows.append(
            {
                "observable": "r42",
                "n": len(sub),
                "fit_n": len(r42_xs),
                "median_resid": _safe_median(resid),
                "median_abs_resid": _median_abs(resid),
                "rms_resid": _rms(resid),
                "median_abs_z": _median_abs(zvals),
                "z_pass_fraction": float(sum(1 for z in zvals if abs(z) <= 3.0) / len(zvals)) if zvals else float("nan"),
                "class_match_fraction": float(cls_hit / cls_n) if cls_n > 0 else float("nan"),
            }
        )

    # Level density proxy summary.

    density_rows: list[dict[str, Any]] = []
    by_band: dict[str, list[int]] = defaultdict(list)
    by_magic: dict[str, list[int]] = defaultdict(list)
    for row in full_rows:
        n_levels = int(row["n_observed_levels"])
        by_band[str(row["A_band"])].append(n_levels)
        by_magic["magic_any" if bool(row["is_magic_any"]) else "nonmagic"].append(n_levels)

    for band in ("A_2_40", "A_41_100", "A_101_plus"):
        vals = by_band.get(band, [])
        density_rows.append(
            {
                "group_type": "A_band",
                "group": band,
                "n": len(vals),
                "median_n_observed_levels": _safe_median([float(v) for v in vals]),
                "rms_n_observed_levels": _rms([float(v) for v in vals]),
            }
        )

    for tag in ("magic_any", "nonmagic"):
        vals = by_magic.get(tag, [])
        density_rows.append(
            {
                "group_type": "magic",
                "group": tag,
                "n": len(vals),
                "median_n_observed_levels": _safe_median([float(v) for v in vals]),
                "rms_n_observed_levels": _rms([float(v) for v in vals]),
            }
        )

    # Representative nuclei table.

    representatives = {(8, 16), (20, 40), (28, 56), (50, 132), (82, 208)}
    rep_rows = [r for r in full_rows if (int(r["Z"]), int(r["A"])) in representatives]
    rep_rows = sorted(rep_rows, key=lambda r: (int(r["A"]), int(r["Z"])))

    out_full = out_dir / "nuclear_excitation_level_prediction_full.csv"
    out_summary = out_dir / "nuclear_excitation_level_prediction_summary.csv"
    out_density = out_dir / "nuclear_excitation_level_prediction_level_density_summary.csv"
    out_rep = out_dir / "nuclear_excitation_level_prediction_representative.csv"
    _write_csv(out_full, full_rows)
    _write_csv(out_summary, summary_rows)
    _write_csv(out_density, density_rows)
    _write_csv(out_rep, rep_rows)

    # Figure.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    e2_rows = [r for r in full_rows if math.isfinite(float(r["e2plus_obs_MeV"])) and math.isfinite(float(r["e2plus_pred_MeV"]))]
    # 条件分岐: `e2_rows` を満たす経路を評価する。
    if e2_rows:
        obs = [float(r["e2plus_obs_MeV"]) for r in e2_rows]
        pred = [float(r["e2plus_pred_MeV"]) for r in e2_rows]
        ax00.scatter(obs, pred, s=11.0, alpha=0.35, c=[int(r["A"]) for r in e2_rows], cmap="viridis")
        lo = min(min(obs), min(pred))
        hi = max(max(obs), max(pred))
        ax00.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.1)

    ax00.set_xlabel("E2+ observed [MeV]")
    ax00.set_ylabel("E2+ predicted [MeV]")
    ax00.set_title("Low-lying E2+ prediction (ENSDF-derived)")
    ax00.grid(True, ls=":", lw=0.6, alpha=0.6)

    e4_rows = [r for r in full_rows if math.isfinite(float(r["e4plus_obs_MeV"])) and math.isfinite(float(r["e4plus_pred_MeV"]))]
    e3_rows = [r for r in full_rows if math.isfinite(float(r["e3minus_obs_MeV"])) and math.isfinite(float(r["e3minus_pred_MeV"]))]
    # 条件分岐: `e4_rows` を満たす経路を評価する。
    if e4_rows:
        ax01.scatter([float(r["e4plus_obs_MeV"]) for r in e4_rows], [float(r["e4plus_pred_MeV"]) for r in e4_rows], s=11.0, alpha=0.35, label="E4+")

    # 条件分岐: `e3_rows` を満たす経路を評価する。

    if e3_rows:
        ax01.scatter([float(r["e3minus_obs_MeV"]) for r in e3_rows], [float(r["e3minus_pred_MeV"]) for r in e3_rows], s=11.0, alpha=0.35, label="E3-")

    combined_obs = [float(r["e4plus_obs_MeV"]) for r in e4_rows] + [float(r["e3minus_obs_MeV"]) for r in e3_rows]
    combined_pred = [float(r["e4plus_pred_MeV"]) for r in e4_rows] + [float(r["e3minus_pred_MeV"]) for r in e3_rows]
    # 条件分岐: `combined_obs` を満たす経路を評価する。
    if combined_obs:
        lo = min(min(combined_obs), min(combined_pred))
        hi = max(max(combined_obs), max(combined_pred))
        ax01.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.1)

    ax01.set_xlabel("Excitation observed [MeV]")
    ax01.set_ylabel("Excitation predicted [MeV]")
    ax01.set_title("E4+ and E3- prediction")
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax01.legend(loc="best", fontsize=8)

    level_rows = [r for r in full_rows if int(r["n_observed_levels"]) >= 0]
    ax10.scatter([int(r["A"]) for r in level_rows], [int(r["n_observed_levels"]) for r in level_rows], s=10.0, alpha=0.35, c="#4c78a8")
    ax10.set_xlabel("A")
    ax10.set_ylabel("n observed low-lying levels")
    ax10.set_title("Level-density proxy from NuDat fields")
    ax10.grid(True, ls=":", lw=0.6, alpha=0.6)

    r42_rows = [r for r in full_rows if math.isfinite(float(r["r42_obs"])) and math.isfinite(float(r["r42_pred"]))]
    # 条件分岐: `r42_rows` を満たす経路を評価する。
    if r42_rows:
        ax11.scatter([float(r["r42_obs"]) for r in r42_rows], [float(r["r42_pred"]) for r in r42_rows], s=10.0, alpha=0.35, c="#f58518")
        lo = min(min(float(r["r42_obs"]) for r in r42_rows), min(float(r["r42_pred"]) for r in r42_rows))
        hi = max(max(float(r["r42_obs"]) for r in r42_rows), max(float(r["r42_pred"]) for r in r42_rows))
        ax11.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.1)

    ax11.set_xlabel("R4/2 observed")
    ax11.set_ylabel("R4/2 predicted")
    ax11.set_title("Collectivity class proxy (R4/2)")
    ax11.grid(True, ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.16.6: low-lying excitation prediction (ENSDF-derived NuDat)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png = out_dir / "nuclear_excitation_level_prediction.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.16.6",
        "inputs": {
            "spectroscopy_json": {"path": str(spec_json), "sha256": _sha256(spec_json)},
            "ame_all_nuclei_csv": {"path": str(ame_csv), "sha256": _sha256(ame_csv)},
        },
        "counts": {
            "n_spectroscopy_rows_input": len(rows_in),
            "n_joined_rows": len(full_rows),
            "n_fit_e2plus": int(fit_counts.get("e2plus", 0)),
            "n_fit_e4plus": int(fit_counts.get("e4plus", 0)),
            "n_fit_e3minus": int(fit_counts.get("e3minus", 0)),
            "n_fit_r42": len(r42_xs),
        },
        "frozen_models": {
            "energy_log_linear_features": ["1", "logA", "I_asym", "is_magic_any"],
            "r42_linear_features": ["1", "logA", "I_asym", "is_magic_any"],
            "beta_e2plus": beta_by_energy.get("e2plus"),
            "beta_e4plus": beta_by_energy.get("e4plus"),
            "beta_e3minus": beta_by_energy.get("e3minus"),
            "beta_r42": beta_r42,
        },
        "summary": summary_rows,
        "level_density_summary": density_rows,
        "notes": [
            "ENSDF-derived NuDat fields include low-lying excitation energies and R4/2; this Step uses them as independent cross-check observables.",
            "This initial Step freezes an operational prediction interface; explicit Jπ level assignment is left as a follow-up extension.",
            "Class proxy uses R4/2 thresholds: <2.0 vibrational-like, 2.0-2.8 transitional, >=2.8 rotational-like.",
        ],
        "outputs": {
            "full_csv": str(out_full),
            "summary_csv": str(out_summary),
            "level_density_summary_csv": str(out_density),
            "representative_csv": str(out_rep),
            "figure_png": str(out_png),
        },
    }
    out_json = out_dir / "nuclear_excitation_level_prediction_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] step=7.16.6 rows={len(full_rows)} metrics={out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
