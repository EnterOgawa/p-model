from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


def _parse_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")

    return out if math.isfinite(out) else float("nan")


def _safe_median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return float(median(finite))


def _rms(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    # 条件分岐: `not finite` を満たす経路を評価する。
    if not finite:
        return float("nan")

    return math.sqrt(sum(v * v for v in finite) / float(len(finite)))


def _pearson(xs: list[float], ys: list[float]) -> float:
    paired = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    # 条件分岐: `len(paired) < 3` を満たす経路を評価する。
    if len(paired) < 3:
        return float("nan")

    xvals = [p[0] for p in paired]
    yvals = [p[1] for p in paired]
    mx = sum(xvals) / float(len(xvals))
    my = sum(yvals) / float(len(yvals))
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    # 条件分岐: `vx <= 0.0 or vy <= 0.0` を満たす経路を評価する。
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")

    cov = sum((x - mx) * (y - my) for x, y in paired)
    return float(cov / math.sqrt(vx * vy))


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


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


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


def _dot(x: list[float], beta: list[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(x, beta)))


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


def _read_beta2(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        raise SystemExit(f"[fail] invalid beta2 source format: {path}")

    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        z = int(row.get("Z", -1))
        n = int(row.get("N", -1))
        a = int(row.get("A", -1))
        b2 = _parse_float(row.get("beta2"))
        b2_sig = _parse_float(row.get("beta2_sigma"))
        # 条件分岐: `z < 1 or n < 0 or a < 2 or not math.isfinite(b2)` を満たす経路を評価する。
        if z < 1 or n < 0 or a < 2 or not math.isfinite(b2):
            continue

        out[(z, n)] = {
            "A": a,
            "beta2_obs": b2,
            "beta2_sigma": b2_sig,
            "reference": str(row.get("reference", "")),
            "adopted_entry_type": str(row.get("adoptedEntryType", "")),
        }

    return out


def _read_pairing(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            n = int(row["N"])
            out[(z, n)] = {
                "A": int(row["A"]),
                "symbol": str(row.get("symbol", "")),
                "parity": str(row.get("parity", "")),
                "is_magic_any": str(row.get("is_magic_any", "False")).strip().lower() in {"true", "1", "yes"},
                "B_obs_MeV": _parse_float(row.get("B_obs_MeV")),
                "resid_after_MeV": _parse_float(row.get("resid_after_MeV")),
                "abs_resid_after_MeV": _parse_float(row.get("abs_resid_after_MeV")),
            }

    return out


def _read_all_nuclei(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            n = int(row["N"])
            out[(z, n)] = {
                "A": int(row["A"]),
                "J_E_MeV": _parse_float(row.get("J_E_MeV")),
                "C_required": _parse_float(row.get("C_required")),
                "log10_ratio_collective": _parse_float(row.get("log10_ratio_collective")),
            }

    return out


def _read_excitation(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["Z"])
            n = int(row["N"])
            out[(z, n)] = {
                "r42_obs": _parse_float(row.get("r42_obs")),
                "r42_sigma_obs": _parse_float(row.get("r42_sigma_obs")),
            }

    return out


def _build_figure(*, rows: list[dict[str, Any]], out_png: Path, r42_fit_coeffs: list[float]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    x_beta2 = [float(r["beta2_obs"]) for r in rows]
    y_beta2 = [float(r["beta2_pred"]) for r in rows]
    colors = ["#4c78a8" if bool(r["is_magic_any"]) else "#72b7b2" for r in rows]
    ax00.scatter(x_beta2, y_beta2, s=10.0, alpha=0.35, c=colors)
    # 条件分岐: `x_beta2` を満たす経路を評価する。
    if x_beta2:
        lo = min(x_beta2)
        hi = max(x_beta2)
        ax00.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#444444")

    ax00.set_xlabel("beta2_obs")
    ax00.set_ylabel("beta2_pred")
    ax00.set_title("beta2 prediction vs observed (color=magic flag)")
    ax00.grid(True, ls=":", lw=0.6, alpha=0.6)

    x_beta4 = [float(r["beta4_proxy_obs"]) for r in rows]
    y_beta4 = [float(r["beta4_proxy_pred"]) for r in rows]
    ax01.scatter(x_beta4, y_beta4, s=10.0, alpha=0.35, color="#f58518")
    # 条件分岐: `x_beta4` を満たす経路を評価する。
    if x_beta4:
        lo = min(x_beta4)
        hi = max(x_beta4)
        ax01.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#444444")

    ax01.set_xlabel("beta4_proxy_obs")
    ax01.set_ylabel("beta4_proxy_pred")
    ax01.set_title("beta4 proxy prediction (beta4 ~= beta2^2 / 3)")
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)

    r42_rows = [r for r in rows if math.isfinite(float(r["r42_obs"])) and math.isfinite(float(r["r42_pred_from_beta2"]))]
    ax10.scatter(
        [float(r["beta2_obs"]) for r in r42_rows],
        [float(r["r42_obs"]) for r in r42_rows],
        s=10.0,
        alpha=0.35,
        color="#54a24b",
        label="r42_obs vs beta2_obs",
    )
    # 条件分岐: `len(r42_fit_coeffs) == 2` を満たす経路を評価する。
    if len(r42_fit_coeffs) == 2:
        b0, b1 = float(r42_fit_coeffs[0]), float(r42_fit_coeffs[1])
        xs = sorted([float(r["beta2_obs"]) for r in r42_rows if math.isfinite(float(r["beta2_obs"]))])
        # 条件分岐: `xs` を満たす経路を評価する。
        if xs:
            ys = [b0 + b1 * x for x in xs]
            ax10.plot(xs, ys, color="#e45756", lw=1.3, label="r42 fit(beta2)")

    ax10.set_xlabel("beta2")
    ax10.set_ylabel("R4/2")
    ax10.set_title("Rotation-band proxy (R4/2) vs deformation")
    ax10.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax10.legend(loc="best", fontsize=8)

    parity_labels = ["ee", "eo", "oe", "oo"]
    medians = []
    for label in parity_labels:
        vals = [
            abs(float(r["beta2_resid"]))
            for r in rows
            if str(r["parity"]) == label and math.isfinite(float(r["beta2_resid"]))
        ]
        medians.append(_safe_median(vals))

    ax11.bar(parity_labels, medians, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax11.set_ylabel("median abs(beta2_pred-beta2_obs)")
    ax11.set_title("Residual by parity class")
    ax11.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.16.17: nuclear deformation parameter prediction audit", y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_beta2_json = root / "data" / "quantum" / "sources" / "nndc_be2_adopted_entries" / "extracted_beta2.json"
    in_pairing_csv = out_dir / "nuclear_pairing_effect_systematics_per_nucleus.csv"
    in_all_csv = out_dir / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv"
    in_excitation_csv = out_dir / "nuclear_excitation_level_prediction_full.csv"

    for p in (in_beta2_json, in_pairing_csv, in_all_csv, in_excitation_csv):
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise SystemExit(f"[fail] missing required input: {p}")

    beta2_by_zn = _read_beta2(in_beta2_json)
    pairing_by_zn = _read_pairing(in_pairing_csv)
    all_by_zn = _read_all_nuclei(in_all_csv)
    excitation_by_zn = _read_excitation(in_excitation_csv)

    rows: list[dict[str, Any]] = []
    features: list[list[float]] = []
    y_beta2: list[float] = []
    w_beta2: list[float] = []

    for (z, n), b2 in sorted(beta2_by_zn.items()):
        pairing = pairing_by_zn.get((z, n))
        alln = all_by_zn.get((z, n))
        # 条件分岐: `pairing is None or alln is None` を満たす経路を評価する。
        if pairing is None or alln is None:
            continue

        a = int(pairing["A"])
        beta2_obs = float(b2["beta2_obs"])
        beta2_sig = float(b2["beta2_sigma"])
        i_asym = float((n - z) / a)
        c_req = float(alln["C_required"])
        c_req_norm = float(c_req / max(a - 1, 1)) if math.isfinite(c_req) else float("nan")
        abs_be_resid = float(pairing["abs_resid_after_MeV"])
        b_obs = float(pairing["B_obs_MeV"])
        rel_be_resid = float(abs_be_resid / b_obs) if (math.isfinite(abs_be_resid) and math.isfinite(b_obs) and b_obs > 0.0) else float("nan")
        feat = [
            1.0,
            float(math.log(a)),
            i_asym,
            1.0 if bool(pairing["is_magic_any"]) else 0.0,
            float(math.log1p(c_req_norm)) if (math.isfinite(c_req_norm) and c_req_norm > -1.0) else 0.0,
            rel_be_resid if math.isfinite(rel_be_resid) else 0.0,
        ]
        sigma = beta2_sig if (math.isfinite(beta2_sig) and beta2_sig > 0.0) else 0.05
        weight = 1.0 / (sigma * sigma)

        features.append(feat)
        y_beta2.append(beta2_obs)
        w_beta2.append(weight)
        rows.append(
            {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": str(pairing["symbol"]),
                "parity": str(pairing["parity"]),
                "is_magic_any": bool(pairing["is_magic_any"]),
                "beta2_obs": beta2_obs,
                "beta2_sigma": beta2_sig,
                "reference": str(b2["reference"]),
                "adopted_entry_type": str(b2["adopted_entry_type"]),
                "B_obs_MeV": b_obs,
                "be_resid_after_MeV": float(pairing["resid_after_MeV"]),
                "abs_be_resid_after_MeV": abs_be_resid,
                "C_required": c_req,
                "C_required_norm": c_req_norm,
                "I_asym": i_asym,
                "r42_obs": float(excitation_by_zn.get((z, n), {}).get("r42_obs", float("nan"))),
                "r42_sigma_obs": float(excitation_by_zn.get((z, n), {}).get("r42_sigma_obs", float("nan"))),
            }
        )

    # 条件分岐: `len(rows) < 100` を満たす経路を評価する。

    if len(rows) < 100:
        raise SystemExit(f"[fail] insufficient joined beta2 rows: n={len(rows)}")

    beta2_coeffs = _weighted_linear_fit(xs=features, ys=y_beta2, ws=w_beta2)

    # R4/2 regression from observed beta2 to rotational proxy.
    r42_xs: list[list[float]] = []
    r42_ys: list[float] = []
    r42_ws: list[float] = []
    for row in rows:
        r42 = float(row["r42_obs"])
        b2 = float(row["beta2_obs"])
        # 条件分岐: `not (math.isfinite(r42) and math.isfinite(b2))` を満たす経路を評価する。
        if not (math.isfinite(r42) and math.isfinite(b2)):
            continue

        sigma = float(row["r42_sigma_obs"])
        weight = 1.0 / (sigma * sigma) if (math.isfinite(sigma) and sigma > 0.0) else 1.0
        r42_xs.append([1.0, b2])
        r42_ys.append(r42)
        r42_ws.append(weight)

    r42_coeffs = _weighted_linear_fit(xs=r42_xs, ys=r42_ys, ws=r42_ws) if len(r42_xs) >= 20 else [float("nan"), float("nan")]

    for row, feat in zip(rows, features):
        beta2_pred = float(_dot(feat, beta2_coeffs))
        row["beta2_pred"] = beta2_pred
        row["beta2_resid"] = float(beta2_pred - float(row["beta2_obs"]))
        sigma = float(row["beta2_sigma"])
        row["beta2_z_obs"] = float(row["beta2_resid"] / sigma) if (math.isfinite(sigma) and sigma > 0.0) else float("nan")

        # beta4 data are not directly available in fixed sources; use deterministic proxy from beta2.
        row["beta4_proxy_obs"] = float((float(row["beta2_obs"]) ** 2) / 3.0)
        row["beta4_proxy_pred"] = float((beta2_pred ** 2) / 3.0)
        row["beta4_proxy_resid"] = float(row["beta4_proxy_pred"] - row["beta4_proxy_obs"])

        # 条件分岐: `len(r42_coeffs) == 2 and math.isfinite(float(r42_coeffs[0])) and math.isfinit...` を満たす経路を評価する。
        if len(r42_coeffs) == 2 and math.isfinite(float(r42_coeffs[0])) and math.isfinite(float(r42_coeffs[1])):
            r42_pred = float(_dot([1.0, beta2_pred], r42_coeffs))
            # 条件分岐: `r42_pred < 0.2` を満たす経路を評価する。
            if r42_pred < 0.2:
                r42_pred = 0.2

            row["r42_pred_from_beta2"] = r42_pred
            r42_obs = float(row["r42_obs"])
            row["r42_resid_from_beta2"] = float(r42_pred - r42_obs) if math.isfinite(r42_obs) else float("nan")
            row["r42_class_obs"] = _r42_class(r42_obs)
            row["r42_class_pred"] = _r42_class(r42_pred)
            row["r42_class_match"] = bool(str(row["r42_class_obs"]) == str(row["r42_class_pred"]))
        else:
            row["r42_pred_from_beta2"] = float("nan")
            row["r42_resid_from_beta2"] = float("nan")
            row["r42_class_obs"] = "unknown"
            row["r42_class_pred"] = "unknown"
            row["r42_class_match"] = False

    rows = sorted(rows, key=lambda r: (int(r["Z"]), int(r["N"])))

    beta2_resids = [float(r["beta2_resid"]) for r in rows]
    beta2_z = [float(r["beta2_z_obs"]) for r in rows if math.isfinite(float(r["beta2_z_obs"]))]
    beta4_resids = [float(r["beta4_proxy_resid"]) for r in rows]
    r42_resids = [float(r["r42_resid_from_beta2"]) for r in rows if math.isfinite(float(r["r42_resid_from_beta2"]))]

    r42_rows = [r for r in rows if math.isfinite(float(r["r42_obs"])) and math.isfinite(float(r["r42_pred_from_beta2"]))]
    r42_match_rate = (
        float(sum(1 for r in r42_rows if bool(r["r42_class_match"])) / len(r42_rows))
        if r42_rows
        else float("nan")
    )

    summary_rows: list[dict[str, Any]] = []
    summary_rows.append(
        {
            "group_type": "overall",
            "group": "beta2",
            "n_rows": len(rows),
            "median_abs_resid": _safe_median([abs(v) for v in beta2_resids]),
            "rms_resid": _rms(beta2_resids),
            "pearson_obs_pred": _pearson(
                [float(r["beta2_obs"]) for r in rows],
                [float(r["beta2_pred"]) for r in rows],
            ),
            "median_abs_z": _safe_median([abs(v) for v in beta2_z]),
            "n_abs_z_gt3": sum(1 for v in beta2_z if abs(v) > 3.0),
        }
    )
    summary_rows.append(
        {
            "group_type": "overall",
            "group": "beta4_proxy",
            "n_rows": len(rows),
            "median_abs_resid": _safe_median([abs(v) for v in beta4_resids]),
            "rms_resid": _rms(beta4_resids),
            "pearson_obs_pred": _pearson(
                [float(r["beta4_proxy_obs"]) for r in rows],
                [float(r["beta4_proxy_pred"]) for r in rows],
            ),
            "median_abs_z": float("nan"),
            "n_abs_z_gt3": float("nan"),
        }
    )
    summary_rows.append(
        {
            "group_type": "overall",
            "group": "r42_from_beta2",
            "n_rows": len(r42_rows),
            "median_abs_resid": _safe_median([abs(v) for v in r42_resids]),
            "rms_resid": _rms(r42_resids),
            "pearson_obs_pred": _pearson(
                [float(r["r42_obs"]) for r in r42_rows],
                [float(r["r42_pred_from_beta2"]) for r in r42_rows],
            ),
            "median_abs_z": float("nan"),
            "n_abs_z_gt3": float("nan"),
            "class_match_rate": r42_match_rate,
        }
    )

    for parity in ("ee", "eo", "oe", "oo"):
        sub = [r for r in rows if str(r["parity"]) == parity]
        # 条件分岐: `not sub` を満たす経路を評価する。
        if not sub:
            continue

        summary_rows.append(
            {
                "group_type": "parity",
                "group": parity,
                "n_rows": len(sub),
                "median_beta2_obs": _safe_median([float(r["beta2_obs"]) for r in sub]),
                "median_beta2_pred": _safe_median([float(r["beta2_pred"]) for r in sub]),
                "median_abs_beta2_resid": _safe_median([abs(float(r["beta2_resid"])) for r in sub]),
                "median_abs_be_resid_MeV": _safe_median([abs(float(r["be_resid_after_MeV"])) for r in sub]),
            }
        )

    for flag in (True, False):
        sub = [r for r in rows if bool(r["is_magic_any"]) == flag]
        # 条件分岐: `not sub` を満たす経路を評価する。
        if not sub:
            continue

        summary_rows.append(
            {
                "group_type": "magic_flag",
                "group": "magic_any" if flag else "nonmagic",
                "n_rows": len(sub),
                "median_beta2_obs": _safe_median([float(r["beta2_obs"]) for r in sub]),
                "median_beta2_pred": _safe_median([float(r["beta2_pred"]) for r in sub]),
                "median_abs_beta2_resid": _safe_median([abs(float(r["beta2_resid"])) for r in sub]),
                "median_abs_be_resid_MeV": _safe_median([abs(float(r["be_resid_after_MeV"])) for r in sub]),
            }
        )

    summary_rows.append(
        {
            "group_type": "correlation",
            "group": "beta2_vs_be_residual",
            "n_rows": len(rows),
            "pearson_beta2_obs_vs_abs_be_resid": _pearson(
                [float(r["beta2_obs"]) for r in rows],
                [abs(float(r["be_resid_after_MeV"])) for r in rows],
            ),
            "pearson_beta2_obs_vs_crequired_norm": _pearson(
                [float(r["beta2_obs"]) for r in rows],
                [float(r["C_required_norm"]) for r in rows],
            ),
            "pearson_beta2_obs_vs_log10_ratio_collective": _pearson(
                [float(r["beta2_obs"]) for r in rows],
                [float(all_by_zn[(int(r["Z"]), int(r["N"]))]["log10_ratio_collective"]) for r in rows],
            ),
        }
    )

    representative_rows: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda r: abs(float(r["beta2_resid"])))
    best = ordered[:15]
    worst = ordered[-15:]
    for idx, row in enumerate(best, start=1):
        representative_rows.append(
            {
                "rank_type": "best",
                "rank": idx,
                "Z": row["Z"],
                "N": row["N"],
                "A": row["A"],
                "symbol": row["symbol"],
                "parity": row["parity"],
                "beta2_obs": row["beta2_obs"],
                "beta2_pred": row["beta2_pred"],
                "beta2_resid": row["beta2_resid"],
                "beta2_sigma": row["beta2_sigma"],
                "beta2_z_obs": row["beta2_z_obs"],
                "r42_obs": row["r42_obs"],
                "r42_pred_from_beta2": row["r42_pred_from_beta2"],
                "r42_class_match": row["r42_class_match"],
            }
        )

    for idx, row in enumerate(reversed(worst), start=1):
        representative_rows.append(
            {
                "rank_type": "worst",
                "rank": idx,
                "Z": row["Z"],
                "N": row["N"],
                "A": row["A"],
                "symbol": row["symbol"],
                "parity": row["parity"],
                "beta2_obs": row["beta2_obs"],
                "beta2_pred": row["beta2_pred"],
                "beta2_resid": row["beta2_resid"],
                "beta2_sigma": row["beta2_sigma"],
                "beta2_z_obs": row["beta2_z_obs"],
                "r42_obs": row["r42_obs"],
                "r42_pred_from_beta2": row["r42_pred_from_beta2"],
                "r42_class_match": row["r42_class_match"],
            }
        )

    out_full_csv = out_dir / "nuclear_deformation_parameter_prediction_full.csv"
    out_summary_csv = out_dir / "nuclear_deformation_parameter_prediction_summary.csv"
    out_representative_csv = out_dir / "nuclear_deformation_parameter_prediction_representative.csv"
    out_png = out_dir / "nuclear_deformation_parameter_prediction_quantification.png"
    out_json = out_dir / "nuclear_deformation_parameter_prediction_metrics.json"

    _write_csv(out_full_csv, rows)
    _write_csv(out_summary_csv, summary_rows)
    _write_csv(out_representative_csv, representative_rows)
    _build_figure(rows=rows, out_png=out_png, r42_fit_coeffs=r42_coeffs)

    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.16.17",
                "inputs": {
                    "beta2_source_json": {"path": str(in_beta2_json), "sha256": _sha256(in_beta2_json)},
                    "pairing_per_nucleus_csv": {"path": str(in_pairing_csv), "sha256": _sha256(in_pairing_csv)},
                    "all_nuclei_csv": {"path": str(in_all_csv), "sha256": _sha256(in_all_csv)},
                    "excitation_full_csv": {"path": str(in_excitation_csv), "sha256": _sha256(in_excitation_csv)},
                },
                "counts": {
                    "n_rows_joined": len(rows),
                    "n_rows_with_r42": len(r42_rows),
                    "n_representative_rows": len(representative_rows),
                },
                "fit_coeffs": {
                    "beta2_model": beta2_coeffs,
                    "r42_from_beta2": r42_coeffs,
                },
                "summary": summary_rows,
                "outputs": {
                    "full_csv": str(out_full_csv),
                    "summary_csv": str(out_summary_csv),
                    "representative_csv": str(out_representative_csv),
                    "figure_png": str(out_png),
                },
                "notes": [
                    "beta2 observed values are taken from NNDC B(E2) adopted entries (extracted_beta2.json).",
                    "beta4 direct observations are not included in the fixed source set; beta4_proxy ~= beta2^2/3 is used as an operational placeholder channel.",
                    "Rotation-band proxy is evaluated through R4/2 from Step 7.16.6 with r42(beta2) cross-check.",
                    "Deformation-B.E. consistency is tracked with correlation against |B_pred_after-B_obs| and C_required_norm.",
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


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
