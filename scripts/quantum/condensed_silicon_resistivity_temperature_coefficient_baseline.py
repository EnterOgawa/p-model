from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rho_at(table: list[dict[str, Any]], t_c: float) -> Optional[float]:
    for r in table:
        # 条件分岐: `abs(float(r.get("T_C", -1.0)) - float(t_c)) < 1e-6` を満たす経路を評価する。
        if abs(float(r.get("T_C", -1.0)) - float(t_c)) < 1e-6:
            v = r.get("rho_ohm_cm")
            # 条件分岐: `isinstance(v, (int, float)) and float(v) > 0` を満たす経路を評価する。
            if isinstance(v, (int, float)) and float(v) > 0:
                return float(v)

    return None


def _coeff_ln_rho_per_k(*, rho20: float, rho30: float) -> float:
    # Approximate d ln(rho)/dT around ~25°C using a 20–30°C finite difference.
    return (math.log(rho30) - math.log(rho20)) / 10.0


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_nbsir74_496_silicon_resistivity"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_silicon_resistivity_nbsir74_496_sources.py"
        )

    extracted = _read_json(extracted_path)
    samples = extracted.get("samples")
    # 条件分岐: `not isinstance(samples, list) or not samples` を満たす経路を評価する。
    if not isinstance(samples, list) or not samples:
        raise SystemExit(f"[fail] samples missing/empty: {extracted_path}")

    rows: list[dict[str, Any]] = []
    for s in samples:
        # 条件分岐: `not isinstance(s, dict)` を満たす経路を評価する。
        if not isinstance(s, dict):
            continue

        sid = str(s.get("sample_id") or "")
        stype = s.get("type")
        doping = s.get("doping")
        table = s.get("rho_range_table")
        # 条件分岐: `not sid or not isinstance(table, list) or not table` を満たす経路を評価する。
        if not sid or not isinstance(table, list) or not table:
            continue

        # 条件分岐: `stype not in ("p", "n")` を満たす経路を評価する。

        if stype not in ("p", "n"):
            continue

        rho20 = _rho_at(table, 20.0)
        rho30 = _rho_at(table, 30.0)
        rho23 = _rho_at(table, 23.0)
        # 条件分岐: `rho20 is None or rho30 is None or rho23 is None` を満たす経路を評価する。
        if rho20 is None or rho30 is None or rho23 is None:
            continue

        coeff = _coeff_ln_rho_per_k(rho20=rho20, rho30=rho30)
        rows.append(
            {
                "sample_id": sid,
                "type": stype,
                "doping": "" if doping is None else str(doping),
                "rho_23C_ohm_cm": float(rho23),
                "dlnrho_dT_23C_per_K_proxy": float(coeff),
                "pct_per_K_proxy": float(100.0 * coeff),
                "rho_20C_ohm_cm": float(rho20),
                "rho_30C_ohm_cm": float(rho30),
            }
        )

    # 条件分岐: `not rows` を満たす経路を評価する。

    if not rows:
        raise SystemExit("[fail] no usable samples with {20,23,30}°C mean rho extracted")

    # CSV

    out_csv = out_dir / "condensed_silicon_resistivity_temperature_coefficient_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "type",
                "doping",
                "rho_23C_ohm_cm",
                "dlnrho_dT_23C_per_K_proxy",
                "pct_per_K_proxy",
                "rho_20C_ohm_cm",
                "rho_30C_ohm_cm",
            ],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: float(x["rho_23C_ohm_cm"])):
            w.writerow(r)

    # Plot

    plt.figure(figsize=(8.5, 4.8))

    def pick_style(t: str, d: str) -> tuple[str, str]:
        # 条件分岐: `t == "p" and d == "Al"` を満たす経路を評価する。
        if t == "p" and d == "Al":
            return ("#d62728", "p-type (Al)")

        # 条件分岐: `t == "p"` を満たす経路を評価する。

        if t == "p":
            return ("#ff7f0e", "p-type (B)")

        return ("#1f77b4", "n-type")

    series: dict[str, dict[str, Any]] = {}
    for r in rows:
        color, label = pick_style(str(r["type"]), str(r.get("doping") or ""))
        # 条件分岐: `label not in series` を満たす経路を評価する。
        if label not in series:
            series[label] = {"color": color, "xs": [], "ys": []}

        series[label]["xs"].append(float(r["rho_23C_ohm_cm"]))
        series[label]["ys"].append(float(r["pct_per_K_proxy"]))

    for label, s in series.items():
        plt.scatter(s["xs"], s["ys"], s=20, alpha=0.85, label=label, color=s["color"])

    plt.xscale("log")
    plt.xlabel("Resistivity ρ(23°C) (Ω·cm)")
    plt.ylabel("d ln ρ / dT @ 23°C ( % / K ) (proxy: 20–30°C)")
    plt.title("Silicon resistivity temperature coefficient near room temperature (NBS IR 74-496; Appendix E)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / "condensed_silicon_resistivity_temperature_coefficient_baseline.png"
    plt.savefig(out_png, dpi=180)
    plt.close()

    # Summary
    by_type: dict[str, int] = {}
    ranges_by_type: dict[str, dict[str, float]] = {}
    for r in rows:
        key = f"{r['type']}:{r.get('doping') or ''}"
        by_type[key] = by_type.get(key, 0) + 1
        v = float(r["pct_per_K_proxy"])
        # 条件分岐: `key not in ranges_by_type` を満たす経路を評価する。
        if key not in ranges_by_type:
            ranges_by_type[key] = {"min": v, "max": v}
        else:
            ranges_by_type[key]["min"] = float(min(ranges_by_type[key]["min"], v))
            ranges_by_type[key]["max"] = float(max(ranges_by_type[key]["max"], v))

    # Correlation: log10(rho_23C) vs pct_per_K_proxy (purely descriptive).

    xs = [math.log10(float(r["rho_23C_ohm_cm"])) for r in rows if float(r["rho_23C_ohm_cm"]) > 0]
    ys = [float(r["pct_per_K_proxy"]) for r in rows if float(r["rho_23C_ohm_cm"]) > 0]
    corr_log10rho_vs_pct = None
    # 条件分岐: `len(xs) == len(ys) and len(xs) >= 3` を満たす経路を評価する。
    if len(xs) == len(ys) and len(xs) >= 3:
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs) / (len(xs) - 1)
        vy = sum((y - my) ** 2 for y in ys) / (len(ys) - 1)
        # 条件分岐: `vx > 0 and vy > 0` を満たす経路を評価する。
        if vx > 0 and vy > 0:
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)
            corr_log10rho_vs_pct = float(cov / math.sqrt(vx * vy))

    min_pct = float(min(r["pct_per_K_proxy"] for r in rows))
    max_pct = float(max(r["pct_per_K_proxy"] for r in rows))

    out_metrics = out_dir / "condensed_silicon_resistivity_temperature_coefficient_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.4",
                "inputs": {
                    "extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)}
                },
                "results": {
                    "n_samples": int(len(rows)),
                    "counts_by_type_doping": by_type,
                    "min_rho_23C_ohm_cm": float(min(r["rho_23C_ohm_cm"] for r in rows)),
                    "max_rho_23C_ohm_cm": float(max(r["rho_23C_ohm_cm"] for r in rows)),
                    "min_pct_per_K_proxy": min_pct,
                    "max_pct_per_K_proxy": max_pct,
                    "corr_log10rho_vs_pct_per_K_proxy": corr_log10rho_vs_pct,
                    "pct_per_K_proxy_range_by_type_doping": ranges_by_type,
                },
                "falsification": {
                    "target": "d ln rho / dT around room temperature (proxy: 20–30°C)",
                    "accept_range_pct_per_K_proxy": {"min": min_pct, "max": max_pct},
                    "accept_ranges_by_type_doping_pct_per_K_proxy": ranges_by_type,
                    "reject_if_pct_per_K_proxy_non_positive": True,
                    "notes": [
                        "NBS IR 74-496 provides mean resistivity values and confidence limits, but per-sample uncertainties are not propagated here.",
                        "This baseline therefore freezes a necessary-condition envelope (min/max) rather than a precise ±σ target.",
                    ],
                },
                "rows": rows,
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "This is an operational baseline extracted from Appendix E of NBS IR 74-496 (v.d. Pauw method data).",
                    "Coefficient is computed as a finite-difference proxy around room temperature: d ln rho / dT ~ (ln rho30 - ln rho20)/10.",
                    "The report provides confidence limits; this baseline prioritizes mean values for robustness of extraction.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
