from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_sha256` の入出力契約と処理意図を定義する。

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


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_cp_shomate` の入出力契約と処理意図を定義する。

def _cp_shomate(*, coeffs: dict[str, float], t_k: float) -> float:
    t = t_k / 1000.0
    a = float(coeffs["A"])
    b = float(coeffs["B"])
    c = float(coeffs["C"])
    d = float(coeffs["D"])
    e = float(coeffs["E"])
    # Cp° = A + B*t + C*t^2 + D*t^3 + E/t^2
    return a + b * t + c * (t**2) + d * (t**3) + (e / (t**2))


# 関数: `_linspace_inclusive` の入出力契約と処理意図を定義する。

def _linspace_inclusive(x0: float, x1: float, n: int) -> list[float]:
    # 条件分岐: `n <= 1` を満たす経路を評価する。
    if n <= 1:
        return [x0]

    step = (x1 - x0) / (n - 1)
    return [x0 + i * step for i in range(n)]


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_webbook_condensed_silicon_si"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_silicon_condensed_thermochemistry_sources.py"
        )

    extracted = _read_json(extracted_path)
    shomate = extracted.get("shomate")
    # 条件分岐: `not isinstance(shomate, list) or not shomate` を満たす経路を評価する。
    if not isinstance(shomate, list) or not shomate:
        raise SystemExit(f"[fail] shomate blocks missing/empty: {extracted_path}")

    blocks: list[dict[str, Any]] = []
    for b in shomate:
        # 条件分岐: `not isinstance(b, dict)` を満たす経路を評価する。
        if not isinstance(b, dict):
            continue

        phase = str(b.get("phase") or "")
        coeffs = b.get("coeffs")
        # 条件分岐: `not phase or not isinstance(coeffs, dict)` を満たす経路を評価する。
        if not phase or not isinstance(coeffs, dict):
            continue

        t_min = float(b.get("t_min_k"))
        t_max = float(b.get("t_max_k"))
        blocks.append({"phase": phase, "t_min_k": t_min, "t_max_k": t_max, "coeffs": coeffs})

    # 条件分岐: `not blocks` を満たす経路を評価する。

    if not blocks:
        raise SystemExit(f"[fail] no usable Shomate blocks parsed: {extracted_path}")

    # Sample points (dense enough for plotting).

    rows: list[dict[str, Any]] = []
    for b in sorted(blocks, key=lambda x: float(x["t_min_k"])):
        t0 = float(b["t_min_k"])
        t1 = float(b["t_max_k"])
        phase = str(b["phase"])
        coeffs = b["coeffs"]

        n = 250 if (t1 - t0) >= 1000 else 180
        for t_k in _linspace_inclusive(t0, t1, n):
            cp = _cp_shomate(coeffs=coeffs, t_k=float(t_k))
            rows.append(
                {
                    "phase": phase,
                    "T_K": float(t_k),
                    "Cp_J_per_molK": float(cp),
                }
            )

    # CSV

    out_csv = out_dir / "condensed_silicon_heat_capacity_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "T_K", "Cp_J_per_molK"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot

    plt.figure(figsize=(8.5, 4.6))
    colors = {"solid": "#1f77b4", "liquid": "#d62728"}
    for phase in sorted({r["phase"] for r in rows}):
        xs = [float(r["T_K"]) for r in rows if r["phase"] == phase]
        ys = [float(r["Cp_J_per_molK"]) for r in rows if r["phase"] == phase]
        plt.plot(xs, ys, label=phase, color=colors.get(phase, None))

    # Mark melting point boundary used in WebBook table split (if present).

    t_melt = None
    for b in blocks:
        # 条件分岐: `str(b["phase"]) == "solid"` を満たす経路を評価する。
        if str(b["phase"]) == "solid":
            t_melt = float(b["t_max_k"])

    # 条件分岐: `t_melt is not None` を満たす経路を評価する。

    if t_melt is not None:
        plt.axvline(t_melt, color="k", alpha=0.25, linewidth=1)
        plt.text(t_melt, plt.ylim()[0], " melt", rotation=90, va="bottom", ha="left", alpha=0.6)

    plt.xlabel("Temperature T (K)")
    plt.ylabel("Cp° (J/mol·K)")
    plt.title("Silicon heat capacity (Shomate; NIST WebBook condensed phase)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / "condensed_silicon_heat_capacity_baseline.png"
    plt.savefig(out_png, dpi=180)
    plt.close()

    # Key points for quick quoting in docs.
    key_points = []
    for target_t in [298.15, 300.0, 1000.0, 1685.0, 2000.0, 3000.0]:
        # Pick the first block whose range covers target_t.
        picked = None
        for b in blocks:
            # 条件分岐: `float(b["t_min_k"]) <= target_t <= float(b["t_max_k"]) + 1e-9` を満たす経路を評価する。
            if float(b["t_min_k"]) <= target_t <= float(b["t_max_k"]) + 1e-9:
                picked = b
                break

        # 条件分岐: `picked is None` を満たす経路を評価する。

        if picked is None:
            continue

        cp = _cp_shomate(coeffs=picked["coeffs"], t_k=float(target_t))
        key_points.append(
            {
                "phase": str(picked["phase"]),
                "T_K": float(target_t),
                "Cp_J_per_molK": float(cp),
            }
        )

    sigma_multiplier = 3.0
    rel_sigma_proxy = 0.02
    abs_sigma_floor = 0.2  # J/mol·K
    falsification_targets = []
    for kp in key_points:
        cp = float(kp["Cp_J_per_molK"])
        sigma_proxy = max(rel_sigma_proxy * abs(cp), abs_sigma_floor)
        falsification_targets.append(
            {
                "phase": str(kp["phase"]),
                "T_K": float(kp["T_K"]),
                "Cp_target_J_per_molK": cp,
                "sigma_proxy_J_per_molK": float(sigma_proxy),
                "reject_if_abs_Cp_minus_target_gt_J_per_molK": float(sigma_multiplier * sigma_proxy),
            }
        )

    out_metrics = out_dir / "condensed_silicon_heat_capacity_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.2",
                "inputs": {
                    "extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)}
                },
                "shomate_blocks": blocks,
                "key_points": key_points,
                "falsification": {
                    "sigma_multiplier": sigma_multiplier,
                    "sigma_proxy": {"rel": rel_sigma_proxy, "abs_floor_J_per_molK": abs_sigma_floor},
                    "targets": falsification_targets,
                    "notes": [
                        "NIST WebBook condensed Shomate blocks do not provide per-point uncertainties in the table view.",
                        "We use a conservative proxy uncertainty: max(2% of Cp, 0.2 J/mol·K), and reject if deviation exceeds ±3σ_proxy.",
                    ],
                },
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "Cp° computed from NIST Shomate coefficients; t=T/1000.",
                    "WebBook condensed-phase Cp table ranges start at 298 K; low-temperature (Debye T^3) regime is not covered here.",
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
