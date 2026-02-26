from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _load_ioffe_elastic_constants(root: Path) -> dict[str, Any]:
    src = root / "data" / "quantum" / "sources" / "ioffe_silicon_mechanical_properties" / "extracted_values.json"
    # 条件分岐: `not src.exists()` を満たす経路を評価する。
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
    # 条件分岐: `t < float(t_min)` を満たす経路を評価する。
    if t < float(t_min):
        t = float(t_min)

    # 条件分岐: `t > float(t_max)` を満たす経路を評価する。

    if t > float(t_max):
        t = float(t_max)

    return float(intercept) + float(slope) * t


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    ioffe = _load_ioffe_elastic_constants(root)
    obj = ioffe["data"]

    vals = obj.get("values", {})
    # 条件分岐: `not isinstance(vals, dict)` を満たす経路を評価する。
    if not isinstance(vals, dict):
        raise SystemExit("[fail] invalid extracted_values.json: missing 'values' dict")

    b_ref_1e11 = float(vals.get("bulk_modulus_from_C11_C12_1e11_dyn_cm2"))

    temp_dep = obj.get("temperature_dependence_linear", {})
    # 条件分岐: `not isinstance(temp_dep, dict)` を満たす経路を評価する。
    if not isinstance(temp_dep, dict):
        raise SystemExit("[fail] invalid extracted_values.json: missing 'temperature_dependence_linear' dict")

    tr = temp_dep.get("T_range_K", {})
    # 条件分岐: `not isinstance(tr, dict)` を満たす経路を評価する。
    if not isinstance(tr, dict):
        raise SystemExit("[fail] invalid extracted_values.json: missing 'T_range_K' dict")

    t_lin_min = float(tr.get("min"))
    t_lin_max = float(tr.get("max"))

    c11 = temp_dep.get("C11", {})
    c12 = temp_dep.get("C12", {})
    # 条件分岐: `not isinstance(c11, dict) or not isinstance(c12, dict)` を満たす経路を評価する。
    if not isinstance(c11, dict) or not isinstance(c12, dict):
        raise SystemExit("[fail] invalid extracted_values.json: missing C11/C12 linear dicts")

    c11_a = float(c11.get("intercept_1e11_dyn_cm2"))
    c11_b = float(c11.get("slope_1e11_dyn_cm2_per_K"))
    c12_a = float(c12.get("intercept_1e11_dyn_cm2"))
    c12_b = float(c12.get("slope_1e11_dyn_cm2_per_K"))

    def b_lin_1e11(t_k: float) -> float:
        c11_t = _cij_linear(t_k=float(t_k), intercept=c11_a, slope=c11_b, t_min=t_lin_min, t_max=t_lin_max)
        c12_t = _cij_linear(t_k=float(t_k), intercept=c12_a, slope=c12_b, t_min=t_lin_min, t_max=t_lin_max)
        return float((c11_t + 2.0 * c12_t) / 3.0)

    # Target range: Si α(T) NIST TRC fit is stated usable up to 600K.

    t_min = 0
    t_max = 600
    t_switch = float(t_lin_min)

    # Ensure continuity at t_switch by shifting the linear-Cij curve to match the room-temperature B_ref.
    b_lin_at_switch = float(b_lin_1e11(t_switch))
    b_offset_1e11 = float(b_ref_1e11 - b_lin_at_switch)

    rows: list[dict[str, float | int | str]] = []
    temps: list[float] = []
    b_gpa: list[float] = []
    for t in range(t_min, t_max + 1):
        # 条件分岐: `float(t) < t_switch` を満たす経路を評価する。
        if float(t) < t_switch:
            b_1e11 = float(b_ref_1e11)
            kind = "const"
        else:
            b_1e11 = float(b_lin_1e11(float(t))) + float(b_offset_1e11)
            kind = "linear_Cij_shifted"

        temps.append(float(t))
        b_gpa.append(_bulk_modulus_GPa_from_1e11_dyn_cm2(b_1e11))
        rows.append(
            {
                "T_K": int(t),
                "B_1e11_dyn_cm2": float(b_1e11),
                "B_GPa": float(_bulk_modulus_GPa_from_1e11_dyn_cm2(b_1e11)),
                "model": kind,
            }
        )

    # Write CSV

    out_csv = out_dir / "condensed_silicon_bulk_modulus_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["T_K", "B_1e11_dyn_cm2", "B_GPa", "model"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Figure

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.6))
    ax.plot(temps, b_gpa, color="#1f77b4", lw=2.0, label="B(T) reference")
    ax.axvline(t_switch, color="#999999", lw=1.0, ls="--", alpha=0.85, label=f"switch at {t_switch:.0f} K")
    ax.set_title("Silicon bulk modulus B(T) (Ioffe elastic constants reference)")
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel("Bulk modulus B (GPa)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_png = out_dir / "condensed_silicon_bulk_modulus_baseline.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    b0 = _bulk_modulus_GPa_from_1e11_dyn_cm2(b_ref_1e11)
    b_switch_raw = _bulk_modulus_GPa_from_1e11_dyn_cm2(b_lin_at_switch)
    b_switch = _bulk_modulus_GPa_from_1e11_dyn_cm2(b_lin_at_switch + b_offset_1e11)
    b600 = float(b_gpa[600])

    out_metrics = out_dir / "condensed_silicon_bulk_modulus_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.16",
                "inputs": {
                    "ioffe_silicon_mechanical_properties_extracted_values": {"path": str(ioffe["path"]), "sha256": str(ioffe["sha256"])},
                },
                "model": {
                    "kind": "piecewise",
                    "below_T_K": float(t_switch),
                    "below": {"kind": "const", "B_1e11_dyn_cm2": float(b_ref_1e11)},
                    "above": {
                        "kind": "linear_Cij_shifted",
                        "valid_T_range_K": {"min": float(t_lin_min), "max": float(t_lin_max)},
                        "C11": {"intercept_1e11_dyn_cm2": float(c11_a), "slope_1e11_dyn_cm2_per_K": float(c11_b)},
                        "C12": {"intercept_1e11_dyn_cm2": float(c12_a), "slope_1e11_dyn_cm2_per_K": float(c12_b)},
                        "offset_1e11_dyn_cm2": float(b_offset_1e11),
                        "definition": "B(T)=(C11(T)+2*C12(T))/3",
                    },
                },
                "summary": {
                    "B_GPa_at_const": float(b0),
                    "B_GPa_at_switch_linear_raw": float(b_switch_raw),
                    "B_GPa_at_switch_linear_shifted": float(b_switch),
                    "B_GPa_at_600K": float(b600),
                    "rel_change_switch_to_600": float((b600 - b_switch) / b_switch),
                },
                "notes": [
                    "This B(T) is used as a reference input for Step 7.14 thermal expansion (Grüneisen) analyses.",
                    "The underlying linear Cij(T) approximation is stated on the Ioffe page for 400K<T<873K; below 400K we hold B constant at the room-temperature reference value.",
                    "To avoid a discontinuity at 400K, the linear-Cij curve is shifted to match B_ref at the switch temperature; only the slope is taken from the linear approximation.",
                    "The absolute uncertainty of B(T) is not provided here; this input is treated as a reference curve (systematic to be refined with a stricter primary source if needed).",
                ],
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
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


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
