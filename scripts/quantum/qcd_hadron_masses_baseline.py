from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


HBAR_C_MEV_FM = 197.3269804


# クラス: `PdgRow` の責務と境界条件を定義する。
@dataclass(frozen=True)
class PdgRow:
    pdg_id: int
    mass_gev: Optional[float]
    mass_err_plus_gev: Optional[float]
    mass_err_minus_gev: Optional[float]
    name_field: str


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(s: str) -> Optional[float]:
    t = s.strip()
    # 条件分岐: `not t` を満たす経路を評価する。
    if not t:
        return None

    try:
        v = float(t.replace("D", "E"))
    except Exception:
        return None

    # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。

    if not math.isfinite(v):
        return None

    return float(v)


# 関数: `_parse_pdg_mcdata_mass_width` の入出力契約と処理意図を定義する。

def _parse_pdg_mcdata_mass_width(lines: Iterable[str]) -> Dict[int, PdgRow]:
    rows: Dict[int, PdgRow] = {}
    for raw in lines:
        line = raw.rstrip("\n")
        # 条件分岐: `not line or line.startswith("*")` を満たす経路を評価する。
        if not line or line.startswith("*"):
            continue

        # 条件分岐: `len(line) < 8` を満たす経路を評価する。

        if len(line) < 8:
            continue

        ids: List[int] = []
        for k in range(4):
            seg = line[k * 8 : (k + 1) * 8].strip()
            # 条件分岐: `seg` を満たす経路を評価する。
            if seg:
                try:
                    ids.append(int(seg))
                except Exception:
                    pass

        # 条件分岐: `not ids` を満たす経路を評価する。

        if not ids:
            continue

        mass_gev = _safe_float(line[33:51])
        mass_err_plus_gev = _safe_float(line[52:60])
        mass_err_minus_gev = _safe_float(line[61:69])
        name_field = line[107:128].strip() if len(line) >= 108 else ""

        pdg_id = ids[0]
        # 条件分岐: `pdg_id in rows` を満たす経路を評価する。
        if pdg_id in rows:
            continue

        rows[pdg_id] = PdgRow(
            pdg_id=pdg_id,
            mass_gev=mass_gev,
            mass_err_plus_gev=mass_err_plus_gev,
            mass_err_minus_gev=mass_err_minus_gev,
            name_field=name_field,
        )

    return rows


# 関数: `_mev` の入出力契約と処理意図を定義する。

def _mev(x_gev: Optional[float]) -> Optional[float]:
    # 条件分岐: `x_gev is None` を満たす経路を評価する。
    if x_gev is None:
        return None

    return float(x_gev) * 1e3


# 関数: `_compton_lambda_fm` の入出力契約と処理意図を定義する。

def _compton_lambda_fm(mass_mev: Optional[float]) -> Optional[float]:
    # 条件分岐: `mass_mev is None or mass_mev <= 0` を満たす経路を評価する。
    if mass_mev is None or mass_mev <= 0:
        return None

    return HBAR_C_MEV_FM / mass_mev


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 7 / Step 7.13: fix a hadron-scale baseline using PDG RPP 2024 masses. "
            "Outputs a small summary plot + machine-readable metrics."
        )
    )
    ap.add_argument(
        "--pdg-file",
        default=None,
        help="Path to PDG mass_width_2024.txt (default: repo cache under data/quantum/sources/...).",
    )
    ap.add_argument(
        "--out-tag",
        default="qcd_hadron_masses_baseline",
        help="Output tag for png/json/csv (default: qcd_hadron_masses_baseline).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    pdg_path = (
        Path(str(args.pdg_file))
        if args.pdg_file
        else root / "data" / "quantum" / "sources" / "pdg_rpp_2024_mass_width" / "mass_width_2024.txt"
    )
    # 条件分岐: `not pdg_path.exists()` を満たす経路を評価する。
    if not pdg_path.exists():
        raise SystemExit(
            "[fail] missing PDG file. Run:\n"
            "  python -B scripts/quantum/fetch_pdg_mass_width_2024.py\n"
            f"expected: {pdg_path}"
        )

    rows = _parse_pdg_mcdata_mass_width(pdg_path.read_text(encoding="utf-8", errors="replace").splitlines())

    # Keep the list short and directly relevant to nuclear-force/QCD scale discussions.
    targets = [
        ("p", 2212),
        ("n", 2112),
        ("π±", 211),
        ("π0", 111),
        ("K±", 321),
        ("Λ0", 3122),
        ("Σ+", 3222),
        ("Ξ0", 3322),
    ]

    out_rows = []
    missing_ids = []
    for label, pid in targets:
        r = rows.get(pid)
        # 条件分岐: `r is None or r.mass_gev is None` を満たす経路を評価する。
        if r is None or r.mass_gev is None:
            missing_ids.append(pid)
            continue

        mass_mev = _mev(r.mass_gev)
        out_rows.append(
            {
                "label": label,
                "pdg_id": int(pid),
                "name_field": r.name_field,
                "mass_mev": mass_mev,
                "mass_err_plus_mev": _mev(r.mass_err_plus_gev),
                "mass_err_minus_mev": _mev(r.mass_err_minus_gev),
                "compton_lambda_fm": _compton_lambda_fm(mass_mev),
            }
        )

    # 条件分岐: `missing_ids` を満たす経路を評価する。

    if missing_ids:
        raise SystemExit(f"[fail] missing PDG ids in file: {missing_ids}")

    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_tag = str(args.out_tag)
    out_png = out_dir / f"{out_tag}.png"
    out_json = out_dir / f"{out_tag}_metrics.json"
    out_csv = out_dir / f"{out_tag}.csv"

    # --- Plot
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in out_rows]
    masses = [float(r["mass_mev"]) for r in out_rows]
    lambdas = [float(r["compton_lambda_fm"]) for r in out_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=150)

    ax = axes[0]
    ax.bar(labels, masses, color="tab:blue", alpha=0.85)
    ax.set_title("Hadron mass baseline (PDG RPP 2024)")
    ax.set_ylabel("mass (MeV)")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    ax.bar(labels, lambdas, color="tab:orange", alpha=0.85)
    ax.set_title("Compton wavelength scale")
    ax.set_ylabel("λ = ħc/(mc²) (fm)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[ok] png : {out_png}")

    # --- Metrics + CSV
    metrics = {
        "generated_utc": _utc_now(),
        "step": "Phase 7 / Step 7.13 (QCD/hadron baseline)",
        "pdg_source": {
            "path": str(pdg_path),
            "edition": "RPP 2024",
            "url": "https://pdg.lbl.gov/2024/mcdata/mass_width_2024.txt",
        },
        "constants": {"hbar_c_mev_fm": HBAR_C_MEV_FM},
        "rows": out_rows,
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] json: {out_json}")

    header = ["label", "pdg_id", "name_field", "mass_mev", "mass_err_plus_mev", "mass_err_minus_mev", "compton_lambda_fm"]
    lines = [",".join(header)]
    for r in out_rows:
        lines.append(
            ",".join(
                [
                    str(r["label"]),
                    str(r["pdg_id"]),
                    str(r["name_field"]).replace(",", " "),
                    f"{float(r['mass_mev']):.9g}",
                    "" if r["mass_err_plus_mev"] is None else f"{float(r['mass_err_plus_mev']):.9g}",
                    "" if r["mass_err_minus_mev"] is None else f"{float(r['mass_err_minus_mev']):.9g}",
                    "" if r["compton_lambda_fm"] is None else f"{float(r['compton_lambda_fm']):.9g}",
                ]
            )
        )

    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] csv : {out_csv}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

