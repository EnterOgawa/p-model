from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(x: object) -> float | None:
    try:
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None


# 関数: `_read_tsv_rows` の入出力契約と処理意図を定義する。

def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: list[dict[str, str]] = []
        for row in reader:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            cleaned: dict[str, str] = {}
            for k, v in row.items():
                # 条件分岐: `k is None` を満たす経路を評価する。
                if k is None:
                    continue

                key = str(k).strip()
                # 条件分岐: `not key` を満たす経路を評価する。
                if not key:
                    continue

                cleaned[key] = "" if v is None else str(v).strip()

            rows.append(cleaned)

        return rows


# 関数: `_find_multiplet_components` の入出力契約と処理意図を定義する。

def _find_multiplet_components(
    *,
    rows: list[dict[str, str]],
    center_lambda_vac_A: float,
    window_A: float,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for row in rows:
        obs_nu_str = row.get("obs_nu(A)", "")
        # 条件分岐: `not obs_nu_str` を満たす経路を評価する。
        if not obs_nu_str:
            continue

        aki_str = row.get("Aki(s^-1)", "")
        # 条件分岐: `not aki_str` を満たす経路を評価する。
        if not aki_str:
            continue

        type_str = (row.get("Type") or "").strip()
        # 条件分岐: `type_str` を満たす経路を評価する。
        if type_str:
            # We focus on the E1 multiplet components for the baseline target.
            continue

        obs_nu = _as_float(obs_nu_str)
        aki = _as_float(aki_str)
        # 条件分岐: `obs_nu is None or aki is None or aki <= 0` を満たす経路を評価する。
        if obs_nu is None or aki is None or aki <= 0:
            continue

        # 条件分岐: `math.isclose(obs_nu, 0.0)` を満たす経路を評価する。

        if math.isclose(obs_nu, 0.0):
            continue

        lambda_vac_A = -1.0 / obs_nu
        # 条件分岐: `not math.isfinite(lambda_vac_A) or lambda_vac_A <= 0` を満たす経路を評価する。
        if not math.isfinite(lambda_vac_A) or lambda_vac_A <= 0:
            continue

        # 条件分岐: `abs(lambda_vac_A - center_lambda_vac_A) > window_A` を満たす経路を評価する。

        if abs(lambda_vac_A - center_lambda_vac_A) > window_A:
            continue

        key = (obs_nu_str, row.get("ritz_wn(A)", ""), aki_str, type_str)
        # 条件分岐: `key in seen` を満たす経路を評価する。
        if key in seen:
            continue

        seen.add(key)

        components.append(
            {
                "lambda_vac_nm": float(lambda_vac_A / 10.0),
                "lambda_vac_A": float(lambda_vac_A),
                "nu_obs_invA": float(obs_nu),
                "Aki_s^-1": float(aki),
                "Acc": row.get("Acc") or None,
                "Type": None,
            }
        )

    components.sort(key=lambda r: float(r["lambda_vac_nm"]))
    return components


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source: NIST ASD cached output (Phase 7 / Step 7.12)
    src_dir = root / "data" / "quantum" / "sources" / "nist_asd_h_i_lines"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing extracted values: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_nist_asd_lines.py --spectra \"H I\""
        )

    extracted = _read_json(extracted_path)
    selected = extracted.get("selected_lines")
    # 条件分岐: `not isinstance(selected, list) or not selected` を満たす経路を評価する。
    if not isinstance(selected, list) or not selected:
        raise SystemExit(f"[fail] selected_lines missing/empty in: {extracted_path}")

    raw_file = extracted.get("raw_file")
    raw_path = Path(raw_file) if isinstance(raw_file, str) else None
    # 条件分岐: `raw_path is None` を満たす経路を評価する。
    if raw_path is None:
        raise SystemExit(f"[fail] raw_file missing in extracted values: {extracted_path}")

    # 条件分岐: `not raw_path.exists()` を満たす経路を評価する。

    if not raw_path.exists():
        raw_path = src_dir / raw_path.name

    # 条件分岐: `not raw_path.exists()` を満たす経路を評価する。

    if not raw_path.exists():
        raise SystemExit(f"[fail] raw TSV missing: {raw_path}")

    raw_rows = _read_tsv_rows(raw_path)

    c = 299_792_458.0
    h = 6.626_070_15e-34  # exact (SI)
    e_charge = 1.602_176_634e-19  # exact (J/eV)

    lines_out: list[dict[str, Any]] = []
    for rec in selected:
        # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
        if not isinstance(rec, dict):
            continue

        sel = rec.get("selected")
        # 条件分岐: `not isinstance(sel, dict)` を満たす経路を評価する。
        if not isinstance(sel, dict):
            continue

        lam_nm = _as_float(sel.get("lambda_vac_nm"))
        lam_unc_A = _as_float(sel.get("lambda_vac_unc_A"))
        aki = _as_float(sel.get("Aki_s^-1"))
        # 条件分岐: `lam_nm is None or lam_nm <= 0` を満たす経路を評価する。
        if lam_nm is None or lam_nm <= 0:
            continue

        lam_m = lam_nm * 1e-9
        freq_hz = c / lam_m
        energy_ev = (h * freq_hz) / e_charge

        lines_out.append(
            {
                "id": str(rec.get("id") or ""),
                "lambda_vac_nm": float(lam_nm),
                "lambda_vac_unc_nm": (None if lam_unc_A is None else float(lam_unc_A / 10.0)),
                "frequency_THz": float(freq_hz / 1e12),
                "photon_energy_eV": float(energy_ev),
                "Aki_s^-1": (None if aki is None else float(aki)),
                "Acc": sel.get("Acc"),
                "Type": sel.get("Type"),
            }
        )

    # 条件分岐: `not lines_out` を満たす経路を評価する。

    if not lines_out:
        raise SystemExit("[fail] no usable lines parsed from extracted_values.json")

    lines_out.sort(key=lambda r: float(r["lambda_vac_nm"]))

    # ---- Fine-structure multiplets (observed, E1) ----
    # We extract nearby observed components around the baseline targets to make
    # "fine structure exists and is fixed as a target" explicit, without claiming
    # a first-principles derivation here.
    multiplet_window_A = 0.5
    multiplets_out: list[dict[str, Any]] = []
    for rec in selected:
        # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
        if not isinstance(rec, dict):
            continue

        line_id = str(rec.get("id") or "")
        center_A = _as_float(rec.get("approx_lambda_vac_A"))
        # 条件分岐: `not line_id or center_A is None or center_A <= 0` を満たす経路を評価する。
        if not line_id or center_A is None or center_A <= 0:
            continue

        comps = _find_multiplet_components(
            rows=raw_rows,
            center_lambda_vac_A=float(center_A),
            window_A=multiplet_window_A,
        )
        # 条件分岐: `not comps` を満たす経路を評価する。
        if not comps:
            continue

        lam_min = float(comps[0]["lambda_vac_nm"])
        lam_max = float(comps[-1]["lambda_vac_nm"])
        multiplets_out.append(
            {
                "id": line_id,
                "center_approx_nm": float(center_A / 10.0),
                "window_A": float(multiplet_window_A),
                "filter": "obs_nu present + Type empty (E1) + Aki present",
                "n_components": len(comps),
                "lambda_min_nm": lam_min,
                "lambda_max_nm": lam_max,
                "delta_lambda_pm": float((lam_max - lam_min) * 1e3),
                "components": comps,
            }
        )

    multiplet_label = {
        "H_I_Lyα": "Lyα",
        "H_I_Hα": "Hα",
        "H_I_Hβ": "Hβ",
        "H_I_Hγ": "Hγ",
    }
    multiplet_note_lines: list[str] = []
    for m in multiplets_out:
        n = m.get("n_components")
        # 条件分岐: `not isinstance(n, int) or n <= 1` を満たす経路を評価する。
        if not isinstance(n, int) or n <= 1:
            continue

        label = multiplet_label.get(str(m.get("id") or ""), str(m.get("id") or ""))
        mn = m.get("lambda_min_nm")
        mx = m.get("lambda_max_nm")
        # 条件分岐: `not isinstance(mn, (int, float)) or not isinstance(mx, (int, float))` を満たす経路を評価する。
        if not isinstance(mn, (int, float)) or not isinstance(mx, (int, float)):
            continue

        multiplet_note_lines.append(f"{label}: {mn:.3f}–{mx:.3f} nm (N={n})")

    # ---- Figure ----

    fig_w, fig_h, dpi = 11.5, 4.6, 180
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_title("Phase 7 / Step 7.12: Atomic baseline (Hydrogen, NIST ASD)", fontsize=13)
    ax.set_xlabel("Vacuum wavelength λ [nm] (from NIST ASD)")
    ax.set_yticks([])
    ax.set_xlim(90, 720)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="x", alpha=0.25)

    for i, r in enumerate(lines_out):
        lam = float(r["lambda_vac_nm"])
        label = str(r.get("id") or "")
        # Color by rough region (UV vs visible)
        color = "#2b6cb0" if lam < 200 else "#c53030"
        ax.axvline(lam, color=color, lw=2.2, alpha=0.95)
        y = 0.82 if (i % 2 == 0) else 0.58
        ax.text(
            lam + 2.0,
            y,
            f"{label}\n{lam:.3f} nm",
            fontsize=9.6,
            ha="left",
            va="center",
            color=color,
        )

    ax.text(
        0.01,
        0.05,
        "Data: NIST ASD (lines1.pl)\n"
        "This figure fixes baseline targets for Part III; it is not a derivation.\n"
        + ("" if not multiplet_note_lines else "Fine-structure multiplets (obs, E1): " + " / ".join(multiplet_note_lines)),
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    fig.tight_layout()
    out_png = out_dir / "atomic_hydrogen_baseline.png"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": "Atomic baseline (Hydrogen; NIST ASD)",
        "source_cache": {
            "extracted_values": str(extracted_path),
            "raw_tsv": str(raw_path),
        },
        "parameters": {
            "multiplet_window_A": multiplet_window_A,
            "multiplet_filter": "obs_nu present + Type empty (E1) + Aki present",
        },
        "lines": lines_out,
        "multiplets": multiplets_out,
        "note": (
            "This output fixes a small set of observed vacuum wavelengths as a reproducible baseline. "
            "P-model derivation of atomic/molecular binding is tracked in Roadmap Step 7.12+."
        ),
    }
    out_json = out_dir / "atomic_hydrogen_baseline_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
