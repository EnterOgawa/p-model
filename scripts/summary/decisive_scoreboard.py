from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class ZRow:
    id: str
    label: str
    z: float
    observed: str
    predicted: str
    sigma: str
    kind: str  # fit/predict
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "z": float(self.z),
            "abs_z": float(abs(self.z)),
            "observed": self.observed,
            "predicted": self.predicted,
            "sigma": self.sigma,
            "kind": self.kind,
            "note": self.note,
        }


def _load_frozen(root: Path, path: Optional[Path]) -> Dict[str, Any]:
    if path and path.exists():
        return _read_json(path)
    default = root / "output" / "private" / "theory" / "frozen_parameters.json"
    if default.exists():
        return _read_json(default)
    return {"beta": 1.0, "beta_sigma": None, "gamma_pmodel": 1.0, "policy": {"beta_source": "default_beta_1"}}


def _format_float(x: float, *, digits: int = 6) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-3 or ax >= 1e5:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _z_fit_cassini_gamma(frozen: Dict[str, Any]) -> Optional[ZRow]:
    # If beta is frozen from Cassini γ, we can show it as a FIT row (z=0 by construction).
    constraints = frozen.get("constraints")
    if not isinstance(constraints, list):
        return None

    cassini = None
    for c in constraints:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "")
        if cid.startswith("cassini_2003_"):
            cassini = c
            break
    if not cassini:
        return None

    gamma = float(cassini.get("gamma"))
    sigma = float(cassini.get("sigma"))
    beta = float(frozen.get("beta", 1.0))
    gamma_pred = float(2.0 * beta - 1.0)
    z = 0.0 if sigma > 0 else float("nan")
    if sigma > 0:
        z = (gamma_pred - gamma) / sigma

    return ZRow(
        id="fit_cassini_gamma",
        label="Cassini：PPN γ（fit）",
        z=z,
        observed=f"γ={_format_float(gamma, digits=8)}",
        predicted=f"γ={_format_float(gamma_pred, digits=8)}",
        sigma=f"σ={_format_float(sigma, digits=3)}",
        kind="fit",
        note="βをCassiniのγ拘束から決めた場合、ここは定義上ほぼ一致（z≈0）。",
    )


def _z_solar_deflection(root: Path, frozen: Dict[str, Any]) -> Optional[ZRow]:
    path = root / "output" / "private" / "theory" / "solar_light_deflection_metrics.json"
    if not path.exists():
        return None
    j = _read_json(path)
    m = dict(j.get("metrics") or {})
    obs_gamma = float(m["observed_gamma_best"])
    obs_sigma = float(m["observed_gamma_best_sigma"])
    obs_label = str(m.get("observed_best_label") or "VLBI")
    beta = float(frozen.get("beta", 1.0))
    gamma_pred = float(2.0 * beta - 1.0)
    if obs_sigma <= 0:
        return None
    z = (gamma_pred - obs_gamma) / obs_sigma
    return ZRow(
        id="solar_deflection_gamma",
        label=f"太陽光偏向：PPN γ（{obs_label}）",
        z=z,
        observed=f"γ={_format_float(obs_gamma, digits=8)}",
        predicted=f"γ={_format_float(gamma_pred, digits=8)}",
        sigma=f"σ={_format_float(obs_sigma, digits=3)}",
        kind="predict",
        note="VLBIのPPN γ推定（一次ソース）との比較。P-modelでは γ=2β−1。",
    )


def _z_gravitational_redshift(root: Path) -> List[ZRow]:
    path = root / "output" / "private" / "theory" / "gravitational_redshift_experiments.json"
    if not path.exists():
        return []
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    out: List[ZRow] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        epsilon = float(r["epsilon"])
        sigma = float(r["sigma"])
        if sigma <= 0:
            continue
        z = (0.0 - epsilon) / sigma
        label = str(r.get("short_label") or r.get("id") or "redshift")
        out.append(
            ZRow(
                id=f"redshift_{r.get('id')}",
                label=f"重力赤方偏移：ε（{label}）",
                z=z,
                observed=f"ε={_format_float(epsilon, digits=3)}",
                predicted="ε=0",
                sigma=f"σ={_format_float(sigma, digits=3)}",
                kind="predict",
                note="定義：z_obs=(1+ε)ΔU/c^2。弱場のP-model予測はε=0（GRと同じ一次）。",
            )
        )
    return out


def _z_eht_ring_vs_shadow(root: Path, frozen: Dict[str, Any]) -> List[ZRow]:
    path = root / "output" / "private" / "eht" / "eht_shadow_compare.json"
    if not path.exists():
        return []
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    coeff_rg_beta1 = float(j.get("pmodel", {}).get("shadow_diameter_coeff_rg", float("nan")))
    beta = float(frozen.get("beta", 1.0))
    out: List[ZRow] = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or r.get("key") or "EHT")
        ring = float(r["ring_diameter_obs_uas"])
        ring_sigma = float(r["ring_diameter_obs_uas_sigma"])
        theta_unit = float(r["theta_unit_uas"])
        theta_unit_sigma = float(r["theta_unit_uas_sigma"])
        if ring_sigma <= 0 or theta_unit <= 0:
            continue

        # P-model shadow diameter prediction (β scales linearly)
        shadow = coeff_rg_beta1 * beta * theta_unit
        shadow_sigma = abs(shadow) * abs(theta_unit_sigma / theta_unit) if theta_unit_sigma > 0 else 0.0

        # Compare ring vs shadow under κ=1, with combined uncertainty (obs + model param).
        sigma_tot = math.sqrt(ring_sigma * ring_sigma + shadow_sigma * shadow_sigma)
        if sigma_tot <= 0:
            continue
        z = (shadow - ring) / sigma_tot

        out.append(
            ZRow(
                id=f"eht_{r.get('key')}",
                label=f"EHT：リング直径 ≈ シャドウ直径（{name}, κ=1）",
                z=z,
                observed=f"{_format_float(ring, digits=4)}±{_format_float(ring_sigma, digits=3)} μas",
                predicted=f"{_format_float(shadow, digits=4)}±{_format_float(shadow_sigma, digits=3)} μas",
                sigma=f"σ_tot={_format_float(sigma_tot, digits=3)} μas",
                kind="predict",
                note="EHTは“リング直径”の推定であり、理論の“シャドウ直径”とは同一ではない。ここでは κ=1 を仮定して整合性を確認する。",
            )
        )
    return out


def build_scoreboard(root: Path, *, frozen: Dict[str, Any]) -> Dict[str, Any]:
    beta = float(frozen.get("beta", 1.0))
    beta_sigma = frozen.get("beta_sigma")
    gamma_pred = float(2.0 * beta - 1.0)

    zrows: List[ZRow] = []
    fit_row = _z_fit_cassini_gamma(frozen)
    if fit_row:
        zrows.append(fit_row)

    solar = _z_solar_deflection(root, frozen)
    if solar:
        zrows.append(solar)

    zrows.extend(_z_gravitational_redshift(root))
    zrows.extend(_z_eht_ring_vs_shadow(root, frozen))

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "beta": beta,
        "beta_sigma": beta_sigma,
        "gamma_pmodel": gamma_pred,
        "beta_source": str((frozen.get("policy") or {}).get("beta_source") or ""),
        "rows": [zr.to_dict() for zr in zrows],
        "notes": [
            "zスコアは（予測−観測）/σで表示（|z|が小さいほど整合）。",
            "EHTはリングとシャドウの対応（κ）やスピン等が支配的な系統誤差になるため、κ=1は整合性チェックの入口として扱う。",
        ],
    }
    return payload


def plot_scoreboard(rows: List[ZRow], *, beta: float, beta_source: str, out_png: Path) -> None:
    _set_japanese_font()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Sort: predict rows by |z| desc (fit row stays on top if present)
    fit = [r for r in rows if r.kind == "fit"]
    pred = sorted([r for r in rows if r.kind != "fit"], key=lambda x: abs(x.z), reverse=True)
    ordered = fit + pred

    labels = [r.label for r in ordered]
    zvals = [r.z for r in ordered]

    # Color by abs(z)
    colors = []
    for r in ordered:
        if r.kind == "fit":
            colors.append("#7f7f7f")
        else:
            az = abs(r.z)
            if az <= 1.0:
                colors.append("#2ca02c")
            elif az <= 2.0:
                # Yellow (needs further verification)
                colors.append("#f1c232")
            else:
                colors.append("#d62728")

    fig_h = max(4.0, 0.55 * len(ordered) + 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(12.5, fig_h))
    y = list(range(len(ordered)))
    ax.barh(y, zvals, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    for x in (-2.0, -1.0, 1.0, 2.0):
        ax.axvline(x, color="#999999", linewidth=1.0, linestyle="--")
    ax.set_xlabel("z スコア（予測−観測）/σ")
    ax.set_title(f"決定的スコアボード（β固定: β={beta:.9f}, source={beta_source}）")
    ax.invert_yaxis()

    # Annotate each bar with compact text
    for i, r in enumerate(ordered):
        s = f"{r.observed} / {r.predicted}"
        ax.text(
            0.02,
            i,
            s,
            va="center",
            ha="left",
            fontsize=9,
            color="#111111",
            transform=ax.get_yaxis_transform(),
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    root = _repo_root()
    default_outdir = root / "output" / "private" / "summary"
    default_out_json = default_outdir / "decisive_scoreboard.json"
    default_out_png = default_outdir / "decisive_scoreboard.png"

    ap = argparse.ArgumentParser(description="Build Phase 7 decisive scoreboard (z-score summary).")
    ap.add_argument(
        "--frozen",
        type=str,
        default="",
        help="Frozen parameters JSON (default: output/private/theory/frozen_parameters.json)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(default_out_json),
        help="Output JSON path (default: output/private/summary/decisive_scoreboard.json)",
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default=str(default_out_png),
        help="Output PNG path (default: output/private/summary/decisive_scoreboard.png)",
    )
    args = ap.parse_args()

    frozen_path = Path(args.frozen) if args.frozen else None
    frozen = _load_frozen(root, frozen_path)

    payload = build_scoreboard(root, frozen=frozen)
    out_json = Path(args.out_json)
    out_png = Path(args.out_png)

    # Recreate ZRow objects from payload rows for plotting
    zrows: List[ZRow] = []
    for r in payload.get("rows") or []:
        if not isinstance(r, dict):
            continue
        zrows.append(
            ZRow(
                id=str(r.get("id") or ""),
                label=str(r.get("label") or ""),
                z=float(r.get("z")),
                observed=str(r.get("observed") or ""),
                predicted=str(r.get("predicted") or ""),
                sigma=str(r.get("sigma") or ""),
                kind=str(r.get("kind") or "predict"),
                note=str(r.get("note") or ""),
            )
        )

    plot_scoreboard(
        zrows,
        beta=float(payload.get("beta", 1.0)),
        beta_source=str(payload.get("beta_source") or ""),
        out_png=out_png,
    )

    payload["outputs"] = {
        "scoreboard_png": str(out_png).replace("\\", "/"),
        "scoreboard_json": str(out_json).replace("\\", "/"),
    }
    _write_json(out_json, payload)

    # Work log (machine readable)
    try:
        from scripts.summary.worklog import append_event

        append_event(
            {
                "event_type": "decisive_scoreboard",
                "argv": list(sys.argv),
                "inputs": {
                    "frozen_parameters_json": frozen_path
                    or (root / "output" / "private" / "theory" / "frozen_parameters.json"),
                    "solar_light_deflection_metrics_json": root
                    / "output"
                    / "private"
                    / "theory"
                    / "solar_light_deflection_metrics.json",
                    "gravitational_redshift_experiments_json": root
                    / "output"
                    / "private"
                    / "theory"
                    / "gravitational_redshift_experiments.json",
                    "eht_shadow_compare_json": root / "output" / "private" / "eht" / "eht_shadow_compare.json",
                },
                "params": {"beta": payload.get("beta"), "beta_source": payload.get("beta_source")},
                "outputs": {"scoreboard_png": out_png, "scoreboard_json": out_json},
            }
        )
    except Exception:
        pass

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
