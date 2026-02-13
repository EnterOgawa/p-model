from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _format_num(x: float, *, digits: int = 4) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-3 or ax >= 1e5:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _max_finite(*vals: float) -> float:
    finite = [v for v in vals if math.isfinite(v)]
    return max(finite) if finite else float("nan")


def _load_frozen(root: Path, frozen_path: Path) -> Dict[str, Any]:
    if frozen_path.exists():
        return _read_json(frozen_path)
    return {"beta": 1.0, "delta": 0.0, "policy": {"beta_source": "default_beta_1"}}


def _load_kappa_error_budget(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        j = _read_json(path)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}


def _extract_eht_requirements(eht: Dict[str, Any], *, kappa_budget: Dict[str, Any]) -> Dict[str, Any]:
    p4 = eht.get("phase4") if isinstance(eht.get("phase4"), dict) else {}
    ratio = float(p4.get("shadow_diameter_coeff_ratio_p_over_gr", float("nan")))
    diff_pct = float(p4.get("shadow_diameter_coeff_diff_percent", float("nan")))
    # Idealized: if the only uncertainty were a single Gaussian error on the inferred coefficient,
    # the 3σ discrimination would require σ_rel ≲ (Δ%)/3. Realistically, M/D, κ, scattering dominate.
    rel_sigma_needed_3sigma_pct = (diff_pct / 3.0) if math.isfinite(diff_pct) else float("nan")

    rows: List[Dict[str, Any]] = []
    for r in eht.get("rows") if isinstance(eht.get("rows"), list) else []:
        if not isinstance(r, dict):
            continue
        key = str(r.get("key") or "")
        name = str(r.get("name") or key or "EHT")
        ring = float(r.get("ring_diameter_obs_uas", float("nan")))
        sigma_ring = float(r.get("ring_diameter_obs_uas_sigma", float("nan")))
        diff_uas = abs(float(r.get("shadow_diameter_diff_p_minus_gr_uas", float("nan"))))

        sigma_pred_p = float(r.get("shadow_diameter_pmodel_uas_sigma", float("nan")))
        sigma_pred_gr = float(r.get("shadow_diameter_gr_uas_sigma", float("nan")))
        shadow_gr = float(r.get("shadow_diameter_gr_uas", float("nan")))

        # Observation uncertainty: treat the published ring diameter uncertainty as σ_obs on shadow diameter under κ=1.
        sigma_obs_now = float(sigma_ring) if math.isfinite(sigma_ring) and sigma_ring >= 0 else float("nan")

        # Optional: include κ systematics estimated from the GR Kerr coefficient range.
        # Here κ is the ring/shadow ratio; Kerr spin/inclination affects the shadow diameter coefficient,
        # which maps into an effective κ range when comparing to a fixed observed ring diameter.
        # This is an order-of-magnitude budget, not a statistical measurement.
        kappa_low = float(r.get("kappa_gr_kerr_coeff_range_low", float("nan")))
        kappa_high = float(r.get("kappa_gr_kerr_coeff_range_high", float("nan")))
        kappa_sigma_assumed_kerr = float("nan")
        sigma_shadow_from_kappa = float("nan")
        sigma_obs_now_with_kappa = float("nan")
        if (
            math.isfinite(kappa_low)
            and math.isfinite(kappa_high)
            and kappa_high > kappa_low
            and math.isfinite(ring)
            and ring > 0
            and math.isfinite(sigma_ring)
            and sigma_ring >= 0
        ):
            kappa_mean = 0.5 * (kappa_low + kappa_high)
            if kappa_mean > 0:
                # Assume κ is roughly uniform within the Kerr range: σ = range / sqrt(12)
                kappa_sigma_assumed_kerr = (kappa_high - kappa_low) / math.sqrt(12.0)
                # shadow ≈ ring / κ  →  d(shadow)/dκ = -ring/κ^2
                sigma_shadow_from_kappa = abs(ring * kappa_sigma_assumed_kerr / (kappa_mean * kappa_mean))
                sigma_obs_now_with_kappa = math.sqrt((sigma_ring / kappa_mean) ** 2 + sigma_shadow_from_kappa**2)

        # Keep Kerr-budget values before potentially overriding with κ_obs.
        sigma_shadow_from_kappa_kerr = sigma_shadow_from_kappa
        sigma_obs_now_with_kappa_kerr = sigma_obs_now_with_kappa

        # Optional: κ directly estimated from published ring vs shadow (when available).
        kappa_obs = float(r.get("kappa_ring_over_shadow_obs", float("nan")))
        kappa_obs_sigma = float(r.get("kappa_ring_over_shadow_obs_sigma", float("nan")))
        sigma_shadow_from_kappa_obs = float("nan")
        sigma_obs_now_with_kappa_obs = float("nan")
        if (
            math.isfinite(kappa_obs)
            and kappa_obs > 0
            and math.isfinite(kappa_obs_sigma)
            and kappa_obs_sigma >= 0
            and math.isfinite(ring)
            and ring > 0
            and math.isfinite(sigma_ring)
            and sigma_ring >= 0
        ):
            sigma_shadow_from_kappa_obs = abs(ring * kappa_obs_sigma / (kappa_obs * kappa_obs))
            sigma_obs_now_with_kappa_obs = math.sqrt((sigma_ring / kappa_obs) ** 2 + sigma_shadow_from_kappa_obs**2)

        # Optional: κ uncertainty proxy from our κ error budget (preferred for the falsification pack).
        kappa_budget_sigma = float("nan")
        sigma_shadow_from_kappa_budget = float("nan")
        sigma_obs_now_with_kappa_budget = float("nan")
        if key == "sgra" and isinstance(kappa_budget, dict):
            rows_budget = kappa_budget.get("rows") if isinstance(kappa_budget.get("rows"), dict) else {}
            budget_row = rows_budget.get("sgra") if isinstance(rows_budget.get("sgra"), dict) else {}
            try:
                kappa_budget_sigma = float(budget_row.get("kappa_sigma_adopted_for_falsification", float("nan")))
            except Exception:
                kappa_budget_sigma = float("nan")
            if (
                math.isfinite(kappa_budget_sigma)
                and kappa_budget_sigma > 0
                and math.isfinite(ring)
                and ring > 0
                and math.isfinite(sigma_ring)
                and sigma_ring >= 0
            ):
                kappa_mean_budget = 1.0
                sigma_shadow_from_kappa_budget = abs(ring * kappa_budget_sigma / (kappa_mean_budget * kappa_mean_budget))
                sigma_obs_now_with_kappa_budget = math.sqrt((sigma_ring / kappa_mean_budget) ** 2 + sigma_shadow_from_kappa_budget**2)

        # Prefer κ budget proxy; then κ from observation; otherwise fall back to Kerr-range budget.
        kappa_systematics_source = ""
        if math.isfinite(sigma_obs_now_with_kappa_kerr):
            kappa_systematics_source = "kerr"
        if math.isfinite(sigma_obs_now_with_kappa_obs):
            kappa_systematics_source = "obs"
            sigma_shadow_from_kappa = sigma_shadow_from_kappa_obs
            sigma_obs_now_with_kappa = sigma_obs_now_with_kappa_obs
        if math.isfinite(sigma_obs_now_with_kappa_budget):
            kappa_systematics_source = "budget"
            sigma_shadow_from_kappa = sigma_shadow_from_kappa_budget
            sigma_obs_now_with_kappa = sigma_obs_now_with_kappa_budget
        if not math.isfinite(sigma_obs_now_with_kappa):
            sigma_obs_now_with_kappa = sigma_obs_now

        # Required observation error on shadow diameter (σ_obs) for 3σ separation under the model-overlap criterion.
        sigma_obs_needed_3sigma = float(r.get("shadow_diameter_sigma_obs_needed_3sigma_uas", float("nan")))

        # Scattering systematic scale (Zhu 2018): treat the mid-range distortion as an order-of-magnitude extra σ_obs term.
        refr_dmin = float(r.get("refractive_distortion_uas_min", float("nan")))
        refr_dmax = float(r.get("refractive_distortion_uas_max", float("nan")))
        sigma_scattering_mid = (
            0.5 * (refr_dmin + refr_dmax)
            if (math.isfinite(refr_dmin) and math.isfinite(refr_dmax) and refr_dmax >= refr_dmin)
            else float("nan")
        )
        sigma_obs_now_with_kappa_scattering = (
            math.sqrt(sigma_obs_now_with_kappa * sigma_obs_now_with_kappa + sigma_scattering_mid * sigma_scattering_mid)
            if (math.isfinite(sigma_obs_now_with_kappa) and math.isfinite(sigma_scattering_mid))
            else float("nan")
        )

        def _z_sep(sigma_obs: float) -> float:
            denom = float("nan")
            if all(math.isfinite(x) for x in (sigma_pred_p, sigma_pred_gr, sigma_obs)):
                denom = math.sqrt(sigma_pred_p * sigma_pred_p + sigma_pred_gr * sigma_pred_gr + 2.0 * sigma_obs * sigma_obs)
            return (diff_uas / denom) if (math.isfinite(diff_uas) and math.isfinite(denom) and denom > 0) else float("nan")

        z_sep_now = _z_sep(sigma_obs_now)
        z_sep_now_with_kappa = _z_sep(sigma_obs_now_with_kappa)
        z_sep_now_with_kappa_scattering = _z_sep(sigma_obs_now_with_kappa_scattering)

        gap = (
            (sigma_obs_now / sigma_obs_needed_3sigma)
            if (math.isfinite(sigma_obs_now) and math.isfinite(sigma_obs_needed_3sigma) and sigma_obs_needed_3sigma > 0)
            else float("nan")
        )
        gap_with_kappa = (
            (sigma_obs_now_with_kappa / sigma_obs_needed_3sigma)
            if (
                math.isfinite(sigma_obs_now_with_kappa)
                and math.isfinite(sigma_obs_needed_3sigma)
                and sigma_obs_needed_3sigma > 0
            )
            else float("nan")
        )
        gap_with_kappa_scattering = (
            (sigma_obs_now_with_kappa_scattering / sigma_obs_needed_3sigma)
            if (
                math.isfinite(sigma_obs_now_with_kappa_scattering)
                and math.isfinite(sigma_obs_needed_3sigma)
                and sigma_obs_needed_3sigma > 0
            )
            else float("nan")
        )

        theta_rel_now_pct = float(r.get("theta_unit_rel_sigma", float("nan"))) * 100.0
        theta_rel_need_pct = float(r.get("theta_unit_rel_sigma_required_3sigma", float("nan"))) * 100.0
        ring_sigma_need_if_kappa1 = float(r.get("ring_diameter_sigma_required_3sigma_uas_if_kappa1", float("nan")))
        ring_sigma_gap_if_kappa1 = float(r.get("ring_diameter_sigma_improvement_factor_to_3sigma_if_kappa1", float("nan")))
        kappa_sigma_req_if_ring_sigma_zero = float(r.get("kappa_sigma_required_3sigma_if_ring_sigma_zero", float("nan")))
        kappa_sigma_req_if_ring_sigma_current = float(
            r.get("kappa_sigma_required_3sigma_if_ring_sigma_current", float("nan"))
        )

        rows.append(
            {
                "key": key,
                "name": name,
                "diff_uas": diff_uas,
                "sigma_obs_now_uas": sigma_obs_now,
                "sigma_obs_now_with_kappa_uas": sigma_obs_now_with_kappa,
                "sigma_obs_now_with_kappa_obs_uas": sigma_obs_now_with_kappa_obs,
                "sigma_obs_now_with_kappa_kerr_uas": sigma_obs_now_with_kappa_kerr,
                "sigma_obs_now_with_kappa_budget_uas": sigma_obs_now_with_kappa_budget,
                "sigma_obs_now_with_kappa_scattering_uas": sigma_obs_now_with_kappa_scattering,
                "ring_diameter_sigma_now_uas": sigma_ring,
                "ring_diameter_sigma_required_3sigma_uas_if_kappa1": ring_sigma_need_if_kappa1,
                "ring_diameter_sigma_improvement_factor_to_3sigma_if_kappa1": ring_sigma_gap_if_kappa1,
                "kappa_gr_kerr_coeff_range_low": kappa_low,
                "kappa_gr_kerr_coeff_range_high": kappa_high,
                "kappa_obs": kappa_obs,
                "kappa_obs_sigma": kappa_obs_sigma,
                "kappa_budget_sigma": kappa_budget_sigma,
                "kappa_sigma_assumed_kerr": kappa_sigma_assumed_kerr,
                "kappa_sigma_required_3sigma_if_ring_sigma_zero": kappa_sigma_req_if_ring_sigma_zero,
                "kappa_sigma_required_3sigma_if_ring_sigma_current": kappa_sigma_req_if_ring_sigma_current,
                "sigma_shadow_from_kappa_uas": sigma_shadow_from_kappa,
                "sigma_shadow_from_kappa_obs_uas": sigma_shadow_from_kappa_obs,
                "sigma_shadow_from_kappa_kerr_uas": sigma_shadow_from_kappa_kerr,
                "sigma_shadow_from_kappa_budget_uas": sigma_shadow_from_kappa_budget,
                "kappa_systematics_source": kappa_systematics_source,
                "z_separation_now_sigma": z_sep_now,
                "z_separation_now_with_kappa_sigma": z_sep_now_with_kappa,
                "z_separation_now_with_kappa_scattering_sigma": z_sep_now_with_kappa_scattering,
                "sigma_obs_needed_3sigma_uas": sigma_obs_needed_3sigma,
                "gap_factor_now_over_needed": gap,
                "gap_factor_now_over_needed_with_kappa": gap_with_kappa,
                "gap_factor_now_over_needed_with_kappa_scattering": gap_with_kappa_scattering,
                "theta_unit_rel_sigma_now_pct": theta_rel_now_pct,
                "theta_unit_rel_sigma_needed_pct": theta_rel_need_pct,
                "source_keys": str(r.get("source_keys") or ""),
            }
        )

    return {
        "shadow_diameter_coeff_ratio_p_over_gr": ratio,
        "shadow_diameter_coeff_diff_percent": diff_pct,
        "rel_sigma_needed_3sigma_percent": rel_sigma_needed_3sigma_pct,
        "rows": rows,
        "rejection_rule": (
            "βをCassini等で固定した上で、観測から得た『影直径係数（=θ_shadow/(GM/c^2D)）』の推定値 r_obs と 1σ誤差 σ_r を用い、"
            "もし |r_obs - r_P| > 3σ_r なら P-model（β固定）は棄却、|r_obs - r_GR| > 3σ_r なら GR（Schwarzschild近似）は棄却。"
            "（EHTはリング直径→影直径の変換 κ、Kerrスピン/散乱などの系統を含むため、σ_r はそれらも含めた“総合誤差”で評価する。）"
        ),
        "notes": [
            "図の『参考：+κ/散乱』は、κの不確かさと屈折散乱スケール（Zhu 2018）を影直径へ伝播させた“誤差予算の目安”。",
            "κ は κ誤差予算（output/private/eht/eht_kappa_error_budget.json）があればそれを優先し、無ければ κ_obs（リング/シャドウ比）→Kerr係数レンジ（参考）の順で用いる。",
            "κ は放射モデル/散乱/スピンなどの影響を受けるため、厳密な評価には追加観測量（形状・非円形性・偏心など）や外部制約が必要。",
        ],
    }


def _extract_delta_constraints(delta_j: Dict[str, Any]) -> Dict[str, Any]:
    delta_adopted = float(delta_j.get("delta_adopted", float("nan")))
    gamma_max = float(delta_j.get("gamma_max_for_delta_adopted", float("nan")))
    rows_out: List[Dict[str, Any]] = []
    for r in delta_j.get("rows") if isinstance(delta_j.get("rows"), list) else []:
        if not isinstance(r, dict):
            continue
        rows_out.append(
            {
                "key": str(r.get("key") or ""),
                "label": str(r.get("label") or ""),
                "gamma_obs": float(r.get("gamma_obs", float("nan"))),
                "delta_upper_from_gamma": float(r.get("delta_upper_from_gamma", float("nan"))),
                "log10_delta_upper": float(r.get("log10_delta_upper", float("nan"))),
                "source": str(r.get("source") or ""),
            }
        )

    # What gamma would be needed to probe the adopted delta (order-of-magnitude)?
    gamma_needed = (1.0 / math.sqrt(delta_adopted)) if (delta_adopted > 0 and math.isfinite(delta_adopted)) else float("nan")

    return {
        "delta_adopted": delta_adopted,
        "gamma_max_for_delta_adopted": gamma_max,
        "gamma_needed_to_probe_delta_adopted": gamma_needed,
        "rows": rows_out,
        "rejection_rule": (
            "P-modelの速度飽和は γ_max ≈ 1/√δ を与える。もし観測で γ_obs > γ_max が確定すれば、その δ は棄却される。"
            "逆に、観測が更新されるほど δ の“上限”が下がる（δ < 1/γ_obs^2）。"
        ),
    }


def _render_figure(payload: Dict[str, Any], *, out_png: Path) -> None:
    _set_japanese_font()

    eht = payload.get("eht") if isinstance(payload.get("eht"), dict) else {}
    eht_rows = eht.get("rows") if isinstance(eht.get("rows"), list) else []
    delta = payload.get("delta") if isinstance(payload.get("delta"), dict) else {}
    delta_rows = delta.get("rows") if isinstance(delta.get("rows"), list) else []

    fig = plt.figure(figsize=(12.5, 8.2), dpi=180)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.35)

    # Panel A: EHT required precision
    ax1 = fig.add_subplot(gs[0, 0])
    names: List[str] = []
    sigma_now: List[float] = []
    sigma_now_kappa: List[float] = []
    sigma_now_kappa_scatt: List[float] = []
    sigma_need: List[float] = []
    diffs: List[float] = []
    for r in eht_rows:
        if not isinstance(r, dict):
            continue
        names.append(str(r.get("name") or r.get("key") or "EHT"))
        sigma_now.append(float(r.get("sigma_obs_now_uas", float("nan"))))
        sigma_now_kappa.append(float(r.get("sigma_obs_now_with_kappa_uas", float("nan"))))
        sigma_now_kappa_scatt.append(float(r.get("sigma_obs_now_with_kappa_scattering_uas", float("nan"))))
        sigma_need.append(float(r.get("sigma_obs_needed_3sigma_uas", float("nan"))))
        diffs.append(float(r.get("diff_uas", float("nan"))))

    x = list(range(len(names)))
    width = 0.20
    ax1.bar([i - 1.5 * width for i in x], sigma_now, width=width, label="現状 σ_obs（リング統計; κ=1）", color="#1f77b4")
    ax1.bar(
        [i - 0.5 * width for i in x],
        sigma_now_kappa,
        width=width,
        label="参考：+κ系統（κ_obs優先 / fallback=Kerr係数レンジ）",
        color="#7f7f7f",
        alpha=0.75,
    )
    ax1.bar(
        [i + 0.5 * width for i in x],
        sigma_now_kappa_scatt,
        width=width,
        label="参考：+κ+散乱（屈折distortionのmid；Zhu 2018）",
        color="#bcbd22",
        alpha=0.75,
    )
    ax1.bar([i + 1.5 * width for i in x], sigma_need, width=width, label="3σ判別に必要な σ_obs（影直径）", color="#ff7f0e")
    for i, (d, s0, sk, sks, s1) in enumerate(zip(diffs, sigma_now, sigma_now_kappa, sigma_now_kappa_scatt, sigma_need, strict=False)):
        if math.isfinite(d):
            top = _max_finite(s0, sk, sks, s1)
            if math.isfinite(top):
                ax1.text(i, top + 0.12, f"|Δ|={_format_num(d, digits=3)} μas", ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(x, names)
    ax1.set_ylabel("観測誤差（1σ）[μas]")
    ax1.set_title("EHT（ブラックホール影）：差分予測を3σで判別するための必要精度（κ/散乱は参考）")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper left")

    ratio = eht.get("shadow_diameter_coeff_ratio_p_over_gr")
    diff_pct = eht.get("shadow_diameter_coeff_diff_percent")
    if isinstance(ratio, (int, float)) and isinstance(diff_pct, (int, float)) and math.isfinite(float(ratio)) and math.isfinite(float(diff_pct)):
        ax1.text(
            0.01,
            0.02,
            f"係数差（β固定）: 係数比 P/GR={float(ratio):.4f}（差 {float(diff_pct):.2f}%）",
            transform=ax1.transAxes,
            fontsize=10,
            color="#444",
        )

    # Panel B: delta constraints (log10)
    ax2 = fig.add_subplot(gs[1, 0])
    labels: List[str] = []
    log10_upper: List[float] = []
    gamma_obs: List[float] = []
    for r in delta_rows:
        if not isinstance(r, dict):
            continue
        labels.append(str(r.get("label") or r.get("key") or "obs"))
        log10_upper.append(float(r.get("log10_delta_upper", float("nan"))))
        gamma_obs.append(float(r.get("gamma_obs", float("nan"))))

    ax2.bar(range(len(labels)), log10_upper, color="#2ca02c", alpha=0.85)
    for i, g in enumerate(gamma_obs):
        if math.isfinite(g) and g > 0:
            ax2.text(i, log10_upper[i] + 0.6, f"γ≈1e{math.log10(g):.1f}", ha="center", va="bottom", fontsize=9)
    delta_adopted = float(delta.get("delta_adopted", float("nan")))
    if math.isfinite(delta_adopted) and delta_adopted > 0:
        y = math.log10(delta_adopted)
        ax2.axhline(y, color="#d62728", linestyle="--", linewidth=1.8, label=f"採用 δ = 1e{y:.0f}")
    ax2.set_xticks(range(len(labels)), labels, rotation=0)
    ax2.set_ylabel("log10(δの上限)   ※観測が許す最大δ（小さいほど厳しい）")
    ax2.set_title("速度飽和 δ：既知の高γ観測が与える上限（δ < 1/γ^2）")
    ax2.grid(True, axis="y", alpha=0.25)
    if math.isfinite(delta_adopted) and delta_adopted > 0:
        ax2.legend(loc="upper right")

    fig.suptitle("反証条件（差分予測）パック：必要精度と棄却条件の要点", fontsize=14, y=0.98)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = _repo_root()

    ap = argparse.ArgumentParser(description="Build Phase 7.3 falsification/required-precision pack (figure + JSON).")
    ap.add_argument(
        "--frozen",
        type=str,
        default=str(root / "output" / "private" / "theory" / "frozen_parameters.json"),
        help="Path to frozen_parameters.json (beta freeze policy).",
    )
    ap.add_argument(
        "--eht",
        type=str,
        default=str(root / "output" / "private" / "eht" / "eht_shadow_compare.json"),
        help="Path to eht_shadow_compare.json (Phase 4/7 inputs).",
    )
    ap.add_argument(
        "--kappa-error-budget",
        type=str,
        default=str(root / "output" / "private" / "eht" / "eht_kappa_error_budget.json"),
        help="Optional κ error budget (output/private/eht/eht_kappa_error_budget.json).",
    )
    ap.add_argument(
        "--delta",
        type=str,
        default=str(root / "output" / "private" / "theory" / "delta_saturation_constraints.json"),
        help="Path to delta_saturation_constraints.json.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(root / "output" / "private" / "summary" / "decisive_falsification.json"),
        help="Output JSON path (fixed name recommended).",
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default=str(root / "output" / "private" / "summary" / "decisive_falsification.png"),
        help="Output PNG path (fixed name recommended).",
    )
    args = ap.parse_args()

    frozen_path = Path(args.frozen)
    eht_path = Path(args.eht)
    kappa_budget_path = Path(args.kappa_error_budget)
    delta_path = Path(args.delta)
    out_json = Path(args.out_json)
    out_png = Path(args.out_png)

    frozen = _load_frozen(root, frozen_path)
    eht = _read_json(eht_path) if eht_path.exists() else {}
    kappa_budget = _load_kappa_error_budget(kappa_budget_path)
    delta_j = _read_json(delta_path) if delta_path.exists() else {}

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "inputs": {
            "frozen_parameters": str(frozen_path),
            "eht_shadow_compare": str(eht_path),
            "eht_kappa_error_budget": str(kappa_budget_path) if kappa_budget_path.exists() else None,
            "delta_saturation_constraints": str(delta_path),
        },
        "policy": {
            "beta": float(frozen.get("beta", 1.0)),
            "beta_source": str((frozen.get("policy") or {}).get("beta_source") or ""),
            "delta": float((delta_j.get("delta_adopted") if isinstance(delta_j, dict) else float("nan"))),
        },
        "eht": _extract_eht_requirements(eht if isinstance(eht, dict) else {}, kappa_budget=kappa_budget),
        "delta": _extract_delta_constraints(delta_j if isinstance(delta_j, dict) else {}),
        "outputs": {"png": str(out_png), "json": str(out_json)},
    }

    _render_figure(payload, out_png=out_png)
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "event_type": "decisive_falsification",
                "argv": sys.argv,
                "inputs": {
                    "frozen_parameters": frozen_path,
                    "eht_shadow_compare": eht_path,
                    "delta_saturation_constraints": delta_path,
                },
                "outputs": {"png": out_png, "json": out_json},
                "metrics": {
                    "eht_coeff_ratio_p_over_gr": payload.get("eht", {}).get("shadow_diameter_coeff_ratio_p_over_gr"),
                    "eht_diff_percent": payload.get("eht", {}).get("shadow_diameter_coeff_diff_percent"),
                    "delta_adopted": payload.get("delta", {}).get("delta_adopted"),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
