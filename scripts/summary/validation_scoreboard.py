from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

_C_M_PER_S = 299_792_458.0


def _repo_root() -> Path:
    return _ROOT


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


def _fmt_float(x: Optional[float], *, digits: int = 4) -> str:
    if x is None:
        return ""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_pct(x: Optional[float], *, digits: int = 2) -> str:
    if x is None:
        return ""
    return f"{x * 100.0:.{digits}f}".rstrip("0").rstrip(".") + "%"


def _status_from_abs_sigma(abs_sigma: Optional[float]) -> str:
    if abs_sigma is None:
        return "info"
    if abs_sigma <= 1.0:
        return "ok"
    if abs_sigma <= 2.0:
        return "mixed"
    return "ng"


def _status_color(status: str) -> str:
    if status == "ok":
        return "#2ca02c"
    if status == "mixed":
        # Yellow (needs further verification)
        return "#f1c232"
    if status == "ng":
        return "#d62728"
    return "#7f7f7f"


def _status_label(status: str) -> str:
    return {"ok": "OK", "mixed": "要改善", "ng": "不一致", "info": "参考"}.get(status, status)


def _status_rates(counts: Optional[Dict[str, int]]) -> Optional[Dict[str, Any]]:
    if not counts:
        return None
    keys = ("ok", "mixed", "ng", "info")
    total = sum(int(counts.get(k, 0) or 0) for k in keys)
    if total <= 0:
        return None
    rates = {k: (int(counts.get(k, 0) or 0) / float(total)) for k in keys}
    ok = int(counts.get("ok", 0) or 0)
    mixed = int(counts.get("mixed", 0) or 0)
    return {
        "total": total,
        "counts": {k: int(counts.get(k, 0) or 0) for k in keys},
        "rates": rates,
        "ok_rate": rates["ok"],
        "ok_or_mixed_rate": rates["ok"] + rates["mixed"],
        "weighted_rate_ok_plus_half_mixed": (ok + 0.5 * mixed) / float(total),
    }


@dataclass(frozen=True)
class ScoreRow:
    id: str
    label: str
    status: str
    score: Optional[float]
    metric: str
    detail: str
    sources: List[str]
    score_kind: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "status_label": _status_label(self.status),
            "score": self.score,
            "score_kind": self.score_kind,
            "metric": self.metric,
            "detail": self.detail,
            "sources": list(self.sources),
        }


def _maybe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _score_lower_better(value: Optional[float], *, ok_max: float, mixed_max: float) -> Optional[float]:
    """Map a 'lower is better' metric to a z-like score axis.

    Score axis meaning:
      - 0: ideal
      - 1: OK threshold (green)
      - 2: mixed threshold (yellow)
      - >2: NG (red)
    """
    if value is None:
        return None
    if ok_max <= 0 or mixed_max <= ok_max:
        return None
    x = float(value)
    if x <= ok_max:
        return x / ok_max
    if x <= mixed_max:
        return 1.0 + (x - ok_max) / (mixed_max - ok_max)
    # extend linearly beyond mixed threshold (scale by mixed_max for readability)
    return 2.0 + (x - mixed_max) / max(1e-12, mixed_max)


def _score_higher_better(value: Optional[float], *, ok_min: float, mixed_min: float, ideal: float = 1.0) -> Optional[float]:
    """Map a 'higher is better' metric (e.g., corr, R^2) to a z-like score axis."""
    if value is None:
        return None
    if not (ideal > ok_min > mixed_min):
        return None
    x = float(value)
    if x >= ok_min:
        # ideal -> 0, ok_min -> 1
        return (ideal - x) / (ideal - ok_min)
    if x >= mixed_min:
        # ok_min -> 1, mixed_min -> 2
        return 1.0 + (ok_min - x) / (ok_min - mixed_min)
    # below mixed threshold
    return 2.0 + (mixed_min - x) / max(1e-12, mixed_min)


def _load_llr_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "llr" / "batch" / "llr_batch_summary.json"
    if not path.exists():
        return None
    j = _read_json(path)
    median = j.get("median_rms_ns") or {}
    best = _maybe_float(median.get("station_reflector_tropo_tide"))
    if best is None:
        return None
    range_m = (_C_M_PER_S * (best * 1e-9)) / 2.0
    # LLR uses meters as an absolute scale; treat <=1m as OK, <=2m as mixed.
    status = _status_from_abs_sigma(range_m)
    if range_m is not None:
        if range_m <= 1.0:
            status = "ok"
        elif range_m <= 2.0:
            status = "mixed"
        else:
            status = "ng"
    score = _score_lower_better(range_m, ok_max=1.0, mixed_max=2.0)
    return ScoreRow(
        id="llr",
        label="LLR（月レーザー測距）",
        status=status,
        score=score,
        metric=f"残差RMS（典型）≈{_fmt_float(range_m, digits=3)} m",
        detail="EDC CRD Normal Point（station×reflector, SR+Tropo+Tide+Ocean）",
        sources=[str(path).replace("\\", "/")],
        score_kind="llr_rms_m",
    )


def _load_llr_nglr1_row(root: Path) -> Optional[ScoreRow]:
    metrics_path = root / "output" / "llr" / "batch" / "llr_batch_metrics.csv"
    if not metrics_path.exists():
        return None

    coverage_path = root / "output" / "llr" / "batch" / "llr_data_coverage.csv"

    import csv
    import statistics

    rms_list: List[float] = []
    n_total = 0
    stations: set[str] = set()

    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("target") or "").strip().lower() != "nglr1":
                continue
            stations.add((r.get("station") or "").strip())
            try:
                n_total += int(float(r.get("n") or 0))
            except Exception:
                pass
            v = _maybe_float(r.get("rms_sr_tropo_tide_ns"))
            if v is not None:
                rms_list.append(v)

    if not rms_list:
        return None

    try:
        rms_med = float(statistics.median(rms_list))
    except Exception:
        return None

    range_m = (_C_M_PER_S * (rms_med * 1e-9)) / 2.0
    if range_m <= 1.0:
        status = "ok"
    elif range_m <= 2.0:
        status = "mixed"
    else:
        status = "ng"

    score = _score_lower_better(range_m, ok_max=1.0, mixed_max=2.0)

    stations_s = ",".join(sorted([s for s in stations if s])) or "-"
    extra_cov: List[str] = []
    if coverage_path.exists():
        try:
            with open(coverage_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if (r.get("target") or "").strip().lower() != "nglr1":
                        continue
                    st = (r.get("station") or "").strip()
                    if not st or st in stations:
                        continue
                    try:
                        n_unique = int(float(r.get("n_unique") or 0))
                    except Exception:
                        n_unique = 0
                    try:
                        min_req = int(float(r.get("min_points_required") or 0))
                    except Exception:
                        min_req = 0
                    if n_unique > 0:
                        extra_cov.append(f"{st}: n={n_unique} (<{min_req})")
        except Exception:
            extra_cov = []

    detail = f"EDC CRD Normal Point（target=nglr1; stations={stations_s}; SR+Tropo+Tide+Ocean）"
    if extra_cov:
        detail += f"（除外: {', '.join(extra_cov)}）"
    metric = f"残差RMS（典型）≈{_fmt_float(range_m, digits=3)} m"
    if n_total > 0:
        metric += f"（N={n_total}）"

    return ScoreRow(
        id="llr_nglr1",
        label="LLR（月レーザー測距：NGLR-1）",
        status=status,
        score=score,
        metric=metric,
        detail=detail,
        sources=[
            str(metrics_path).replace("\\", "/"),
            *([str(coverage_path).replace("\\", "/")] if coverage_path.exists() else []),
        ],
        score_kind="llr_rms_m",
    )


def _load_cassini_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "cassini" / "cassini_fig2_metrics.csv"
    if not path.exists():
        return None

    import csv

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in dict(r).items()})
    if not rows:
        return None

    pick = None
    for r in rows:
        if (r.get("window") or "").strip() == "-10 to +10 days":
            pick = r
            break
    if pick is None:
        pick = rows[0]

    rmse = _maybe_float(pick.get("rmse"))
    corr = _maybe_float(pick.get("corr"))

    status = "info"
    if corr is not None:
        if corr >= 0.95:
            status = "ok"
        elif corr >= 0.90:
            status = "mixed"
        else:
            status = "ng"
    score = _score_higher_better(corr, ok_min=0.95, mixed_min=0.90, ideal=1.0)

    metric = f"corr={_fmt_float(corr, digits=3)}"
    if rmse is not None:
        metric += f", RMSE={rmse:.2e}"

    return ScoreRow(
        id="cassini",
        label="Cassini（太陽会合）",
        status=status,
        score=score,
        metric=metric,
        detail="ドップラー y(t) の形状一致（±10日窓）",
        sources=[str(path).replace("\\", "/")],
        score_kind="cassini_corr",
    )


def _load_viking_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "viking" / "viking_shapiro_result.csv"
    if not path.exists():
        return None

    import csv

    max_us: Optional[float] = None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            us = _maybe_float(r.get("shapiro_delay_us"))
            if us is None:
                continue
            if max_us is None or us > max_us:
                max_us = us

    if max_us is None:
        return None

    # This is a sanity check range (non-decisive, literature typical range).
    status = "info"
    if 200.0 <= max_us <= 250.0:
        status = "ok"
    elif 150.0 <= max_us <= 300.0:
        status = "mixed"
    else:
        status = "ng"
    score = {"ok": 0.5, "mixed": 1.5, "ng": 3.0}.get(status)

    return ScoreRow(
        id="viking",
        label="Viking（太陽会合）",
        status=status,
        score=score,
        metric=f"Shapiroピーク≈{_fmt_float(max_us, digits=2)} μs",
        detail="往復Shapiro遅延（最大値の目安）",
        sources=[str(path).replace("\\", "/")],
        score_kind="viking_lit_range",
    )


def _load_mercury_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "mercury" / "mercury_precession_metrics.json"
    if not path.exists():
        return None
    j = _read_json(path)
    pmodel = _maybe_float(j.get("pmodel_precession_arcsec_per_century"))
    reference = _maybe_float(j.get("reference_residual_arcsec_per_century"))
    if pmodel is None or reference is None or reference == 0:
        return None
    diff = pmodel - reference
    rel = diff / reference
    abs_rel = abs(rel)
    abs_pct = abs_rel * 100.0

    status = "info"
    if abs_rel <= 0.001:
        status = "ok"
    elif abs_rel <= 0.01:
        status = "mixed"
    else:
        status = "ng"
    score = _score_lower_better(abs_pct, ok_max=0.1, mixed_max=1.0)

    return ScoreRow(
        id="mercury",
        label="Mercury（近日点移動）",
        status=status,
        score=score,
        metric=f"差≈{_fmt_float(diff, digits=5)} ″/世紀（{_fmt_pct(abs_rel, digits=2)}）",
        detail="実Cでの近日点移動（角秒/世紀）",
        sources=[str(path).replace("\\", "/")],
        score_kind="mercury_abs_percent",
    )


def _load_gps_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "gps" / "gps_compare_metrics.json"
    if not path.exists():
        return None
    j = _read_json(path)
    med = j.get("median_rms_ns") or {}
    brdc = _maybe_float(med.get("brdc"))
    pmodel = _maybe_float(med.get("pmodel"))
    if brdc is None or pmodel is None or brdc <= 0:
        return None
    ratio = pmodel / brdc
    diff = ratio - 1.0

    # Here "OK" means P-model RMS <= BRDC; otherwise treat as improvement-needed.
    status = "info"
    if ratio <= 1.0:
        status = "ok"
    elif ratio <= 1.10:
        status = "mixed"
    else:
        status = "ng"
    score = _score_lower_better(ratio, ok_max=1.0, mixed_max=1.10)

    return ScoreRow(
        id="gps",
        label="GPS（衛星時計）",
        status=status,
        score=score,
        metric=f"中央値RMS: P/BRDC={_fmt_float(ratio, digits=3)}（+{_fmt_pct(diff, digits=1)}）",
        detail="観測（準実測）= IGS Final CLK/SP3（RMSが小さいほど良い）",
        sources=[str(path).replace("\\", "/")],
        score_kind="gps_rms_ratio",
    )


def _load_solar_deflection_row(root: Path) -> Optional[ScoreRow]:
    metrics_path = root / "output" / "theory" / "solar_light_deflection_metrics.json"
    frozen_path = root / "output" / "theory" / "frozen_parameters.json"
    if not metrics_path.exists() or not frozen_path.exists():
        return None
    m = (_read_json(metrics_path).get("metrics") or {})
    obs_gamma = _maybe_float(m.get("observed_gamma_best"))
    obs_sigma = _maybe_float(m.get("observed_gamma_best_sigma"))
    if obs_gamma is None or obs_sigma is None or obs_sigma <= 0:
        return None
    beta = _maybe_float(_read_json(frozen_path).get("beta"))
    if beta is None:
        beta = 1.0
    gamma_pred = 2.0 * beta - 1.0
    z = (gamma_pred - obs_gamma) / obs_sigma
    abs_z = abs(z)
    return ScoreRow(
        id="solar_deflection",
        label="光偏向（太陽）",
        status=_status_from_abs_sigma(abs_z),
        score=abs_z,
        metric=f"|z|={_fmt_float(abs_z, digits=3)}（PPN γ）",
        detail=f"観測γ={_fmt_float(obs_gamma, digits=8)}±{_fmt_float(obs_sigma, digits=3)} vs 予測γ={_fmt_float(gamma_pred, digits=8)}",
        sources=[str(metrics_path).replace("\\", "/"), str(frozen_path).replace("\\", "/")],
        score_kind="abs_z",
    )


def _load_redshift_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "theory" / "gravitational_redshift_experiments.json"
    if not path.exists():
        return None
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []

    zs: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        eps = _maybe_float(r.get("epsilon"))
        sig = _maybe_float(r.get("sigma"))
        if eps is None or sig is None or sig <= 0:
            continue
        z = (0.0 - eps) / sig
        zs.append(abs(z))
    if not zs:
        return None
    worst = max(zs)
    return ScoreRow(
        id="redshift",
        label="重力赤方偏移",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=3)}（ε=0）",
        detail="複数実験の最大|z|（P-modelの弱場一次はGRと同じ ε=0）",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


def _load_cosmology_distance_duality_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json"
    if not path.exists():
        return None
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    if not rows:
        return None
    best = None
    best_sig = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        sig = _maybe_float(r.get("epsilon0_sigma"))
        if sig is None or sig <= 0:
            continue
        if best is None or best_sig is None or sig < best_sig:
            best = r
            best_sig = sig
    r0 = best if isinstance(best, dict) else (rows[0] if isinstance(rows[0], dict) else {})
    r0_id = str(r0.get("id") or "")
    z_pbg = _maybe_float(r0.get("z_pbg_static"))
    if z_pbg is None:
        return None
    delta_eps = _maybe_float(r0.get("epsilon0_extra_needed_to_match_obs"))
    extra_eta_z1 = _maybe_float(r0.get("extra_eta_factor_needed_z1"))
    delta_mu_z1 = _maybe_float(r0.get("delta_distance_modulus_mag_z1"))
    tau_z1 = _maybe_float(r0.get("tau_equivalent_dimming_z1"))
    abs_z = abs(z_pbg)

    metric = f"|z|={_fmt_float(abs_z, digits=2)}（Pbg静的）"

    # Optional: incorporate category-level systematic width (sigma_cat) from Step 16.5.3/16.5.4.
    # This reduces the risk of over-interpreting a single tight constraint when multiple modeling choices exist.
    z_sys = None
    sigma_cat = None
    min_no_bao_abs_z_sys = None
    min_no_bao_label_sys = None
    sys_path = root / "output" / "cosmology" / "cosmology_distance_duality_systematics_envelope_metrics.json"
    if sys_path.exists():
        try:
            sj = _read_json(sys_path)
            srows = sj.get("rows") if isinstance(sj.get("rows"), list) else []
            for sr in srows:
                if not isinstance(sr, dict):
                    continue
                if r0_id and str(sr.get("id") or "") == r0_id:
                    z_sys = _maybe_float(sr.get("abs_z_with_category_sys"))
                    sigma_cat = _maybe_float(sr.get("sigma_sys_category"))
                    break
            for sr in srows:
                if not isinstance(sr, dict):
                    continue
                if bool(sr.get("uses_bao", False)):
                    continue
                az = _maybe_float(sr.get("abs_z_with_category_sys"))
                if az is None:
                    continue
                if min_no_bao_abs_z_sys is None or az < min_no_bao_abs_z_sys:
                    min_no_bao_abs_z_sys = az
                    min_no_bao_label_sys = str(sr.get("short_label") or sr.get("id") or "")
        except Exception:
            pass
    if z_sys is not None:
        metric += f" / σ_cat込み|z|={_fmt_float(abs(z_sys), digits=2)}"
        if sigma_cat is not None:
            metric += f"（σ_cat={_fmt_float(sigma_cat, digits=3)}）"
    if delta_eps is not None and extra_eta_z1 is not None:
        metric += f" / z=1でD_L×{_fmt_float(extra_eta_z1, digits=2)}（Δε={_fmt_float(delta_eps, digits=3)}）"
        if delta_mu_z1 is not None and tau_z1 is not None:
            metric += f"（Δμ={_fmt_float(delta_mu_z1, digits=2)}mag, τ={_fmt_float(tau_z1, digits=2)}）"

    # Also report the best (least rejecting) non-BAO constraint as an intuition (DDR depends on distance-indicator assumptions).
    min_no_bao_abs_z = None
    min_no_bao_label = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        if bool(r.get("uses_bao", False)):
            continue
        z = _maybe_float(r.get("z_pbg_static"))
        if z is None:
            continue
        az = abs(z)
        if min_no_bao_abs_z is None or az < min_no_bao_abs_z:
            min_no_bao_abs_z = az
            min_no_bao_label = str(r.get("short_label") or r.get("id") or "")
    if min_no_bao_abs_z is not None:
        metric += f" / no-BAO最小abs(z)={_fmt_float(min_no_bao_abs_z, digits=2)}"
        if min_no_bao_abs_z_sys is not None:
            metric += f"→{_fmt_float(min_no_bao_abs_z_sys, digits=2)}"
        metric += f"（{min_no_bao_label}）"
        if min_no_bao_abs_z_sys is not None and min_no_bao_label_sys and min_no_bao_label_sys != min_no_bao_label:
            metric += f" / （σ_cat込み最小: {min_no_bao_label_sys}）"
    return ScoreRow(
        id="cosmo_ddr",
        label="宇宙論（距離二重性）",
        status=_status_from_abs_sigma(abs_z),
        score=abs_z,
        metric=metric,
        detail="距離二重性（DDR）は距離推定I/Fに膨張側(1+z)が埋め込まれているかに強く依存する。ここでは公表ε0に対する P-model最小（ε0=-1）との差を“前提監査の診断”として表示する（直ちに物理棄却とは解釈しない）。",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


def _load_cosmology_tolman_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "cosmology" / "cosmology_tolman_surface_brightness_constraints_metrics.json"
    if not path.exists():
        return None
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    zs: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        z_pbg = _maybe_float(r.get("z_pbg_static"))
        if z_pbg is None:
            continue
        zs.append(abs(z_pbg))
    if not zs:
        return None
    worst = max(zs)
    return ScoreRow(
        id="cosmo_tolman",
        label="宇宙論（Tolman表面輝度）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=2)}（Pbg静的）",
        detail="Tolman表面輝度の一次ソース制約（進化が系統。ここでは“静的P最小(n=2)”との差のみを示す）",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


def _load_cosmology_independent_probes_row(root: Path) -> Optional[ScoreRow]:
    p_sn = root / "output" / "cosmology" / "cosmology_sn_time_dilation_constraints_metrics.json"
    p_tz = root / "output" / "cosmology" / "cosmology_cmb_temperature_scaling_constraints_metrics.json"

    zs: List[Tuple[str, float]] = []
    sources: List[str] = []

    if p_sn.exists():
        try:
            sn = _read_json(p_sn)
            r = (sn.get("rows") or [None])[0] or {}
            z = _maybe_float(r.get("z_frw"))
            if z is not None:
                zs.append(("SN time dilation", abs(float(z))))
                sources.append(str(p_sn).replace("\\", "/"))
        except Exception:
            pass

    if p_tz.exists():
        try:
            tz = _read_json(p_tz)
            r = (tz.get("rows") or [None])[0] or {}
            z = _maybe_float(r.get("z_std"))
            if z is not None:
                zs.append(("CMB T(z)", abs(float(z))))
                sources.append(str(p_tz).replace("\\", "/"))
        except Exception:
            pass

    if not zs:
        return None

    worst = max(z for _, z in zs)
    parts = [f"{name}={_fmt_float(z, digits=2)}" for name, z in zs]
    metric = f"最大|z|={_fmt_float(worst, digits=2)}（p_t=1, p_T=1） / " + ", ".join(parts)

    return ScoreRow(
        id="cosmo_independent",
        label="宇宙論（独立プローブ）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=metric,
        detail="距離指標と独立：SNスペクトル時間伸長 / CMB温度スケーリング（背景Pの最小予測と整合）",
        sources=sources,
        score_kind="abs_z",
    )


def _load_cosmology_jwst_mast_row(root: Path) -> Optional[ScoreRow]:
    """
    JWST/MAST spectra (x1d) primary-data pipeline status.

    This is not a direct P-model validation metric yet; we track it as an "info" row so
    the reproducible entry point (cache/QC/z-candidates) is visible on the scoreboard.
    """

    manifest_all = root / "data" / "cosmology" / "mast" / "jwst_spectra" / "manifest_all.json"
    if not manifest_all.exists():
        return None

    waitlist_path = root / "output" / "cosmology" / "jwst_spectra_release_waitlist.json"

    j = _read_json(manifest_all)
    items = j.get("items") if isinstance(j.get("items"), dict) else {}
    targets = j.get("targets") if isinstance(j.get("targets"), list) else []
    n_targets = len(targets) if targets else len(items)

    qc_ok = 0
    z_ok = 0
    z_conf_ok = 0
    line_id_present = 0
    missing_local = 0
    for _, it in items.items():
        if not isinstance(it, dict):
            continue
        qc = it.get("qc")
        if isinstance(qc, dict) and bool(qc.get("ok")):
            qc_ok += 1
        z = it.get("z_estimate")
        if isinstance(z, dict) and bool(z.get("ok")):
            z_ok += 1
        if isinstance(z, dict) and str(z.get("reason") or "") == "no_local_x1d":
            missing_local += 1
        zc = it.get("z_confirmed")
        if isinstance(zc, dict) and bool(zc.get("ok")):
            z_conf_ok += 1
        lid = it.get("line_id")
        if isinstance(lid, dict) and lid.get("path"):
            line_id_present += 1

    metric = f"x1d(QC ok)={qc_ok}/{n_targets}, z候補={z_ok}/{n_targets}, z確定={z_conf_ok}/{n_targets}"

    blocked_n = None
    next_release_utc = ""
    if waitlist_path.exists():
        try:
            wl = _read_json(waitlist_path)
            summ = wl.get("summary") if isinstance(wl.get("summary"), dict) else {}
            blocked_n = int(summ.get("blocked_targets_n") or 0) if isinstance(summ, dict) else 0
            rels = []
            for b in wl.get("blocked_targets") or []:
                if not isinstance(b, dict):
                    continue
                s = str(b.get("next_release_utc") or "").strip()
                if s:
                    rels.append(s)
            next_release_utc = min(rels) if rels else ""
        except Exception:
            blocked_n = None
            next_release_utc = ""
    if blocked_n is not None and blocked_n > 0:
        metric += f"（公開待ち={blocked_n}"
        if next_release_utc:
            metric += f"; next={next_release_utc}"
        metric += "）"
    if missing_local > 0:
        metric += f"（localなし={missing_local}）"
    if line_id_present > 0:
        metric += f"（line_id={line_id_present}/{n_targets}）"

    detail = "JWST/MAST x1d（スペクトル一次データ）の取得状況と z候補抽出→手動線同定→z確定の入口"
    return ScoreRow(
        id="jwst_mast",
        label="JWST/MAST（スペクトル一次データ）",
        status="info",
        score=None,
        metric=metric,
        detail=detail,
        sources=[str(manifest_all).replace("\\", "/")] + ([str(waitlist_path).replace("\\", "/")] if waitlist_path.exists() else []),
    )


def _load_cosmology_bao_catalog_row(root: Path) -> Optional[ScoreRow]:
    """
    BAO geometry from catalog-based ξℓ (galaxy+random).

    We use the fitted ε significance (|ε|/σ_ε) from the smooth+peak peakfit outputs.

    - Phase A: screening（diag proxy cov）
    - Phase B: decisive（Ross 2016 full cov + MW multigrid recon + dist差し替え）
    - pre-recon: cross-check（Satpathy 2016 full cov）
    """
    # Phase B (decisive): MW multigrid + Ross full covariance.
    paths_b = [
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_metrics.json",
    ]

    # pre-recon cross-check: Satpathy full covariance (z-bin only).
    paths_pre = [
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__prerecon_metrics.json",
    ]

    # Phase A (screening): include combined and NGC/SGC splits (north/south) so the scoreboard reflects systematics.
    desi_peakfit_candidates = [
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_reservoir_r0to17_mix__rascalc_cov_reservoir_r0to17_mix_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_shrink0p2_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_both_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins__jk_cov_both_metrics.json",
    ]
    desi_peakfit_path = next((p for p in desi_peakfit_candidates if p.exists()), desi_peakfit_candidates[-1])

    paths_a = [
        # combined
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_cmass_combined_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_lowz_combined_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly_metrics.json",
        # eBOSS extension (Phase 4.5B.21; screening)
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_lrgpcmass_rec_combined_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_lrgpcmass_rec_north_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_lrgpcmass_rec_south_metrics.json",
        # eBOSS extension (Phase 4.5B.21.4; screening)
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_qso_combined_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_qso_north_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_qso_south_metrics.json",
        # DESI extension (Phase 4.5B.21.4.4.4; cov alternative): LRG1/LRG2 bins with sky jackknife cov.
        desi_peakfit_path,
        # north/south (NGC/SGC)
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_cmass_north_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_cmass_south_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_lowz_north_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_lowz_south_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_cmasslowztot_north_zbinonly_metrics.json",
        root / "output" / "cosmology" / "cosmology_bao_catalog_peakfit_cmasslowztot_south_zbinonly_metrics.json",
    ]

    def _collect(paths: List[Path]) -> Tuple[List[Dict[str, Any]], List[str]]:
        items: List[Dict[str, Any]] = []
        sources: List[str] = []
        for p in paths:
            if not p.exists():
                continue
            try:
                j = _read_json(p)
                rows = j.get("results") if isinstance(j.get("results"), list) else []
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    if str(r.get("dist") or "") != "pbg":
                        continue
                    sc = (r.get("screening") or {}) if isinstance(r.get("screening"), dict) else {}
                    abs_sigma = _maybe_float(sc.get("abs_sigma"))
                    if abs_sigma is None:
                        continue
                    lower_bound = bool(sc.get("abs_sigma_is_lower_bound"))
                    fit = (r.get("fit") or {}) if isinstance(r.get("fit"), dict) else {}
                    free = (fit.get("free") or {}) if isinstance(fit.get("free"), dict) else {}
                    eps = _maybe_float(free.get("eps"))
                    sig_eps = _maybe_float(sc.get("sigma_eps_1sigma"))
                    items.append(
                        {
                            "sample": str(r.get("sample") or ""),
                            "z_bin": str(r.get("z_bin") or "none"),
                            "z_eff": _maybe_float(r.get("z_eff")),
                            "abs_sigma": abs_sigma,
                            "lower_bound": lower_bound,
                            "eps": eps,
                            "sigma_eps": sig_eps,
                        }
                    )
                sources.append(str(p).replace("\\", "/"))
            except Exception:
                continue
        return items, sources

    items_b, sources_b = _collect(paths_b)
    items_pre, sources_pre = _collect(paths_pre)
    items_a, sources_a = _collect(paths_a)

    if not items_b and not items_pre and not items_a:
        return None

    def _summarize(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not items:
            return None
        worst = max(float(it["abs_sigma"]) for it in items)
        worst_lb = any((abs(float(it["abs_sigma"]) - worst) < 1e-12) and bool(it.get("lower_bound")) for it in items)
        cmass = next((it for it in items if it.get("sample") == "cmass"), None)
        lowz = next((it for it in items if it.get("sample") == "lowz"), None)
        zbin = [it for it in items if it.get("z_bin") not in ("", "none")]
        zbin_worst = max((float(it["abs_sigma"]) for it in zbin), default=None)
        zbin_worst_lb = (
            any((abs(float(it["abs_sigma"]) - float(zbin_worst)) < 1e-12) and bool(it.get("lower_bound")) for it in zbin)
            if zbin_worst is not None
            else False
        )
        worst_item = next((it for it in items if abs(float(it.get("abs_sigma") or 0.0) - worst) < 1e-12), None)
        return {
            "worst": worst,
            "worst_lb": worst_lb,
            "worst_item": worst_item,
            "cmass": cmass,
            "lowz": lowz,
            "zbin_worst": zbin_worst,
            "zbin_worst_lb": zbin_worst_lb,
        }

    sum_b = _summarize(items_b)
    sum_pre = _summarize(items_pre)
    sum_a = _summarize(items_a)

    # Score: prefer Phase B when available.
    score_sum = sum_b or sum_a
    assert score_sum is not None
    worst = float(score_sum["worst"])
    worst_lb = bool(score_sum.get("worst_lb"))

    def _fmt_sigma(v: Optional[float], *, lower_bound: bool = False) -> str:
        if v is None:
            return "?σ"
        return f"≥{_fmt_float(v, digits=2)}σ" if lower_bound else f"{_fmt_float(v, digits=2)}σ"

    def _fmt_item(it: Optional[Dict[str, Any]]) -> str:
        if not it:
            return ""
        z = _maybe_float(it.get("z_eff"))
        ztag = f"z={_fmt_float(z, digits=3)}" if z is not None else "z=?"
        return f"{ztag}:{_fmt_sigma(_maybe_float(it.get('abs_sigma')), lower_bound=bool(it.get('lower_bound')))}"

    def _fmt_eps_pm(item: Optional[Dict[str, Any]]) -> str:
        if not item:
            return ""
        eps = _maybe_float(item.get("eps"))
        sig = _maybe_float(item.get("sigma_eps"))
        if eps is None:
            return ""
        if sig is None or sig <= 0:
            return f"ε={_fmt_float(eps, digits=3)}"
        return f"ε={_fmt_float(eps, digits=3)}±{_fmt_float(sig, digits=3)}"

    def _fmt_phase(sum_: Optional[Dict[str, Any]], *, label: str) -> str:
        if not sum_:
            return ""

        parts: List[str] = []
        cmass = sum_.get("cmass")
        lowz = sum_.get("lowz")
        zbin_worst = _maybe_float(sum_.get("zbin_worst"))
        zbin_worst_lb = bool(sum_.get("zbin_worst_lb"))

        if cmass:
            parts.append(f"CMASS({_fmt_item(cmass)})")
        if lowz:
            parts.append(f"LOWZ({_fmt_item(lowz)})")
        if zbin_worst is not None:
            parts.append(f"z-bin(max={_fmt_sigma(zbin_worst, lower_bound=zbin_worst_lb)})")

        worst_item = sum_.get("worst_item") if isinstance(sum_.get("worst_item"), dict) else None
        worst_item_txt = ""
        if worst_item:
            zb = str(worst_item.get("z_bin") or "")
            z = _maybe_float(worst_item.get("z_eff"))
            ztag = f"z={_fmt_float(z, digits=3)}" if z is not None else ""
            eps_txt = _fmt_eps_pm(worst_item)
            extra = ", ".join([t for t in (zb, ztag, eps_txt) if t])
            if extra:
                worst_item_txt = f"（worst:{extra}）"

        s = f"{label}: 最大|z|={_fmt_sigma(_maybe_float(sum_.get('worst')), lower_bound=bool(sum_.get('worst_lb')))}{worst_item_txt}"
        if parts:
            s += " / " + ", ".join([p for p in parts if p])
        return s

    metric_parts: List[str] = []
    if sum_b:
        metric_parts.append(_fmt_phase(sum_b, label="PhaseB（Ross cov; MW multigrid）"))
    if sum_pre:
        metric_parts.append(_fmt_phase(sum_pre, label="pre-recon（Satpathy cov）"))
    if sum_a:
        metric_parts.append(_fmt_phase(sum_a, label="PhaseA（screening）"))

    # Cross-check (independent pipeline): P(k) multipoles peakfit (Beutler et al.; window-convolved).
    sources_pk: List[str] = []
    pk_path = root / "output" / "cosmology" / "cosmology_bao_pk_multipole_peakfit_window_metrics.json"
    if pk_path.exists():
        try:
            jk = _read_json(pk_path)
            rows_k = jk.get("results") if isinstance(jk.get("results"), list) else []
            pk_items: List[Tuple[float, int, float, float]] = []
            for r in rows_k:
                if not isinstance(r, dict):
                    continue
                zbin_k = int(_maybe_float(r.get("zbin")) or 0)
                fit_k = (r.get("fit") or {}) if isinstance(r.get("fit"), dict) else {}
                free_k = (fit_k.get("free") or {}) if isinstance(fit_k.get("free"), dict) else {}
                eps_k = _maybe_float(free_k.get("eps"))
                ci = free_k.get("eps_ci_1sigma")
                if eps_k is None or not (isinstance(ci, list) and len(ci) == 2):
                    continue
                lo = _maybe_float(ci[0])
                hi = _maybe_float(ci[1])
                if lo is None or hi is None:
                    continue
                sig_k = (float(hi) - float(lo)) / 2.0
                if not (sig_k > 0.0):
                    continue
                pk_items.append((abs(float(eps_k)) / float(sig_k), zbin_k, float(eps_k), float(sig_k)))
            if pk_items:
                abs_sigma_k, zbin_k, eps_k, sig_k = max(pk_items, key=lambda t: float(t[0]))
                metric_parts.append(
                    f"P(k)window(post; Beutler): 最大|z|={_fmt_float(abs_sigma_k, digits=2)}σ（zbin{int(zbin_k)}: ε={_fmt_float(eps_k, digits=3)}±{_fmt_float(sig_k, digits=3)}）"
                )
                sources_pk.append(str(pk_path).replace("\\", "/"))
        except Exception:
            pass
    metric = " / ".join([p for p in metric_parts if p])

    return ScoreRow(
        id="cosmo_bao_catalog",
        label="宇宙論（BAO一次情報）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=metric,
        detail="BOSS DR12v5: 銀河+random→ξℓ→smooth+peakでAP warping ε を推定（|ε|/σ_ε）。スコアは PhaseB（Ross full covariance + MW multigrid recon + dist差し替え）を優先し、pre-recon（Satpathy cov）は整合のクロスチェック、PhaseA（screening）は補助情報として併記。加えて、独立パイプラインとして P(k) multipoles（Beutler et al.; 窓関数込み）peakfit の結果も参考として併記する。",
        sources=[*sources_b, *sources_pre, *sources_a, *sources_pk],
        score_kind="abs_z",
    )


def _load_frame_dragging_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "theory" / "frame_dragging_experiments.json"
    if not path.exists():
        return None
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    zs: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        z = _maybe_float(r.get("z_score"))
        if z is None:
            continue
        zs.append(abs(z))
    if not zs:
        return None
    worst = max(zs)
    return ScoreRow(
        id="frame_dragging",
        label="回転（フレームドラッグ）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=3)}（μ=1）",
        detail="GP-B / LAGEOS の μ=|Ω_obs|/|Ω_pred|（μ=1が一致）",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


def _load_eht_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "eht" / "eht_shadow_compare.json"
    if not path.exists():
        return None
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    zs: List[Tuple[float, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        z = _maybe_float(r.get("zscore_pmodel"))
        if z is None:
            continue
        name = str(r.get("name") or r.get("key") or "")
        zs.append((abs(z), name))
    if not zs:
        return None
    worst, worst_name = max(zs, key=lambda x: x[0])
    return ScoreRow(
        id="eht",
        label="EHT（ブラックホール影）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=3)}（κ=1仮定）",
        detail=f"M87*/SgrA* のリング直径 vs シャドウ直径（κ=1近似）。worst={worst_name}",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


def _load_binary_pulsar_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json"
    if not path.exists():
        return None
    j = _read_json(path)
    metrics = j.get("metrics") if isinstance(j.get("metrics"), list) else []
    zs: List[Tuple[float, str]] = []
    for s in metrics:
        if not isinstance(s, dict):
            continue
        R = _maybe_float(s.get("R"))
        sig = _maybe_float(s.get("sigma_1"))
        if R is None or sig is None or sig <= 0:
            continue
        z = (R - 1.0) / sig
        name = str(s.get("name") or s.get("id") or "")
        zs.append((abs(z), name))
    if not zs:
        return None
    worst, worst_name = max(zs, key=lambda x: x[0])
    return ScoreRow(
        id="binary_pulsar",
        label="連星パルサー（軌道減衰）",
        status=_status_from_abs_sigma(worst),
        score=worst,
        metric=f"最大|z|={_fmt_float(worst, digits=2)}（R=1）",
        detail=f"Pdot_b(obs)/Pdot_b(P-model quad) の一致度。worst={worst_name}",
        sources=[str(path).replace("\\", "/")],
        score_kind="abs_z",
    )


def _load_gw_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "gw" / "gw_multi_event_summary_metrics.json"
    if not path.exists():
        return None
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []

    by_event: Dict[str, List[float]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        ev = str(r.get("event") or "")
        r2 = _maybe_float(r.get("r2"))
        if not ev or r2 is None:
            continue
        by_event.setdefault(ev, []).append(r2)

    if not by_event:
        return None

    max_r2_by_event: List[float] = []
    for rs in by_event.values():
        if rs:
            max_r2_by_event.append(max(rs))

    max_r2_by_event.sort()
    n_events = len(max_r2_by_event)
    median = None
    if n_events:
        mid = n_events // 2
        if (n_events % 2) == 1:
            median = max_r2_by_event[mid]
        else:
            median = 0.5 * (max_r2_by_event[mid - 1] + max_r2_by_event[mid])
    min_v = min(max_r2_by_event) if max_r2_by_event else None
    n_ge_09 = sum(1 for v in max_r2_by_event if v >= 0.9)
    n_ge_06 = sum(1 for v in max_r2_by_event if v >= 0.6)

    status = "info"
    if median is not None:
        if median >= 0.9 and (n_ge_09 / n_events) >= 0.7:
            status = "ok"
        elif median >= 0.6 and (n_ge_06 / n_events) >= 0.7:
            status = "mixed"
        else:
            status = "ng"
    score = _score_higher_better(median, ok_min=0.9, mixed_min=0.6, ideal=1.0)

    match_omit = j.get("match_omitted_by_reason") or {}
    omit_short = int(match_omit.get("match_window_too_short") or 0)

    metric_parts = []
    if median is not None:
        metric_parts.append(f"median(max R²)={_fmt_float(median, digits=3)}")
    if n_events:
        metric_parts.append(f">=0.6: {n_ge_06}/{n_events}")
    if min_v is not None:
        metric_parts.append(f"min(max R²)={_fmt_float(min_v, digits=3)}")
    if omit_short:
        metric_parts.append(f"match省略={omit_short}件")
    metric = " / ".join(metric_parts)

    return ScoreRow(
        id="gw",
        label="重力波（複数イベント）",
        status=status,
        score=score,
        metric=metric,
        detail="chirp位相（f(t)抽出→四重極チャープ則fit）をイベント×検出器で集計（R²）。",
        sources=[str(path).replace("\\", "/")],
        score_kind="gw_median_max_r2",
    )


def _load_delta_row(root: Path) -> Optional[ScoreRow]:
    path = root / "output" / "theory" / "delta_saturation_constraints.json"
    if not path.exists():
        return None
    j = _read_json(path)
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    deltas: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = _maybe_float(r.get("delta_upper_from_gamma"))
        if d is None:
            continue
        deltas.append(d)
    if not deltas:
        return None
    tightest = min(deltas)
    return ScoreRow(
        id="delta_saturation",
        label="速度飽和 δ（理論）",
        status="info",
        score=None,
        metric=f"観測上限: δ<{tightest:.2e}",
        detail="既知の高γ観測から『δが大きすぎるとγ_maxが足りない』という上限制約（参考）。",
        sources=[str(path).replace("\\", "/")],
        score_kind="delta_upper_bound",
    )


def _compute_sigma_stats_from_table1(table1_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Parse z-like values from "metric", and the pulsar sigma from "metric_public".
    z_re = re.compile(
        r"(?P<prefix>z|Z)\s*(?:\((?P<label>[^)]*)\))?\s*=\s*(?P<val>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )
    sigma_items: List[Dict[str, Any]] = []
    for r in table1_rows:
        if not isinstance(r, dict):
            continue
        metric = str(r.get("metric") or "")
        matches = list(z_re.finditer(metric))
        if not matches:
            continue
        chosen = None
        for m in matches:
            lab = (m.group("label") or "")
            if "P" in lab or "p" in lab:
                chosen = m
                break
        if chosen is None:
            chosen = matches[0]
        try:
            z = float(chosen.group("val"))
        except Exception:
            continue
        sigma_items.append(
            {
                "topic": str(r.get("topic") or ""),
                "observable": str(r.get("observable") or ""),
                "abs_z": abs(z),
                "raw": metric,
            }
        )

    # Pulsar: parse "最大 1.07σ" style metric_public.
    for r in table1_rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("topic") or "") != "連星パルサー（軌道減衰）":
            continue
        pub = str(r.get("metric_public") or "")
        m = re.search(r"最大\s*([0-9]+(?:\.[0-9]+)?)\s*σ", pub)
        if not m:
            continue
        try:
            z = float(m.group(1))
        except Exception:
            continue
        sigma_items.append(
            {
                "topic": str(r.get("topic") or ""),
                "observable": str(r.get("observable") or ""),
                "abs_z": abs(z),
                "raw": pub,
            }
        )
        break

    within_1 = [it for it in sigma_items if it.get("abs_z") is not None and float(it["abs_z"]) <= 1.0]
    within_2 = [it for it in sigma_items if it.get("abs_z") is not None and float(it["abs_z"]) <= 2.0]
    worst = None
    if sigma_items:
        worst = max(sigma_items, key=lambda x: float(x.get("abs_z") or 0.0))
    return {
        "n": len(sigma_items),
        "n_within_1sigma": len(within_1),
        "n_within_2sigma": len(within_2),
        "rate_within_1sigma": (len(within_1) / len(sigma_items)) if sigma_items else None,
        "rate_within_2sigma": (len(within_2) / len(sigma_items)) if sigma_items else None,
        "worst": worst,
    }


def _table1_status_from_abs_sigma(abs_sigma: float) -> str:
    # Keep consistent with other sigma-based rows.
    return _status_from_abs_sigma(abs_sigma)


def _classify_table1_rows(table1_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sigma_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*σ")
    corr_re = re.compile(r"corr\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    pct_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
    meter_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*m[（(]")
    r2_re = re.compile(r"R\^2\s*=\s*([0-9]+(?:\.[0-9]+)?)")

    breakdown: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {"ok": 0, "mixed": 0, "ng": 0, "info": 0}

    for r in table1_rows:
        if not isinstance(r, dict):
            continue
        topic = str(r.get("topic") or "")
        observable = str(r.get("observable") or "")
        metric_public = str(r.get("metric_public") or "").strip()
        metric = str(r.get("metric") or "").strip()

        status = "info"
        score_kind = ""
        score_value = None

        # Prefer public-friendly metric text for parsing.
        text = metric_public or metric

        # Special-case: Viking line is a coarse sanity check in literature range (not a z-score).
        if "文献代表レンジ(200" in text and "μs" in text and "内" in text:
            status = "ok"
            score_kind = "lit_range"
        else:
            m = sigma_re.search(text)
            if m:
                try:
                    abs_sigma = abs(float(m.group(1)))
                    status = _table1_status_from_abs_sigma(abs_sigma)
                    score_kind = "abs_sigma"
                    score_value = abs_sigma
                except Exception:
                    pass

        if status == "info":
            m = corr_re.search(text)
            if m:
                try:
                    corr = float(m.group(1))
                    if corr >= 0.95:
                        status = "ok"
                    elif corr >= 0.90:
                        status = "mixed"
                    else:
                        status = "ng"
                    score_kind = "corr"
                    score_value = corr
                except Exception:
                    pass

        if status == "info":
            # GW rows: use the best detector's R^2 as a coarse agreement metric.
            r2_vals: List[float] = []
            for m in r2_re.finditer(metric):
                try:
                    r2_vals.append(float(m.group(1)))
                except Exception:
                    continue
            if r2_vals:
                best = max(r2_vals)
                if best >= 0.9:
                    status = "ok"
                elif best >= 0.6:
                    status = "mixed"
                else:
                    status = "ng"
                score_kind = "gw_max_r2"
                score_value = best

        if status == "info":
            m = pct_re.search(text)
            if m:
                try:
                    abs_pct = abs(float(m.group(1)))
                    if abs_pct <= 0.1:
                        status = "ok"
                    elif abs_pct <= 1.0:
                        status = "mixed"
                    else:
                        status = "ng"
                    score_kind = "abs_percent"
                    score_value = abs_pct
                except Exception:
                    pass

        if status == "info":
            m = meter_re.search(text)
            if m:
                try:
                    meters = float(m.group(1))
                    if meters <= 1.0:
                        status = "ok"
                    elif meters <= 2.0:
                        status = "mixed"
                    else:
                        status = "ng"
                    score_kind = "meters"
                    score_value = meters
                except Exception:
                    pass

        counts[status] = counts.get(status, 0) + 1

        breakdown.append(
            {
                "topic": topic,
                "observable": observable,
                "status": status,
                "status_label": _status_label(status),
                "metric_public": metric_public,
                "metric": metric,
                "score_kind": score_kind,
                "score_value": score_value,
            }
        )

    return {"rows": breakdown, "counts": counts}


def build_validation_scoreboard(root: Path) -> Dict[str, Any]:
    rows: List[ScoreRow] = []
    for fn in [
        _load_llr_row,
        _load_llr_nglr1_row,
        _load_cassini_row,
        _load_viking_row,
        _load_mercury_row,
        _load_gps_row,
        _load_solar_deflection_row,
        _load_redshift_row,
        _load_cosmology_distance_duality_row,
        _load_cosmology_tolman_row,
        _load_cosmology_independent_probes_row,
        _load_cosmology_jwst_mast_row,
        _load_cosmology_bao_catalog_row,
        _load_frame_dragging_row,
        _load_eht_row,
        _load_binary_pulsar_row,
        _load_gw_row,
        _load_delta_row,
    ]:
        try:
            r = fn(root)
        except Exception:
            r = None
        if r:
            rows.append(r)

    table1_path = root / "output" / "summary" / "paper_table1_results.json"
    sigma_stats = None
    table1_breakdown = None
    table1_status_counts = None
    if table1_path.exists():
        try:
            j = _read_json(table1_path)
            table1 = j.get("table1") if isinstance(j.get("table1"), dict) else {}
            table_rows = table1.get("rows") if isinstance(table1.get("rows"), list) else []
            sigma_stats = _compute_sigma_stats_from_table1(table_rows)
            classified = _classify_table1_rows(table_rows)
            table1_breakdown = classified.get("rows")
            table1_status_counts = classified.get("counts")
        except Exception:
            sigma_stats = None
            table1_breakdown = None
            table1_status_counts = None

    status_counts: Dict[str, int] = {"ok": 0, "mixed": 0, "ng": 0, "info": 0}
    for r in rows:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    table1_status_summary = _status_rates(table1_status_counts)
    row_status_summary = _status_rates(status_counts)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": [r.to_dict() for r in rows],
        "status_counts": status_counts,
        "status_summary": row_status_summary,
        "sigma_stats": sigma_stats,
        "table1_breakdown": table1_breakdown,
        "table1_status_counts": table1_status_counts,
        "table1_status_summary": table1_status_summary,
        "policy": {
            "sigma_thresholds": {"ok_max": 1.0, "mixed_max": 2.0},
            "llr_rms_m_thresholds": {"ok_max": 1.0, "mixed_max": 2.0},
            "cassini_corr_thresholds": {"ok_min": 0.95, "mixed_min": 0.90},
            "mercury_rel_percent_thresholds": {"ok_max": 0.1, "mixed_max": 1.0},
            "gps_rms_ratio_thresholds": {"ok_max": 1.0, "mixed_max": 1.10},
            "gw_r2_thresholds": {"ok_min": 0.9, "mixed_min": 0.6},
            "exceptions": {
                "cosmo_ddr": "距離二重性（DDR）は『距離指標（SNIa/BAO）の前提（標準光源/標準定規/不透明度）』に強く依存する。ここでは静的背景P“最小モデル（ε0=-1）”の棄却度を示すが、静的無限空間の仮説で進める場合は距離指標の再導出が必須。",
                "cosmo_tolman": "Tolman表面輝度は銀河進化（系統）が支配的になり得るため、本スコアボードでは『差の符号/スケールの補助情報』として扱う。",
                "cosmo_bao_catalog": "BAO一次情報（銀河+random→ξℓ）は Phase A（screening）と Phase B（Ross full cov + recon + dist差し替え）で評価軸が異なる。本スコアボードは Phase B を優先し、Phase A は補助として併記する。",
                "eht": "EHTは κ（リング/シャドウ比）、散乱、Kerrスピン/傾斜などの系統が支配的。ここでの z は κ=1 近似の入口であり、厳密な判定には κ と系統誤差の一次ソース詰めが必要。",
                "gw": "GWの R^2/match は前処理（bandpass/whiten）や窓取りに依存する。短窓は match を省略し、主に chirp位相や波形の整合（定性的）を確認する。",
            },
            "notes": "OK/要改善/不一致 は“目安”。各テーマの厳密な判定は一次ソース・系統誤差・モデル仮定の確認が必要。",
        },
        "notes": [
            "これは『全検証を1枚で俯瞰する』ための要約スコアボード。詳細は Table 1 と各章の図を参照。",
            "OK/要改善/不一致 は、zスコア（|z|<=1/2）や相関・RMS等の暫定しきい値に基づく“目安”。",
            "宇宙論（距離二重性/Tolman）は『静的背景Pの最小モデル』の棄却度（系統・進化が支配的になり得る点に注意）。",
        ],
    }
    return payload


def plot_validation_scoreboard(
    payload: Dict[str, Any],
    *,
    out_png: Path,
    title: str = "総合スコアボード（全検証：緑=OK / 黄=要改善 / 赤=不一致）",
    xlabel: str = "正規化スコア（0=理想, 1=OK境界, 2=要改善境界）",
    target_fig_h_in: float = 6.0,
) -> None:
    _set_japanese_font()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    rows_raw = payload.get("rows") or []
    rows: List[Dict[str, Any]] = [r for r in rows_raw if isinstance(r, dict)]

    n = len(rows)
    if n == 0:
        return

    severity = {"ng": 3, "mixed": 2, "ok": 1, "info": 0}

    def sort_key(r: Dict[str, Any]) -> Tuple[int, float]:
        st = str(r.get("status") or "info")
        sev = int(severity.get(st, 0))
        score = r.get("score")
        try:
            s = float(score) if score is not None else 0.0
        except Exception:
            s = 0.0
        return (sev, s)

    ordered = sorted(rows, key=sort_key, reverse=True)
    labels = [str(r.get("label") or "") for r in ordered]

    scores: List[float] = []
    for r in ordered:
        s = r.get("score")
        try:
            scores.append(float(s) if s is not None else 0.0)
        except Exception:
            scores.append(0.0)

    # Clamp x-range so a single huge z does not make the rest unreadable.
    x_max = 6.0
    scores_clipped = [min(s, x_max) for s in scores]
    colors = [_status_color(str(r.get("status") or "info")) for r in ordered]

    import textwrap

    def _wrap_label(s: str, *, width: int) -> str:
        t = (s or "").strip()
        if not t:
            return ""
        # Prefer wrapping at punctuation/parentheses boundaries when possible.
        if len(t) <= width:
            return t
        return "\n".join(textwrap.wrap(t, width=width, break_long_words=False, break_on_hyphens=False))

    row_h_nominal = 0.55
    base_h = 1.8
    fig_h_ideal = row_h_nominal * float(len(ordered)) + base_h
    fig_h = max(4.2, min(fig_h_ideal, float(target_fig_h_in)))
    fig_w = 12.5

    # Split vertically into up to 3 panels (stacked), so y-labels never get clipped by neighboring axes.
    max_rows_per_panel = max(4, int((float(target_fig_h_in) - base_h) / row_h_nominal))
    n_panels = max(1, int(math.ceil(len(ordered) / float(max_rows_per_panel))))
    n_panels = min(n_panels, 3)
    rows_per_panel = int(math.ceil(len(ordered) / float(n_panels)))

    row_h_eff = (fig_h - base_h) / max(1.0, float(len(ordered)))
    if row_h_eff >= 0.40:
        font_size = 9
        label_width = 18
    elif row_h_eff >= 0.28:
        font_size = 8
        label_width = 18
    else:
        font_size = 7
        label_width = 16

    if n_panels == 1:
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        y = list(range(len(ordered)))
        ax.barh(y, scores_clipped, color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap_label(s, width=label_width) for s in labels], fontsize=font_size)
        ax.invert_yaxis()
        ax.set_xlim(0.0, x_max)
        ax.axvline(0.0, color="#333333", linewidth=1.0)
        for x in (1.0, 2.0):
            ax.axvline(x, color="#999999", linewidth=1.0, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.subplots_adjust(left=0.36, right=0.98, top=0.92, bottom=0.12)
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        return

    panels: List[List[str]] = []
    panels_scores: List[List[float]] = []
    panels_colors: List[List[str]] = []
    height_ratios: List[int] = []
    for p in range(n_panels):
        start = p * rows_per_panel
        end = min(start + rows_per_panel, len(ordered))
        if start >= end:
            break
        panels.append(labels[start:end])
        panels_scores.append(scores_clipped[start:end])
        panels_colors.append(colors[start:end])
        height_ratios.append(max(1, end - start))

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(fig_w, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    try:
        axes = list(axes)
    except TypeError:
        axes = [axes]

    for ax, sub_labels, sub_scores, sub_colors in zip(axes, panels, panels_scores, panels_colors, strict=False):
        y = list(range(len(sub_labels)))
        ax.barh(y, sub_scores, color=sub_colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap_label(s, width=label_width) for s in sub_labels], fontsize=font_size)
        ax.invert_yaxis()

        ax.set_xlim(0.0, x_max)
        ax.axvline(0.0, color="#333333", linewidth=1.0)
        for x in (1.0, 2.0):
            ax.axvline(x, color="#999999", linewidth=1.0, linestyle="--")
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(title, y=0.98)
    fig.supxlabel(xlabel)

    # Note: metric details are intentionally not embedded in the PNG
    # (they live in validation_scoreboard.json / Table 1 captions).

    fig.subplots_adjust(left=0.36, right=0.98, top=0.92, bottom=0.14, hspace=0.35)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    root = _repo_root()
    out_dir = root / "output" / "summary"
    default_json = out_dir / "validation_scoreboard.json"
    default_png = out_dir / "validation_scoreboard.png"

    ap = argparse.ArgumentParser(description="Build an 'all validations' scoreboard (overview).")
    ap.add_argument("--out-json", type=str, default=str(default_json), help="Output JSON path")
    ap.add_argument("--out-png", type=str, default=str(default_png), help="Output PNG path")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_png = Path(args.out_png)

    payload = build_validation_scoreboard(root)
    plot_validation_scoreboard(payload, out_png=out_png)

    payload["outputs"] = {
        "scoreboard_png": str(out_png).replace("\\", "/"),
        "scoreboard_json": str(out_json).replace("\\", "/"),
    }
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "event_type": "validation_scoreboard",
                "argv": list(sys.argv),
                "inputs": {
                    "paper_table1_results_json": root / "output" / "summary" / "paper_table1_results.json",
                    "llr_batch_summary_json": root / "output" / "llr" / "batch" / "llr_batch_summary.json",
                    "cassini_fig2_metrics_csv": root / "output" / "cassini" / "cassini_fig2_metrics.csv",
                    "viking_shapiro_result_csv": root / "output" / "viking" / "viking_shapiro_result.csv",
                    "mercury_precession_metrics_json": root / "output" / "mercury" / "mercury_precession_metrics.json",
                    "gps_compare_metrics_json": root / "output" / "gps" / "gps_compare_metrics.json",
                    "solar_light_deflection_metrics_json": root / "output" / "theory" / "solar_light_deflection_metrics.json",
                    "frozen_parameters_json": root / "output" / "theory" / "frozen_parameters.json",
                    "gravitational_redshift_experiments_json": root / "output" / "theory" / "gravitational_redshift_experiments.json",
                    "cosmology_distance_duality_constraints_metrics_json": root
                    / "output"
                    / "cosmology"
                    / "cosmology_distance_duality_constraints_metrics.json",
                    "cosmology_tolman_surface_brightness_constraints_metrics_json": root
                    / "output"
                    / "cosmology"
                    / "cosmology_tolman_surface_brightness_constraints_metrics.json",
                    "mast_jwst_spectra_manifest_all_json": root / "data" / "cosmology" / "mast" / "jwst_spectra" / "manifest_all.json",
                    "frame_dragging_experiments_json": root / "output" / "theory" / "frame_dragging_experiments.json",
                    "eht_shadow_compare_json": root / "output" / "eht" / "eht_shadow_compare.json",
                    "binary_pulsar_orbital_decay_metrics_json": root / "output" / "pulsar" / "binary_pulsar_orbital_decay_metrics.json",
                    "gw_multi_event_summary_metrics_json": root / "output" / "gw" / "gw_multi_event_summary_metrics.json",
                    "delta_saturation_constraints_json": root / "output" / "theory" / "delta_saturation_constraints.json",
                },
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
