#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_tables.py

Phase 8 / Step 8.2（本論文）向けに、`output/` で確定した検証結果から「論文用サマリ表」を生成する。

生成物（固定）:
  - output/private/summary/paper_table1_results.json
  - output/private/summary/paper_table1_results.csv
  - output/private/summary/paper_table1_results.md
  - output/private/summary/paper_table1_quantum_results.json
  - output/private/summary/paper_table1_quantum_results.csv
  - output/private/summary/paper_table1_quantum_results.md

方針:
  - “本文参照の固定パス” を優先する（図と同様、表も差し替え運用にする）。
  - 入力は `output/private/<topic>/` と `output/public/<topic>/` からのみ読み、計算の再実行は行わない。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog

_OUT_PUBLIC = _ROOT / "output" / "public"
_OUT_PRIVATE = _ROOT / "output" / "private"


def _repo_root() -> Path:
    return _ROOT


_C_M_PER_S = 299_792_458.0


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(x: Optional[float], *, digits: int = 6) -> str:
    if x is None:
        return ""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_sci(x: Optional[float], *, digits: int = 3) -> str:
    if x is None:
        return ""
    return f"{x:.{digits}e}"


def _fmt_pct(x: Optional[float], *, digits: int = 2) -> str:
    if x is None:
        return ""
    return f"{x * 100.0:.{digits}f}".rstrip("0").rstrip(".") + "%"


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


@dataclass(frozen=True)
class TableRow:
    topic: str
    observable: str
    data: str
    n: Optional[int]
    reference: str
    pmodel: str
    metric: str
    metric_public: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "observable": self.observable,
            "data": self.data,
            "n": self.n,
            "reference": self.reference,
            "pmodel": self.pmodel,
            "metric": self.metric,
            "metric_public": self.metric_public,
        }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[TableRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic", "observable", "data", "n", "reference", "pmodel", "metric", "metric_public"])
        for r in rows:
            w.writerow(
                [
                    r.topic,
                    r.observable,
                    r.data,
                    r.n if r.n is not None else "",
                    r.reference,
                    r.pmodel,
                    r.metric,
                    r.metric_public,
                ]
            )


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def _write_markdown(path: Path, *, title: str, rows: Sequence[TableRow], notes: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| テーマ | 観測量/指標 | データ | N | 参照 | P-model | 差/指標 |")
    lines.append("|---|---|---:|---:|---|---|---|")
    for r in rows:
        n = "" if r.n is None else str(r.n)
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(r.topic),
                    _md_escape(r.observable),
                    _md_escape(r.data),
                    n,
                    _md_escape(r.reference),
                    _md_escape(r.pmodel),
                    _md_escape(r.metric),
                ]
            )
            + " |"
        )
    if notes:
        lines.append("")
        lines.append("## 注記")
        for n in notes:
            lines.append(f"- {n}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_llr_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "llr" / "batch" / "llr_batch_summary.json"
    if not path.exists():
        return []
    j = _read_json(path)
    n = int(j.get("n_points_total") or 0) or None
    median = j.get("median_rms_ns") or {}
    best = median.get("station_reflector_tropo_tide")
    best_f = float(best) if best is not None else None
    metric_public = ""
    if best_f is not None:
        # LLR is round-trip TOF; convert ns -> range meters (c*dt/2).
        range_m = (_C_M_PER_S * (best_f * 1e-9)) / 2.0
        metric_public = f"典型的なズレ: {_fmt_float(range_m, digits=3)} m（小さいほど良い）"

    rows = [
        TableRow(
            topic="LLR（月レーザー測距）",
            observable="残差RMS（中央値, station×reflector）",
            data="EDC CRD Normal Point",
            n=n,
            reference="0（理想）",
            pmodel=f"{_fmt_float(best_f)} ns",
            metric="tropo+潮汐+海洋荷重まで含む",
            metric_public=metric_public,
        )
    ]

    # Optional: include NGLR-1 (new reflector target) if available in metrics CSV.
    metrics_path = _OUT_PRIVATE / "llr" / "batch" / "llr_batch_metrics.csv"
    if metrics_path.exists():
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
                v = _safe_float(r.get("rms_sr_tropo_tide_ns"))
                if v is not None:
                    rms_list.append(v)

        rms_med = None
        if rms_list:
            try:
                rms_med = float(statistics.median(rms_list))
            except Exception:
                rms_med = None

        if rms_med is not None:
            range_m = (_C_M_PER_S * (rms_med * 1e-9)) / 2.0
            metric_public_nglr1 = f"典型的なズレ: {_fmt_float(range_m, digits=3)} m（小さいほど良い）"
            stations_s = ",".join(sorted([s for s in stations if s])) or "-"
            rows.append(
                TableRow(
                    topic="LLR（月レーザー測距：NGLR-1）",
                    observable="残差RMS（中央値, station×NGLR-1）",
                    data="EDC CRD Normal Point",
                    n=n_total or None,
                    reference="0（理想）",
                    pmodel=f"{_fmt_float(rms_med)} ns",
                    metric=f"tropo+潮汐+海洋荷重まで含む（stations={stations_s}）",
                    metric_public=metric_public_nglr1,
                )
            )

    return rows


def _load_cassini_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "cassini" / "cassini_fig2_metrics.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    pick = None
    for r in rows:
        if r.get("window") == "-10 to +10 days":
            pick = r
            break
    if pick is None and rows:
        pick = rows[0]

    if pick is None:
        return []

    n = int(float(pick.get("n") or 0)) or None
    rmse = float(pick.get("rmse") or "nan")
    corr = float(pick.get("corr") or "nan")

    metric_public = ""
    if corr == corr:
        metric_public = f"形状の一致度: corr={_fmt_float(corr, digits=3)}（1に近いほど一致）"

    return [
        TableRow(
            topic="Cassini（太陽会合）",
            observable="ドップラー y(t)（観測 vs P-model）",
            data="PDS TDF（処理後）",
            n=n,
            reference="観測 y(t)",
            pmodel="P-model y(t)（β凍結）",
            metric=f"RMSE={_fmt_sci(rmse)} / corr={_fmt_float(corr, digits=6)}",
            metric_public=metric_public,
        )
    ]


def _load_viking_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "viking" / "viking_shapiro_result.csv"
    if not path.exists():
        return []

    max_us = None
    max_t = None
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n += 1
            us = float(r["shapiro_delay_us"])
            if max_us is None or us > max_us:
                max_us = us
                max_t = r["time_utc"]

    metric_public = ""
    if max_us is not None:
        # Rough literature peak range (public-friendly check).
        lo, hi = 200.0, 250.0
        if lo <= max_us <= hi:
            metric_public = "文献代表レンジ(200–250 μs)内（目安）"
        else:
            d = (lo - max_us) if max_us < lo else (max_us - hi)
            metric_public = f"文献代表レンジ(200–250 μs)からのずれ: {_fmt_float(d, digits=2)} μs（0が理想）"

    return [
        TableRow(
            topic="Viking（太陽会合）",
            observable="往復Shapiro遅延ピーク",
            data="HORIZONS（Earth/Mars, Sun-center ICRF）",
            n=n or None,
            reference="文献代表値: 200-250 μs",
            pmodel=f"{_fmt_float(max_us, digits=2)} μs",
            metric=f"ピーク時刻={max_t}",
            metric_public=metric_public,
        )
    ]


def _load_mercury_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "mercury" / "mercury_precession_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)
    ref = float(j.get("reference_arcsec_century"))
    p = float(j["simulation_physical"]["pmodel"]["arcsec_per_century"])
    diff = p - ref
    metric_public = ""
    if ref != 0.0:
        metric_public = (
            f"代表値との差: {_fmt_float(abs(diff), digits=6)} ″/世紀（{_fmt_pct(abs(diff) / abs(ref), digits=2)}）"
            "（小さいほど良い）"
        )
    return [
        TableRow(
            topic="Mercury（近日点移動）",
            observable="近日点移動（角秒/世紀）",
            data="数値シミュレーション（実C）",
            n=int(j["simulation_physical"].get("num_orbits") or 0) or None,
            reference=f"{_fmt_float(ref, digits=4)} ″/世紀（代表値）",
            pmodel=f"{_fmt_float(p, digits=6)} ″/世紀",
            metric=f"差={_fmt_float(diff, digits=6)} ″/世紀",
            metric_public=metric_public,
        )
    ]


def _load_gps_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "gps" / "gps_compare_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)
    m = j.get("metrics") or {}
    n_sats = int(float(m.get("n_sats") or 0)) or None
    brdc_med = float(m.get("brdc_rms_ns_median"))
    p_med = float(m.get("pmodel_rms_ns_median"))
    metric_public = ""
    if brdc_med > 0.0 and p_med >= 0.0:
        ratio = p_med / brdc_med
        if ratio < 1.0:
            metric_public = f"RMSが小さいほど良い。P-modelはBRDCより{_fmt_pct(1.0 - ratio, digits=1)}小さい（中央値）"
        elif ratio > 1.0:
            metric_public = f"RMSが小さいほど良い。P-modelはBRDCより{_fmt_pct(ratio - 1.0, digits=1)}大きい（中央値）"
        else:
            metric_public = "RMSが小さいほど良い。P-modelとBRDCは同程度（中央値）"
    return [
        TableRow(
            topic="GPS（衛星時計）",
            observable="残差RMS（中央値, 衛星ごと）",
            data="IGS Final CLK/SP3（準実測, 2025-10-01）",
            n=n_sats,
            reference=f"BRDC: {_fmt_float(brdc_med, digits=4)} ns",
            pmodel=f"P-model: {_fmt_float(p_med, digits=4)} ns",
            metric=f"P-model優位={int(float(m.get('pmodel_better_count') or 0))}/{n_sats}",
            metric_public=metric_public,
        )
    ]


def _load_solar_light_deflection_rows(root: Path) -> List[TableRow]:
    """
    Solar light deflection summarized via PPN parameter γ (VLBI etc.).

    Source: output/private/theory/solar_light_deflection_metrics.json (fixed-name artifact).
    """
    path = _OUT_PRIVATE / "theory" / "solar_light_deflection_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)
    m = j.get("metrics") or {}
    if not isinstance(m, dict):
        return []

    n_meas = None
    try:
        n_meas = int(float(m.get("measurement_count") or 0)) or None
    except Exception:
        n_meas = None

    label = str(m.get("observed_best_label") or m.get("observed_best_id") or "")
    if not label:
        label = "VLBI（代表）"

    gamma_obs = None
    gamma_sig = None
    try:
        gamma_obs = float(m.get("observed_gamma_best"))
        gamma_sig = float(m.get("observed_gamma_best_sigma"))
    except Exception:
        gamma_obs = None
        gamma_sig = None

    gamma_pred = None
    beta = None
    try:
        gamma_pred = float(m.get("gamma_pmodel"))
        beta = float(m.get("beta"))
    except Exception:
        gamma_pred = None
        beta = None

    z = None
    try:
        z = float(m.get("observed_z_score_best"))
    except Exception:
        z = None

    obs_txt = ""
    if gamma_obs is not None and gamma_sig is not None:
        obs_txt = f"γ={_fmt_float(gamma_obs, digits=6)}±{_fmt_float(gamma_sig, digits=6)}"

    pm_txt = ""
    if gamma_pred is not None:
        pm_txt = f"γ={_fmt_float(gamma_pred, digits=6)}"
        if beta is not None:
            pm_txt += f"（β={_fmt_float(beta, digits=6)}）"

    metric_parts: List[str] = []
    if z is not None:
        metric_parts.append(f"z={_fmt_float(z, digits=3)}")

    alpha_obs = m.get("observed_alpha_arcsec_limb_best")
    alpha_sig = m.get("observed_alpha_sigma_arcsec_limb_best")
    try:
        if alpha_obs is not None and alpha_sig is not None:
            metric_parts.append(
                f"太陽縁α={_fmt_float(float(alpha_obs), digits=7)}±{_fmt_float(float(alpha_sig), digits=7)} ″"
            )
    except Exception:
        pass

    metric_public = ""
    if z is not None:
        metric_public = f"観測との差: {_fmt_float(abs(float(z)), digits=2)}σ（0に近いほど一致）"

    return [
        TableRow(
            topic="光偏向（太陽）",
            observable="PPN γ（GR=1）",
            data=label,
            n=n_meas,
            reference=obs_txt or "γ（観測）",
            pmodel=pm_txt or "γ=2β-1",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_eht_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "eht" / "eht_shadow_compare.json"
    if not path.exists():
        return []
    j = _read_json(path)

    # NOTE:
    # In eht_shadow_compare.json, `shadow_diameter_diff_percent` is *P-model vs GR coefficient difference*
    # (i.e., (4eβ)/(2√27) - 1), and is constant across objects for fixed β.
    # Do not present it as "obs-model difference" without labeling, to avoid confusion.
    coeff_diff_pct = None
    try:
        coeff_diff_pct = float(((j.get("phase4") or {}).get("shadow_diameter_coeff_diff_percent")))
    except Exception:
        coeff_diff_pct = None

    out: List[TableRow] = []
    for r in j.get("rows") or []:
        name = str(r.get("name") or r.get("key") or "")
        obs = float(r["ring_diameter_obs_uas"])
        obs_sig = float(r["ring_diameter_obs_uas_sigma"])
        pm = float(r["shadow_diameter_pmodel_uas"])
        pm_sig = float(r["shadow_diameter_pmodel_uas_sigma"])
        z_p = float(r.get("zscore_pmodel") or float("nan"))
        k_fit = float(r.get("kappa_ring_over_shadow_fit_pmodel") or float("nan"))
        k_fit_sig = float(r.get("kappa_ring_over_shadow_fit_pmodel_sigma") or float("nan"))

        coeff_txt = ""
        if coeff_diff_pct is not None:
            s = _fmt_float(coeff_diff_pct, digits=3)
            if coeff_diff_pct > 0 and not s.startswith("-"):
                s = "+" + s
            coeff_txt = f"係数差(P/GR)={s}%"

        z_txt = ""
        if z_p == z_p:  # not NaN
            z_txt = f"z(P, κ=1)={_fmt_float(z_p, digits=3)}"

        k_txt = ""
        if k_fit == k_fit and k_fit_sig == k_fit_sig:
            k_txt = f"κ_fit={_fmt_float(k_fit, digits=3)}±{_fmt_float(k_fit_sig, digits=3)}"

        metric_parts = [p for p in (z_txt, k_txt, coeff_txt) if p]
        metric = " / ".join(metric_parts) if metric_parts else ""

        metric_public = "κ（リング/影比）など系統が支配的（指標は目安）"
        if z_p == z_p:
            metric_public = f"κ=1仮定の差: {_fmt_float(abs(z_p), digits=2)}σ（0に近いほど一致）"
        elif k_fit == k_fit and k_fit_sig == k_fit_sig:
            metric_public = f"κ_fitが1付近なら整合（κ_fit={_fmt_float(k_fit, digits=2)}）"

        out.append(
            TableRow(
                topic="EHT（ブラックホール影）",
                observable=f"リング直径（{name}）",
                data="EHT公表値（リング~影の近似）",
                n=1,
                reference=f"{_fmt_float(obs, digits=3)}±{_fmt_float(obs_sig, digits=3)} μas",
                pmodel=f"{_fmt_float(pm, digits=3)}±{_fmt_float(pm_sig, digits=3)} μas",
                metric=metric,
                metric_public=metric_public,
            )
        )
    return out


def _load_pulsar_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "pulsar" / "binary_pulsar_orbital_decay_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)
    metrics = j.get("metrics") or []
    if not isinstance(metrics, list):
        return []

    ref_parts: List[str] = []
    bound_parts: List[str] = []
    z_abs_list: List[float] = []
    for m in metrics:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or m.get("id") or "")
        name = name.replace("（", "(").replace("）", ")").replace("−", "-")

        r = _safe_float(m.get("R"))
        if r is None:
            continue

        sigma_note = str(m.get("sigma_note") or "")
        if "95" in sigma_note:
            err = _safe_float(m.get("sigma_95"))
            if err is None:
                s1 = _safe_float(m.get("sigma_1"))
                err = 1.959963984540054 * s1 if s1 is not None else None
            conf = "95%"
        else:
            err = _safe_float(m.get("sigma_1"))
            conf = "1σ"

        if err is not None:
            ref_parts.append(f"{name}: {r:.6f}±{_fmt_float(err, digits=6)} ({conf})")
            if conf == "1σ" and err > 0:
                z_abs_list.append(abs(r - 1.0) / err)
        else:
            ref_parts.append(f"{name}: {r:.6f}")

        upper_3s = _safe_float(m.get("non_gr_fraction_upper_3sigma"))
        if upper_3s is not None:
            bound_parts.append(f"{name}: <{_fmt_float(upper_3s * 100.0, digits=3)}%")

    reference = "; ".join(ref_parts) if ref_parts else ""
    metric = ""
    if bound_parts:
        metric = "追加放射上限（概算, 3σ）: " + "; ".join(bound_parts)

    metric_public = "Rが1に近いほど一致（四重極放射の支持が強い）"
    if z_abs_list:
        metric_public = f"R=1からのずれ: 最大 {_fmt_float(max(z_abs_list), digits=2)}σ（小さいほど良い）"
    if bound_parts:
        metric_public += " / 余分な放射はシステム依存（概算）"

    return [
        TableRow(
            topic="連星パルサー（軌道減衰）",
            observable="一致度 R = Pdot_b(obs)/Pdot_b(P-model quad)（R=1が一致）",
            data="一次ソース（Pb,e,m,Pdot_b補正）からPdot_b(quad)を再計算",
            n=len(ref_parts) if ref_parts else None,
            reference=reference,
            pmodel="R=1（四重極則, 弱場必要条件）",
            metric=metric,
            metric_public=metric_public,
        )
    ]


def _load_gw_rows(root: Path) -> List[TableRow]:
    def _load_event_pairs() -> List[Tuple[str, str]]:
        path = root / "data" / "gw" / "event_list.json"
        if not path.exists():
            return []
        try:
            obj = _read_json(path)
        except Exception:
            return []
        evs = obj.get("events")
        if not isinstance(evs, list):
            return []
        out: List[Tuple[str, str]] = []
        for e in evs:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "").strip()
            if not name:
                continue
            slug = str(e.get("slug") or name.lower()).strip() or name.lower()
            out.append((name, slug))
        return out

    events = _load_event_pairs() or [
        ("GW150914", "gw150914"),
        ("GW151226", "gw151226"),
        ("GW170104", "gw170104"),
        ("GW170817", "gw170817"),
        ("GW190425", "gw190425"),
    ]

    out: List[TableRow] = []
    for name, slug in events:
        path = _OUT_PRIVATE / "gw" / f"{slug}_chirp_phase_metrics.json"
        if not path.exists():
            continue
        j = _read_json(path)
        dets = j.get("detectors") or []
        if not isinstance(dets, list):
            continue

        r2_parts: List[str] = []
        match_parts: List[str] = []
        det_wave_parts: List[str] = []
        det_wave_override = False
        n_track_total = 0
        match_omitted = False

        params = j.get("params") or {}
        win = ""
        wwin = ""
        ww_param: Optional[Tuple[float, float]] = None
        wave_policy = ""
        wfrange = ""
        if isinstance(params, dict):
            wave_policy = str(params.get("wave_window_policy") or "")
            tw = params.get("track_window_s")
            if isinstance(tw, list) and len(tw) == 2:
                win = f"window={_fmt_float(_safe_float(tw[0]), digits=3)}..{_fmt_float(_safe_float(tw[1]), digits=3)} s"
            wf = params.get("wave_frange_hz")
            if isinstance(wf, list) and len(wf) == 2:
                flo = _safe_float(wf[0])
                fhi = _safe_float(wf[1])
                if flo is not None and fhi is not None:
                    wfrange = f"wave_f={_fmt_float(flo, digits=1)}..{_fmt_float(fhi, digits=1)} Hz"
            ww = params.get("wave_window_s")
            if wave_policy != "frange" and isinstance(ww, list) and len(ww) == 2:
                w0p = _safe_float(ww[0])
                w1p = _safe_float(ww[1])
                if w0p is not None and w1p is not None:
                    ww_param = (float(w0p), float(w1p))
                wwin = f"wave={_fmt_float(_safe_float(ww[0]), digits=3)}..{_fmt_float(_safe_float(ww[1]), digits=3)} s"

        for d in dets:
            if not isinstance(d, dict):
                continue
            det = str(d.get("detector") or "")
            n_track = int(d.get("n_track") or 0)
            n_track_total += n_track
            fit = d.get("fit") or {}
            if isinstance(fit, dict):
                r2 = _safe_float(fit.get("r2"))
            else:
                r2 = None
            if r2 is not None:
                r2_parts.append(f"{det}: R^2={_fmt_float(r2, digits=4)}")

            wf = d.get("waveform_fit") or {}
            match_txt = ""
            overlap = _safe_float(wf.get("overlap")) if isinstance(wf, dict) else None
            if overlap is not None:
                match_txt = f"match={_fmt_float(overlap, digits=3)}"
            elif isinstance(wf, dict):
                reason = str(wf.get("reason") or "").strip()
                if reason:
                    match_omitted = True
                    if reason == "match_window_too_short":
                        n_window = int(_safe_float(wf.get("n_window")) or 0)
                        match_txt = f"match=省略（短窓 n={n_window}）" if n_window else "match=省略（短窓）"
                    else:
                        match_txt = f"match=省略（{reason}）"
            if match_txt:
                match_parts.append(f"{det}: {match_txt}")
            if isinstance(wf, dict):
                w = wf.get("window_s")
                if isinstance(w, list) and len(w) == 2:
                    w0 = _safe_float(w[0])
                    w1 = _safe_float(w[1])
                    if w0 is not None and w1 is not None:
                        det_wave_parts.append(
                            f"{det}: wave={_fmt_float(w0, digits=3)}..{_fmt_float(w1, digits=3)} s"
                        )
                        if ww_param is not None:
                            if abs(w0 - ww_param[0]) > 1e-9 or abs(w1 - ww_param[1]) > 1e-9:
                                det_wave_override = True
                if bool(wf.get("window_auto_shifted")):
                    det_wave_override = True

        metric_parts = []
        if r2_parts:
            metric_parts.append(", ".join(r2_parts))
        if match_parts:
            metric_parts.append(", ".join(match_parts))
        if wfrange:
            metric_parts.append(wfrange)
        if win:
            metric_parts.append(win)
        if wave_policy == "frange" and det_wave_parts:
            metric_parts.append(", ".join(det_wave_parts))
        elif det_wave_override and det_wave_parts:
            metric_parts.append(", ".join(det_wave_parts))
        elif wwin:
            metric_parts.append(wwin)
        metric = " / ".join(metric_parts)

        metric_public = ""
        if r2_parts or match_parts:
            metric_public = "R^2 はchirp則の整合性、matchは波形テンプレートとの近さ（いずれも粗い指標）"
            if wfrange:
                metric_public += f" / match窓={wfrange.replace('wave_f=', '')}"
            if match_omitted:
                metric_public += " / 短窓などでmatchを省略する場合あり"

        out.append(
            TableRow(
                topic=f"重力波（{name}）",
                observable="chirp位相（f(t)抽出→四重極チャープ則fit）",
                data="GWOSC 公開 strain（32秒, 4 kHz, 検出器はイベントにより異なる）",
                n=n_track_total or None,
                reference="GR: t=t_c-A f^{-8/3}（Newton近似）",
                pmodel="四重極チャープ則と整合（弱場の必要条件）",
                metric=metric,
                metric_public=metric_public,
            )
        )
    return out


def _load_gw250114_rows(root: Path) -> List[TableRow]:
    area_path = _first_existing(
        [
            _OUT_PRIVATE / "gw" / "gw250114_area_theorem_test.json",
            _OUT_PUBLIC / "gw" / "gw250114_area_theorem_test.json",
            root / "output" / "gw" / "gw250114_area_theorem_test.json",
        ]
    )
    qnm_path = _first_existing(
        [
            _OUT_PRIVATE / "gw" / "gw250114_ringdown_qnm_fit.json",
            _OUT_PUBLIC / "gw" / "gw250114_ringdown_qnm_fit.json",
            root / "output" / "gw" / "gw250114_ringdown_qnm_fit.json",
        ]
    )
    imr_path = _first_existing(
        [
            _OUT_PRIVATE / "gw" / "gw250114_imr_consistency.json",
            _OUT_PUBLIC / "gw" / "gw250114_imr_consistency.json",
            root / "output" / "gw" / "gw250114_imr_consistency.json",
        ]
    )
    if area_path is None and qnm_path is None and imr_path is None:
        return []

    area_sigma = None
    first_time_ge5 = None
    n_combined = None
    if area_path is not None:
        try:
            ja = _read_json(area_path)
            summary = ja.get("summary") if isinstance(ja.get("summary"), dict) else {}
            sigma_ref = summary.get("sigma_ref") if isinstance(summary.get("sigma_ref"), dict) else {}
            area_sigma = _safe_float(sigma_ref.get("sigma_gaussian_combined"))
            first_time_ge5 = _safe_float(summary.get("first_time_sigma_ge_5_combined"))
            n_combined = int(_safe_float(summary.get("n_combined")) or 0)
        except Exception:
            area_sigma = None
            first_time_ge5 = None

    qnm_f_hz = None
    qnm_tau_s = None
    if qnm_path is not None:
        try:
            jq = _read_json(qnm_path)
            res = jq.get("results") if isinstance(jq.get("results"), dict) else {}
            combined = res.get("combined") if isinstance(res.get("combined"), dict) else {}
            med = combined.get("median") if isinstance(combined.get("median"), dict) else {}
            qnm_f_hz = _safe_float(med.get("f_hz"))
            qnm_tau_s = _safe_float(med.get("tau_s"))
        except Exception:
            qnm_f_hz = None
            qnm_tau_s = None

    z_mass = None
    z_spin = None
    p_imr = None
    if imr_path is not None:
        try:
            ji = _read_json(imr_path)
            cons = ji.get("consistency") if isinstance(ji.get("consistency"), dict) else {}
            z_mass = _safe_float(cons.get("z_final_mass_det_1d"))
            z_spin = _safe_float(cons.get("z_final_spin_1d"))
            p_imr = _safe_float(cons.get("p_value_mahalanobis2"))
        except Exception:
            z_mass = None
            z_spin = None
            p_imr = None

    metric_parts: List[str] = []
    if area_sigma is not None:
        metric_parts.append(f"面積定理(合成): {_fmt_float(area_sigma, digits=3)}σ")
    if first_time_ge5 is not None:
        metric_parts.append(f"first σ≥5 at t_ref={_fmt_float(first_time_ge5, digits=1)}M")
    if qnm_f_hz is not None:
        metric_parts.append(f"QNM(220): f={_fmt_float(qnm_f_hz, digits=3)} Hz")
    if qnm_tau_s is not None:
        metric_parts.append(f"τ={_fmt_float(qnm_tau_s, digits=6)} s")
    if z_mass is not None:
        metric_parts.append(f"IMR mass z={_fmt_float(z_mass, digits=3)}")
    if z_spin is not None:
        metric_parts.append(f"IMR spin z={_fmt_float(z_spin, digits=3)}")
    if p_imr is not None:
        metric_parts.append(f"IMR p={_fmt_float(p_imr, digits=3)}")

    metric_public_parts: List[str] = []
    if area_sigma is not None:
        metric_public_parts.append(f"面積定理有意度={_fmt_float(area_sigma, digits=2)}σ")
    if z_mass is not None or z_spin is not None:
        zvals = [abs(v) for v in (z_mass, z_spin) if v is not None]
        if zvals:
            metric_public_parts.append(f"IMR整合 max|z|={_fmt_float(max(zvals), digits=2)}")
    if p_imr is not None:
        metric_public_parts.append(f"IMR整合 p={_fmt_float(p_imr, digits=3)}")

    return [
        TableRow(
            topic="重力波（GW250114: 面積定理/QNM/IMR）",
            observable="面積定理の有意度 + ringdown QNM(220) + IMR整合",
            data="GW250114 data release（Zenodo/GWOSC）",
            n=n_combined if (n_combined is not None and n_combined > 0) else None,
            reference="面積定理（ΣA_f>A_i）とGR ringdown/IMR整合",
            pmodel="同一I/Fで area・QNM・IMR を横断整合",
            metric=" / ".join(metric_parts),
            metric_public=" / ".join(metric_public_parts),
        )
    ]


def _load_gravitational_redshift_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "theory" / "gravitational_redshift_experiments.json"
    if not path.exists():
        return []
    j = _read_json(path)
    rows = j.get("rows") or []
    out: List[TableRow] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        label = str(r.get("short_label") or r.get("id") or "")
        eps = float(r.get("epsilon") or 0.0)
        sig = float(r.get("sigma") or 0.0)
        z = r.get("z_score")
        z_txt = "" if z is None else f"{float(z):+.2f}"

        src = r.get("source") or {}
        if not isinstance(src, dict):
            src = {}
        doi = str(src.get("doi") or "")
        year = str(src.get("year") or "")
        src_txt = doi
        if year and doi:
            src_txt = f"{year}, {doi}"
        elif year and not src_txt:
            src_txt = year

        obs_txt = f"ε={_fmt_sci(eps, digits=3)}±{_fmt_sci(sig, digits=3)}"
        metric_parts = [p for p in (f"z={z_txt}" if z_txt else "", src_txt) if p]
        metric = " / ".join(metric_parts)

        metric_public = ""
        if sig > 0:
            metric_public = f"観測との差: {_fmt_float(abs(eps) / sig, digits=2)}σ（0に近いほど一致）"

        out.append(
            TableRow(
                topic="重力赤方偏移",
                observable="偏差 ε（GR=0）",
                data=label,
                n=None,
                reference=obs_txt,
                pmodel="ε=0",
                metric=metric,
                metric_public=metric_public,
            )
        )
    return out


def _load_cosmology_distance_duality_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "cosmology" / "cosmology_distance_duality_constraints_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)
    rows = j.get("rows") or []
    best = None
    best_sig = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        sig = _safe_float(r.get("epsilon0_sigma"))
        if sig is None or sig <= 0:
            continue
        if best is None or best_sig is None or sig < best_sig:
            best = r
            best_sig = sig
    if best is None:
        for r in rows:
            if isinstance(r, dict):
                best = r
                break
    if best is None or not isinstance(best, dict):
        return []

    r = best
    r_id = str(r.get("id") or "")
    label = str(r.get("short_label") or r.get("id") or "")
    eps_obs = _safe_float(r.get("epsilon0_obs"))
    sig = _safe_float(r.get("epsilon0_sigma"))
    z_frw = _safe_float(r.get("z_frw"))
    z_pbg = _safe_float(r.get("z_pbg_static"))
    p_obs = _safe_float(r.get("eta_p_exponent_obs"))
    p_sig = _safe_float(r.get("eta_p_exponent_sigma"))
    eta_p_obs_z1 = _safe_float(r.get("eta_p_obs_z1"))
    eta_p_sig_z1 = _safe_float(r.get("eta_p_sigma_approx_z1"))
    z_eta_frw = _safe_float(r.get("z_eta_frw_z1"))
    z_eta_pbg = _safe_float(r.get("z_eta_pbg_static_z1"))
    delta_eps = _safe_float(r.get("epsilon0_extra_needed_to_match_obs"))
    extra_eta_z1 = _safe_float(r.get("extra_eta_factor_needed_z1"))
    delta_mu_z1 = _safe_float(r.get("delta_distance_modulus_mag_z1"))
    tau_z1 = _safe_float(r.get("tau_equivalent_dimming_z1"))
    # Optional: category systematic width (sigma_cat) from the fixed envelope metrics (Step 16.5.3/16.5.4).
    sys_path = _OUT_PRIVATE / "cosmology" / "cosmology_distance_duality_systematics_envelope_metrics.json"
    z_pbg_sys = None
    sigma_cat = None
    min_no_bao_abs_z = None
    min_no_bao_abs_z_sys = None
    if sys_path.exists():
        try:
            sj = _read_json(sys_path)
            srows = sj.get("rows") if isinstance(sj.get("rows"), list) else []
            for sr in srows:
                if not isinstance(sr, dict):
                    continue
                if r_id and str(sr.get("id") or "") == r_id:
                    z_pbg_sys = _safe_float(sr.get("abs_z_with_category_sys"))
                    sigma_cat = _safe_float(sr.get("sigma_sys_category"))
                    break
            for sr in srows:
                if not isinstance(sr, dict):
                    continue
                if bool(sr.get("uses_bao", False)):
                    continue
                z_raw = _safe_float(sr.get("abs_z_raw"))
                z_sys = _safe_float(sr.get("abs_z_with_category_sys"))
                if z_raw is not None and (min_no_bao_abs_z is None or z_raw < min_no_bao_abs_z):
                    min_no_bao_abs_z = z_raw
                if z_sys is not None and (min_no_bao_abs_z_sys is None or z_sys < min_no_bao_abs_z_sys):
                    min_no_bao_abs_z_sys = z_sys
        except Exception:
            pass

    src = r.get("source") or {}
    if not isinstance(src, dict):
        src = {}
    doi = str(src.get("doi") or "")
    year = str(src.get("year") or "")
    src_txt = doi
    if year and doi:
        src_txt = f"{year}, {doi}"
    elif year and not src_txt:
        src_txt = year

    obs_txt = ""
    if eta_p_obs_z1 is not None and eta_p_sig_z1 is not None:
        obs_txt = f"η^(P)(z=1)={_fmt_float(eta_p_obs_z1, digits=3)}±{_fmt_float(eta_p_sig_z1, digits=3)}"
    elif eta_p_obs_z1 is not None:
        obs_txt = f"η^(P)(z=1)={_fmt_float(eta_p_obs_z1, digits=3)}"
    elif eps_obs is not None and sig is not None:
        obs_txt = f"ε0={_fmt_float(eps_obs, digits=3)}±{_fmt_float(sig, digits=3)}"
    elif eps_obs is not None:
        obs_txt = f"ε0={_fmt_float(eps_obs, digits=3)}"

    metric_parts: List[str] = []
    if z_frw is not None:
        metric_parts.append(f"Z_eps(FRW)={_fmt_float(z_frw, digits=3)}")
    if z_pbg is not None:
        metric_parts.append(f"Z_eps(Pbg_static)={_fmt_float(z_pbg, digits=2)}")
    if z_pbg_sys is not None:
        metric_parts.append(f"Z_eps_sys={_fmt_float(z_pbg_sys, digits=2)}")
        if sigma_cat is not None:
            metric_parts.append(f"σ_cat={_fmt_float(sigma_cat, digits=3)}")
    if p_obs is not None and p_sig is not None:
        metric_parts.append(f"p={_fmt_float(p_obs, digits=3)}±{_fmt_float(p_sig, digits=3)}")
    if z_eta_frw is not None:
        metric_parts.append(f"Z_eta(FRW)={_fmt_float(z_eta_frw, digits=3)}")
    if z_eta_pbg is not None:
        metric_parts.append(f"Z_eta(Pbg_static)={_fmt_float(z_eta_pbg, digits=2)}")
    if delta_eps is not None:
        metric_parts.append(f"Δε_needed={_fmt_float(delta_eps, digits=3)}")
    if min_no_bao_abs_z is not None:
        if min_no_bao_abs_z_sys is not None:
            metric_parts.append(
                f"no-BAO最小abs(z)={_fmt_float(min_no_bao_abs_z, digits=2)}→{_fmt_float(min_no_bao_abs_z_sys, digits=2)}"
            )
        else:
            metric_parts.append(f"no-BAO最小abs(z)={_fmt_float(min_no_bao_abs_z, digits=2)}")
    if src_txt:
        metric_parts.append(src_txt)

    metric_public = ""
    if z_eta_pbg is not None:
        metric_public = f"η^(P)(z=1)の差: {_fmt_float(abs(z_eta_pbg), digits=1)}σ（距離推定I/F監査を前提）"
        if z_pbg is not None:
            metric_public += f" / 公表ε0での差: {_fmt_float(abs(z_pbg), digits=1)}σ"
            if z_pbg_sys is not None:
                metric_public += f"（σ_cat込み: {_fmt_float(abs(z_pbg_sys), digits=1)}σ）"
        if delta_eps is not None and extra_eta_z1 is not None:
            metric_public += f" / 整合にはz=1でD_L×{_fmt_float(extra_eta_z1, digits=2)}（Δε={_fmt_float(delta_eps, digits=3)}）"
            if delta_mu_z1 is not None:
                metric_public += f"（Δμ={_fmt_float(delta_mu_z1, digits=2)} mag"
                if tau_z1 is not None:
                    metric_public += f", τ={_fmt_float(tau_z1, digits=2)}"
                metric_public += "）"
        if min_no_bao_abs_z is not None:
            if min_no_bao_abs_z_sys is not None:
                metric_public += (
                    f" / no-BAO最小abs(z)={_fmt_float(min_no_bao_abs_z, digits=2)}→{_fmt_float(min_no_bao_abs_z_sys, digits=2)}"
                )
            else:
                metric_public += f" / no-BAO最小abs(z)={_fmt_float(min_no_bao_abs_z, digits=2)}"
    elif z_pbg is not None:
        metric_public = f"公表ε0での差: {_fmt_float(abs(z_pbg), digits=1)}σ（距離推定I/F監査を前提）"

    return [
        TableRow(
            topic="宇宙論（距離二重性）",
            observable="η^(P)(z=1)（D_L/((1+z)D_A)）",
            data=label,
            n=None,
            reference=obs_txt or "η^(P)(z=1)（観測）",
            pmodel="η^(P)=1（P-model最小; p=0）",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_cosmology_tolman_surface_brightness_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "cosmology" / "cosmology_tolman_surface_brightness_constraints_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)
    rows = j.get("rows") or []
    out: List[TableRow] = []
    for r in rows:
        if not isinstance(r, dict):
            continue

        label = str(r.get("short_label") or r.get("id") or "")
        n_obs = _safe_float(r.get("n_obs"))
        sig = _safe_float(r.get("n_sigma"))
        z_frw = _safe_float(r.get("z_frw"))
        z_pbg = _safe_float(r.get("z_pbg_static"))
        evol_pbg = _safe_float(r.get("evolution_exponent_needed_pbg_static"))

        src = r.get("source") or {}
        if not isinstance(src, dict):
            src = {}
        doi = str(src.get("doi") or "")
        year = str(src.get("year") or "")
        src_txt = doi
        if year and doi:
            src_txt = f"{year}, {doi}"
        elif year and not src_txt:
            src_txt = year

        obs_txt = ""
        if n_obs is not None and sig is not None:
            obs_txt = f"n={_fmt_float(n_obs, digits=3)}±{_fmt_float(sig, digits=3)}"
        elif n_obs is not None:
            obs_txt = f"n={_fmt_float(n_obs, digits=3)}"

        metric_parts: List[str] = []
        if z_frw is not None:
            metric_parts.append(f"z(FRW)={_fmt_float(z_frw, digits=2)}")
        if z_pbg is not None:
            metric_parts.append(f"z(Pbg_static)={_fmt_float(z_pbg, digits=2)}")
        if evol_pbg is not None:
            metric_parts.append(f"evol_needed(Pbg)={_fmt_float(evol_pbg, digits=2)}")
        if src_txt:
            metric_parts.append(src_txt)

        metric_public = "進化（系統）が支配的（指標は目安）"
        if z_pbg is not None:
            metric_public = f"背景P（静的, n=2）との差: {_fmt_float(abs(z_pbg), digits=1)}σ（大きいほど不一致）"
            if evol_pbg is not None and evol_pbg < 0:
                metric_public += " / 整合には進化補正が逆符号"

        out.append(
            TableRow(
                topic="宇宙論（Tolman表面輝度）",
                observable="指数 n（SB ∝ (1+z)^-n）",
                data=label,
                n=None,
                reference=obs_txt or "n（観測）",
                pmodel="n=2（静的背景P最小）",
                metric=" / ".join(metric_parts),
                metric_public=metric_public,
            )
        )
    return out


def _fmt_source_short(source: Any) -> str:
    if not isinstance(source, dict):
        return ""
    year = str(source.get("year") or "").strip()
    doi = str(source.get("doi") or "").strip()
    arxiv = str(source.get("arxiv_id") or "").strip()
    if doi and year:
        return f"{year}, {doi}"
    if doi:
        return doi
    if arxiv and year:
        return f"{year}, arXiv:{arxiv}"
    if arxiv:
        return f"arXiv:{arxiv}"
    return year


def _load_cosmology_independent_probe_rows(root: Path) -> List[TableRow]:
    out: List[TableRow] = []

    # SN time dilation (spectral aging): compare to background-P minimal prediction p_t=1.
    p_sn = _OUT_PRIVATE / "cosmology" / "cosmology_sn_time_dilation_constraints_metrics.json"
    if p_sn.exists():
        try:
            j = _read_json(p_sn)
            r = (j.get("rows") or [None])[0] or {}
            label = str(r.get("short_label") or r.get("id") or "")
            pt = _safe_float(r.get("p_t_obs"))
            pts = _safe_float(r.get("p_t_sigma"))
            z = _safe_float(r.get("z_frw"))
            src_txt = _fmt_source_short(r.get("source"))

            obs_txt = ""
            if pt is not None and pts is not None:
                obs_txt = f"p_t={_fmt_float(pt, digits=3)}±{_fmt_float(pts, digits=3)}"
            elif pt is not None:
                obs_txt = f"p_t={_fmt_float(pt, digits=3)}"

            metric = " / ".join([p for p in (f"z={_fmt_float(z, digits=3)}" if z is not None else "", src_txt) if p])
            metric_public = ""
            if z is not None:
                metric_public = f"観測との差: {_fmt_float(abs(z), digits=2)}σ（0に近いほど一致）"

            out.append(
                TableRow(
                    topic="宇宙論（独立プローブ）",
                    observable="SN time dilation（p_t）",
                    data=label or "SNe Ia（スペクトル年齢）",
                    n=None,
                    reference=obs_txt or "p_t（観測）",
                    pmodel="p_t=1（背景P予測）",
                    metric=metric,
                    metric_public=metric_public,
                )
            )
        except Exception:
            pass

    # CMB temperature scaling: compare to background-P minimal prediction p_T=1 (β_T=0).
    p_tz = _OUT_PRIVATE / "cosmology" / "cosmology_cmb_temperature_scaling_constraints_metrics.json"
    if p_tz.exists():
        try:
            j = _read_json(p_tz)
            r = (j.get("rows") or [None])[0] or {}
            label = str(r.get("short_label") or r.get("id") or "")
            pt = _safe_float(r.get("p_T_obs"))
            pts = _safe_float(r.get("p_T_sigma"))
            z = _safe_float(r.get("z_std"))
            src_txt = _fmt_source_short(r.get("source"))

            obs_txt = ""
            if pt is not None and pts is not None:
                obs_txt = f"p_T={_fmt_float(pt, digits=3)}±{_fmt_float(pts, digits=3)}"
            elif pt is not None:
                obs_txt = f"p_T={_fmt_float(pt, digits=3)}"

            metric = " / ".join([p for p in (f"z={_fmt_float(z, digits=3)}" if z is not None else "", src_txt) if p])
            metric_public = ""
            if z is not None:
                metric_public = f"観測との差: {_fmt_float(abs(z), digits=2)}σ（0に近いほど一致）"

            out.append(
                TableRow(
                    topic="宇宙論（独立プローブ）",
                    observable="CMB温度スケーリング（p_T）",
                    data=label or "SZなど（CMB温度）",
                    n=None,
                    reference=obs_txt or "p_T（観測）",
                    pmodel="p_T=1（背景P予測）",
                    metric=metric,
                    metric_public=metric_public,
                )
            )
        except Exception:
            pass

    return out


def _load_cosmology_jwst_mast_rows(root: Path) -> List[TableRow]:
    """
    JWST/MAST x1d pipeline status (Phase 4 / Step 4.6).

    This is treated as a reference/screening row (not sigma-evaluable) so Table 1 can show
    the reproducible entry point and the current 'release wait' blocker.
    """
    path = _OUT_PRIVATE / "cosmology" / "jwst_spectra_integration_metrics.json"
    if not path.exists():
        return []
    try:
        j = _read_json(path)
    except Exception:
        return []
    jw = j.get("jwst_mast") if isinstance(j.get("jwst_mast"), dict) else {}
    t1 = j.get("table1") if isinstance(j.get("table1"), dict) else {}

    targets_n = int(jw.get("targets_n") or 0) if str(jw.get("targets_n") or "").strip() else 0
    qc_ok = int(jw.get("qc_ok_n") or 0) if str(jw.get("qc_ok_n") or "").strip() else 0
    z_ok = int(jw.get("z_estimate_ok_n") or 0) if str(jw.get("z_estimate_ok_n") or "").strip() else 0
    zc_ok = int(jw.get("z_confirmed_ok_n") or 0) if str(jw.get("z_confirmed_ok_n") or "").strip() else 0
    blocked = int(jw.get("blocked_targets_n") or 0) if str(jw.get("blocked_targets_n") or "").strip() else 0
    next_rel = str(jw.get("next_release_utc") or "").strip()

    status = str(t1.get("status") or "").strip() or "info_only"
    reasons = t1.get("reasons") if isinstance(t1.get("reasons"), list) else []
    reason_txt = "; ".join([str(x) for x in reasons if str(x).strip()]) if reasons else ""

    metric_parts: List[str] = []
    metric_public_parts: List[str] = []
    if targets_n > 0:
        metric_parts.append(f"targets={targets_n}")
        metric_public_parts.append(f"targets={targets_n}")
    metric_parts.append(f"x1d_qc_ok={qc_ok}/{targets_n}" if targets_n else f"x1d_qc_ok={qc_ok}")
    metric_parts.append(f"z_candidates={z_ok}/{targets_n}" if targets_n else f"z_candidates={z_ok}")
    metric_parts.append(f"z_confirmed={zc_ok}/{targets_n}" if targets_n else f"z_confirmed={zc_ok}")
    metric_public_parts.append(f"x1d_qc_ok={qc_ok}/{targets_n}" if targets_n else f"x1d_qc_ok={qc_ok}")
    metric_public_parts.append(f"z_confirmed={zc_ok}/{targets_n}" if targets_n else f"z_confirmed={zc_ok}")
    if blocked > 0:
        btxt = f"release_wait={blocked}"
        if next_rel:
            btxt += f"; next_release={next_rel}"
        metric_parts.append(btxt)
        metric_public_parts.append(btxt)
    if status and status != "adopted":
        metric_parts.append(f"Table1={status}")
        metric_public_parts.append(f"Table1={status}")
    if reason_txt:
        metric_parts.append(f"reason={reason_txt}")

    return [
        TableRow(
            topic="JWST/MAST（スペクトル一次データ）",
            observable="x1d→z（線同定；距離指標非依存）",
            data="MAST public x1d（cache→QC→z候補→z確定）",
            n=targets_n if targets_n > 0 else None,
            reference="スペクトル一次データ（線のズレ）",
            pmodel="1+z=P_em/P_obs（距離指標に依存しない）",
            metric=" / ".join([p for p in metric_parts if p]),
            metric_public=" / ".join([p for p in metric_public_parts if p]),
        )
    ]


def _load_xrism_rows(root: Path) -> List[TableRow]:
    """
    XRISM integration rows (Phase 4 / Step 4.13).

    Note:
    - These rows are currently treated as screening (not sigma-evaluable) unless explicitly adopted
      in `output/private/xrism/xrism_integration_metrics.json`.
    """
    path = _OUT_PRIVATE / "xrism" / "xrism_integration_metrics.json"
    if not path.exists():
        return []

    try:
        j = _read_json(path)
    except Exception:
        return []

    xr = j.get("xrism") if isinstance(j.get("xrism"), dict) else {}
    bh = xr.get("bh") if isinstance(xr.get("bh"), dict) else {}
    cl = xr.get("cluster") if isinstance(xr.get("cluster"), dict) else {}

    out: List[TableRow] = []

    # BH/AGN: absorption line -> v/c.
    try:
        n_total = int(bh.get("n_obsids_total") or 0)
    except Exception:
        n_total = 0
    try:
        n_det = int(bh.get("n_obsids_detected") or 0)
    except Exception:
        n_det = 0

    best = bh.get("best_detected_row") if isinstance(bh.get("best_detected_row"), dict) else {}
    beta = _safe_float(best.get("beta"))
    beta_sig = _safe_float(best.get("beta_sigma_total"))
    ratio = _safe_float(best.get("sys_over_stat"))

    bh_table1 = bh.get("table1") if isinstance(bh.get("table1"), dict) else {}
    bh_status = str(bh_table1.get("status") or "screening")
    bh_reasons = bh_table1.get("reasons") if isinstance(bh_table1.get("reasons"), list) else []
    bh_reason_txt = "; ".join([str(x) for x in bh_reasons if str(x).strip()]) if bh_reasons else ""

    bh_metric_parts: List[str] = []
    bh_metric_public_parts: List[str] = []
    if n_total > 0:
        bh_metric_parts.append(f"detected_obsids={n_det}/{n_total}")
        bh_metric_public_parts.append(f"detected_obsids={n_det}/{n_total}")
    if beta is not None:
        bh_metric_parts.append(f"|β|_best≈{_fmt_float(abs(beta), digits=3)}")
        bh_metric_public_parts.append(f"|β|_best≈{_fmt_float(abs(beta), digits=3)}")
    if beta_sig is not None:
        bh_metric_parts.append(f"σ_total≈{_fmt_float(beta_sig, digits=3)}")
    if ratio is not None:
        bh_metric_parts.append(f"sys/stat≈{_fmt_float(ratio, digits=2)}")
        bh_metric_public_parts.append(f"sys/stat≈{_fmt_float(ratio, digits=2)}")
    if bh_status and bh_status != "adopted":
        bh_metric_parts.append(f"Table1={bh_status}")
        bh_metric_public_parts.append(f"Table1={bh_status}")
    if bh_reason_txt:
        bh_metric_parts.append(f"reason={bh_reason_txt}")

    out.append(
        TableRow(
            topic="XRISM（BH/AGN）",
            observable="Fe-K吸収線→v/c（SRドップラー）",
            data="Resolve（公開FITS; PI+RMF）",
            n=n_total if n_total > 0 else None,
            reference="β=(D^2−1)/(D^2+1), D=E_obs(1+z_sys)/E_rest",
            pmodel="SRドップラー（δ上限が極小のため実質同一）",
            metric=" / ".join([p for p in bh_metric_parts if p]),
            metric_public=" / ".join([p for p in bh_metric_public_parts if p]),
        )
    )

    # Cluster: emission line centroid/width -> z_xray, σ_v.
    try:
        n_total = int(cl.get("n_obsids_total") or 0)
    except Exception:
        n_total = 0
    try:
        n_det = int(cl.get("n_obsids_detected") or 0)
    except Exception:
        n_det = 0

    cl_table1 = cl.get("table1") if isinstance(cl.get("table1"), dict) else {}
    cl_status = str(cl_table1.get("status") or "screening")
    cl_reasons = cl_table1.get("reasons") if isinstance(cl_table1.get("reasons"), list) else []
    cl_reason_txt = "; ".join([str(x) for x in cl_reasons if str(x).strip()]) if cl_reasons else ""

    # Take the first per-obsid best row as a representative (kept compact).
    per_obsid_best = cl.get("per_obsid_best") if isinstance(cl.get("per_obsid_best"), list) else []
    rep = (per_obsid_best[0] if per_obsid_best and isinstance(per_obsid_best[0], dict) else {}) or {}
    dv = _safe_float(rep.get("delta_v_kms"))
    zsig = _safe_float(rep.get("z_xray_sigma_total"))
    ratio = _safe_float(rep.get("sys_over_stat"))

    cl_metric_parts: List[str] = []
    cl_metric_public_parts: List[str] = []
    if n_total > 0:
        cl_metric_parts.append(f"detected_obsids={n_det}/{n_total}")
        cl_metric_public_parts.append(f"detected_obsids={n_det}/{n_total}")
    if dv is not None:
        cl_metric_parts.append(f"Δv(opt−X)≈{_fmt_float(dv, digits=1)} km/s")
        cl_metric_public_parts.append(f"Δv(opt−X)≈{_fmt_float(dv, digits=1)} km/s")
    if zsig is not None:
        cl_metric_parts.append(f"σ_z_total≈{_fmt_sci(zsig, digits=2)}")
    if ratio is not None:
        cl_metric_parts.append(f"sys/stat≈{_fmt_float(ratio, digits=2)}")
        cl_metric_public_parts.append(f"sys/stat≈{_fmt_float(ratio, digits=2)}")
    if cl_status and cl_status != "adopted":
        cl_metric_parts.append(f"Table1={cl_status}")
        cl_metric_public_parts.append(f"Table1={cl_status}")
    if cl_reason_txt:
        cl_metric_parts.append(f"reason={cl_reason_txt}")

    out.append(
        TableRow(
            topic="XRISM（銀河団）",
            observable="Fe-K輝線→z_xray（距離指標非依存）/ σ_v（線幅）",
            data="Resolve（公開FITS; PI+RMF）",
            n=n_total if n_total > 0 else None,
            reference="z_opt（光学; catalog）",
            pmodel="z_xray=z_opt（同一のz写像）",
            metric=" / ".join([p for p in cl_metric_parts if p]),
            metric_public=" / ".join([p for p in cl_metric_public_parts if p]),
        )
    )

    return out


def _load_cosmology_bao_primary_rows(root: Path) -> List[TableRow]:
    """
    BAO primary-statistics row for Table 1.

    Uses the Phase 4.5B (decisive) output: MW multigrid recon + Ross 2016 full covariance peakfit,
    and reports ε (AP warping) under dist=pbg (and lcdm as a reference computed in the same pipeline).
    """
    out_rows: List[TableRow] = []

    def _collect_peakfit_results(payload: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        rows = payload.get("results") if isinstance(payload.get("results"), list) else []
        for r in rows:
            if not isinstance(r, dict):
                continue
            dist = str(r.get("dist") or "")
            z_bin = str(r.get("z_bin") or "")
            if dist not in ("lcdm", "pbg") or not z_bin:
                continue
            sc = (r.get("screening") or {}) if isinstance(r.get("screening"), dict) else {}
            fit = (r.get("fit") or {}) if isinstance(r.get("fit"), dict) else {}
            free = (fit.get("free") or {}) if isinstance(fit.get("free"), dict) else {}
            out[(dist, z_bin)] = {
                "z_eff": _safe_float(r.get("z_eff")),
                "eps": _safe_float(free.get("eps")),
                "sigma_eps": _safe_float(sc.get("sigma_eps_1sigma")),
                "abs_sigma": _safe_float(sc.get("abs_sigma")),
                "abs_sigma_is_lower_bound": bool(sc.get("abs_sigma_is_lower_bound")),
            }
        return out

    def _fmt_pm(eps: Any, sig: Any) -> str:
        e = _safe_float(eps)
        s = _safe_float(sig)
        if e is None:
            return "ε=?"
        if s is None or s <= 0:
            return f"ε={_fmt_float(e, digits=3)}"
        return f"ε={_fmt_float(e, digits=3)}±{_fmt_float(s, digits=3)}"

    # Phase 4.5B decisive: BOSS DR12v5 (CMASSLOWZTOT; z-bins) with MW multigrid recon + Ross cov.
    path = _first_existing(
        [
            _OUT_PRIVATE
            / "cosmology"
            / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_metrics.json",
            _OUT_PUBLIC
            / "cosmology"
            / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_metrics.json",
            root
            / "output"
            / "cosmology"
            / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_metrics.json",
        ]
    )
    if path is not None and path.exists():
        try:
            j = _read_json(path)
        except Exception:
            j = None
        if j is not None:
            by = _collect_peakfit_results(j)
            zbins = [z for z in ("b1", "b2", "b3") if ("pbg", z) in by]
            if zbins:
                worst_zbin = max(zbins, key=lambda z: float(by[("pbg", z)].get("abs_sigma") or -1.0))
                w = by[("pbg", worst_zbin)]
                w_l = by.get(("lcdm", worst_zbin)) or {}

                z_eff = w.get("z_eff")
                eps_p = w.get("eps")
                sig_p = w.get("sigma_eps")
                abs_sigma_p = w.get("abs_sigma")
                lb_p = bool(w.get("abs_sigma_is_lower_bound"))

                eps_l = w_l.get("eps")
                sig_l = w_l.get("sigma_eps")

                metric_parts: List[str] = []
                if z_eff is not None:
                    metric_parts.append(f"z_eff={_fmt_float(z_eff, digits=3)}")
                if abs_sigma_p is not None:
                    metric_parts.append(
                        f"pbg({worst_zbin}): |z|={'≥' if lb_p else ''}{_fmt_float(abs(abs_sigma_p), digits=2)}σ"
                    )
                if eps_p is not None:
                    metric_parts.append(f"pbg({worst_zbin}): {_fmt_pm(eps_p, sig_p)}")
                if eps_l is not None:
                    metric_parts.append(f"lcdm({worst_zbin}): {_fmt_pm(eps_l, sig_l)}")
                if eps_p is not None and eps_l is not None:
                    metric_parts.append(f"Δε(pbg−lcdm)={_fmt_float(float(eps_p) - float(eps_l), digits=3)}")

                # Pre-recon cross-check (Satpathy 2016 covariance).
                prerecon_path = _first_existing(
                    [
                        _OUT_PRIVATE
                        / "cosmology"
                        / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__prerecon_metrics.json",
                        _OUT_PUBLIC
                        / "cosmology"
                        / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__prerecon_metrics.json",
                        root
                        / "output"
                        / "cosmology"
                        / "cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__prerecon_metrics.json",
                    ]
                )
                if prerecon_path is not None and prerecon_path.exists():
                    try:
                        jp = _read_json(prerecon_path)
                        by_p = _collect_peakfit_results(jp)
                        zbins_p = [z for z in ("b1", "b2", "b3") if ("pbg", z) in by_p]
                        if zbins_p:
                            worst_p = max(zbins_p, key=lambda z: float(by_p[("pbg", z)].get("abs_sigma") or -1.0))
                            wp = by_p[("pbg", worst_p)]
                            wl = by_p.get(("lcdm", worst_p)) or {}
                            z_eff_p = wp.get("z_eff")
                            eps_p_p = wp.get("eps")
                            sig_p_p = wp.get("sigma_eps")
                            abs_sigma_p_p = wp.get("abs_sigma")
                            lb_p_p = bool(wp.get("abs_sigma_is_lower_bound"))
                            eps_l_p = wl.get("eps")
                            sig_l_p = wl.get("sigma_eps")
                            if abs_sigma_p_p is not None:
                                msg = f"pre-recon(Satpathy cov) pbg({worst_p}): |z|={'≥' if lb_p_p else ''}{_fmt_float(abs(abs_sigma_p_p), digits=2)}σ"
                                metric_parts.append(msg)
                            if z_eff_p is not None:
                                metric_parts.append(f"pre-recon z_eff={_fmt_float(z_eff_p, digits=3)}")
                            if eps_p_p is not None:
                                metric_parts.append(f"pre-recon pbg({worst_p}): {_fmt_pm(eps_p_p, sig_p_p)}")
                            if eps_l_p is not None:
                                metric_parts.append(f"pre-recon lcdm({worst_p}): {_fmt_pm(eps_l_p, sig_l_p)}")
                            if eps_p_p is not None and eps_l_p is not None:
                                metric_parts.append(
                                    f"pre-recon Δε(pbg−lcdm)={_fmt_float(float(eps_p_p) - float(eps_l_p), digits=3)}"
                                )
                    except Exception:
                        pass

                # Cross-check: P(k) multipoles peakfit (Beutler et al.; window-convolved).
                pk_post_path = _OUT_PRIVATE / "cosmology" / "cosmology_bao_pk_multipole_peakfit_window_metrics.json"
                if pk_post_path.exists():
                    try:
                        jk = _read_json(pk_post_path)
                        rows_k = jk.get("results") if isinstance(jk.get("results"), list) else []
                        pk_items: List[Tuple[float, int, float, float]] = []
                        for r in rows_k:
                            if not isinstance(r, dict):
                                continue
                            zbin_k = int(_safe_float(r.get("zbin")) or 0)
                            fit_k = (r.get("fit") or {}) if isinstance(r.get("fit"), dict) else {}
                            free_k = (fit_k.get("free") or {}) if isinstance(fit_k.get("free"), dict) else {}
                            eps_k = _safe_float(free_k.get("eps"))
                            ci = free_k.get("eps_ci_1sigma")
                            if eps_k is None or not (isinstance(ci, list) and len(ci) == 2):
                                continue
                            lo = _safe_float(ci[0])
                            hi = _safe_float(ci[1])
                            if lo is None or hi is None:
                                continue
                            sig_k = (float(hi) - float(lo)) / 2.0
                            if not (sig_k > 0.0):
                                continue
                            pk_items.append((abs(float(eps_k)) / float(sig_k), zbin_k, float(eps_k), float(sig_k)))
                        if pk_items:
                            abs_sigma_k, zbin_k, eps_k, sig_k = max(pk_items, key=lambda t: float(t[0]))
                            metric_parts.append(
                                f"P(k)window(post; Beutler) max|z|={_fmt_float(abs_sigma_k, digits=2)}σ（zbin{int(zbin_k)}: {_fmt_pm(eps_k, sig_k)}）"
                            )
                    except Exception:
                        pass

                metric_public = ""
                if abs_sigma_p is not None:
                    metric_public = f"幾何の歪み（ε）: {'≥' if lb_p else ''}{_fmt_float(abs(abs_sigma_p), digits=1)}σ（0に近いほど一致）"

                out_rows.append(
                    TableRow(
                        topic="宇宙論（BAO一次統計）",
                        observable="AP warping ε（ξ0+ξ2; smooth+peak）",
                        data="BOSS DR12v5（CMASSLOWZTOT; combined; z-bin; MW multigrid; Ross cov）",
                        n=len(zbins),
                        reference="ε=0（距離写像が正しければ）",
                        pmodel=_fmt_pm(eps_p, sig_p),
                        metric=" / ".join([p for p in metric_parts if p]),
                        metric_public=metric_public,
                    )
                )

    # Phase 4.5B.21 extension: eBOSS DR16 LRGpCMASS (recon) screening (diag cov).
    eboss_path = _OUT_PRIVATE / "cosmology" / "cosmology_bao_catalog_peakfit_lrgpcmass_rec_combined_metrics.json"
    if eboss_path.exists():
        try:
            je = _read_json(eboss_path)
        except Exception:
            je = None
        if je is not None:
            by = _collect_peakfit_results(je)
            if ("pbg", "none") in by:
                w = by[("pbg", "none")]
                w_l = by.get(("lcdm", "none")) or {}
                z_eff = w.get("z_eff")
                eps_p = w.get("eps")
                sig_p = w.get("sigma_eps")
                abs_sigma_p = w.get("abs_sigma")
                lb_p = bool(w.get("abs_sigma_is_lower_bound"))
                eps_l = w_l.get("eps")
                sig_l = w_l.get("sigma_eps")

                metric_parts: List[str] = []
                if z_eff is not None:
                    metric_parts.append(f"z_eff={_fmt_float(z_eff, digits=3)}")
                if abs_sigma_p is not None:
                    metric_parts.append(f"pbg: |z|={'≥' if lb_p else ''}{_fmt_float(abs(abs_sigma_p), digits=2)}σ")
                if eps_p is not None:
                    metric_parts.append(f"pbg: {_fmt_pm(eps_p, sig_p)}")
                if eps_l is not None:
                    metric_parts.append(f"lcdm: {_fmt_pm(eps_l, sig_l)}")
                if eps_p is not None and eps_l is not None:
                    metric_parts.append(f"Δε(pbg−lcdm)={_fmt_float(float(eps_p) - float(eps_l), digits=3)}")

                metric_public = ""
                if abs_sigma_p is not None:
                    metric_public = f"幾何の歪み（ε）: {'≥' if lb_p else ''}{_fmt_float(abs(abs_sigma_p), digits=1)}σ（0に近いほど一致）"

                out_rows.append(
                    TableRow(
                        topic="宇宙論（BAO一次統計）",
                        observable="AP warping ε（ξ0+ξ2; smooth+peak）",
                        data=(
                            "eBOSS DR16（LRGpCMASS; recon; combined; "
                            + _infer_catalog_sampling_label(root, sample="lrgpcmass_rec", caps="combined")
                            + "; diag）"
                        ),
                        n=1,
                        reference="ε=0（距離写像が正しければ）",
                        pmodel=_fmt_pm(eps_p, sig_p),
                        metric=" / ".join([p for p in metric_parts if p]) + f" / source={eboss_path.name}",
                        metric_public=metric_public,
                    )
                )

    # Phase 4.5B.21.4 extension: eBOSS DR16 QSO (z~1.5) screening (diag cov).
    eboss_qso_path = _OUT_PRIVATE / "cosmology" / "cosmology_bao_catalog_peakfit_qso_combined_metrics.json"
    if eboss_qso_path.exists():
        try:
            jq = _read_json(eboss_qso_path)
        except Exception:
            jq = None
        if jq is not None:
            by = _collect_peakfit_results(jq)
            if ("pbg", "none") in by:
                w = by[("pbg", "none")]
                eps_p = w.get("eps")
                sig_p = w.get("sigma_eps")
                abs_sigma_p = w.get("abs_sigma")
                lb_p = bool(w.get("abs_sigma_is_lower_bound"))
                z_eff = w.get("z_eff")

                wl = by.get(("lcdm", "none"), {})
                eps_l = wl.get("eps")
                sig_l = wl.get("sigma_eps")

                metric_parts = []
                if abs_sigma_p is not None:
                    metric_parts.append(f"pbg: |z|={'≥' if lb_p else ''}{_fmt_float(abs(abs_sigma_p), digits=2)}σ")
                if z_eff is not None:
                    metric_parts.append(f"z_eff={_fmt_float(z_eff, digits=3)}")
                if eps_p is not None:
                    metric_parts.append(f"pbg: {_fmt_pm(eps_p, sig_p)}")
                if eps_l is not None:
                    metric_parts.append(f"lcdm: {_fmt_pm(eps_l, sig_l)}")
                if eps_p is not None and eps_l is not None:
                    metric_parts.append(f"Δε(pbg−lcdm)={_fmt_float(float(eps_p) - float(eps_l), digits=3)}")

                metric_public = ""
                if abs_sigma_p is not None:
                    metric_public = f"幾何の歪み（ε）: {'≥' if lb_p else ''}{_fmt_float(abs(abs_sigma_p), digits=1)}σ（0に近いほど一致）"

                out_rows.append(
                    TableRow(
                        topic="宇宙論（BAO一次統計）",
                        observable="AP warping ε（ξ0+ξ2; smooth+peak）",
                        data=(
                            "eBOSS DR16（QSO; combined; "
                            + _infer_catalog_sampling_label(root, sample="qso", caps="combined")
                            + "; diag）"
                        ),
                        n=1,
                        reference="ε=0（距離写像が正しければ）",
                        pmodel=_fmt_pm(eps_p, sig_p),
                        metric=" / ".join([p for p in metric_parts if p]) + f" / source={eboss_qso_path.name}",
                        metric_public=metric_public,
                    )
                )

    # Phase 4.5B.21.4.4 extension: DESI DR1 LRG (catalog-based; dv=[xi0,xi2] + cov; LRG1/LRG2 bins).
    # Prefer the "primary product" pipeline outputs (RascalC/jackknife) when available.
    desi_candidates = [
        _OUT_PRIVATE
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_reservoir_r0to17_mix__rascalc_cov_reservoir_r0to17_mix_metrics.json",
        _OUT_PUBLIC
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_reservoir_r0to17_mix__rascalc_cov_reservoir_r0to17_mix_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_reservoir_r0to17_mix__rascalc_cov_reservoir_r0to17_mix_metrics.json",
        _OUT_PRIVATE
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_shrink0p2_full_r0r1_mean_metrics.json",
        _OUT_PUBLIC
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_shrink0p2_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_shrink0p2_full_r0r1_mean_metrics.json",
        _OUT_PRIVATE
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_both_full_r0r1_mean_metrics.json",
        _OUT_PUBLIC
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_both_full_r0r1_mean_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0r1_mean__jk_cov_both_full_r0r1_mean_metrics.json",
        _OUT_PRIVATE
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0_metrics.json",
        _OUT_PUBLIC
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0_metrics.json",
        _OUT_PRIVATE
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins__jk_cov_both_metrics.json",
        _OUT_PUBLIC
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins__jk_cov_both_metrics.json",
        root
        / "output"
        / "cosmology"
        / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins__jk_cov_both_metrics.json",
    ]
    desi_path = next((p for p in desi_candidates if p.exists()), None)
    if desi_path is not None:
        try:
            jd = _read_json(desi_path)
        except Exception:
            jd = None
        if jd is not None:
            rows = jd.get("results") if isinstance(jd.get("results"), list) else []

            def _collect_by_dist(dist: str) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    if str(r.get("dist") or "") != dist:
                        continue
                    z_eff = _safe_float(r.get("z_eff"))
                    if z_eff is None:
                        continue
                    sc = (r.get("screening") or {}) if isinstance(r.get("screening"), dict) else {}
                    fit = (r.get("fit") or {}) if isinstance(r.get("fit"), dict) else {}
                    free = (fit.get("free") or {}) if isinstance(fit.get("free"), dict) else {}
                    out.append(
                        {
                            "z_eff": float(z_eff),
                            "eps": _safe_float(free.get("eps")),
                            "sigma_eps": _safe_float(sc.get("sigma_eps_1sigma")),
                            "abs_sigma": _safe_float(sc.get("abs_sigma")),
                            "abs_sigma_is_lower_bound": bool(sc.get("abs_sigma_is_lower_bound")),
                        }
                    )
                out.sort(key=lambda d: float(d.get("z_eff") or 0.0))
                return out

            pts_p = _collect_by_dist("pbg")
            pts_l = _collect_by_dist("lcdm")
            if pts_p:
                worst_p = max(pts_p, key=lambda d: float(d.get("abs_sigma") or -1.0))
                abs_sigma_p = worst_p.get("abs_sigma")
                lb_p = bool(worst_p.get("abs_sigma_is_lower_bound"))
                eps_p = worst_p.get("eps")
                sig_p = worst_p.get("sigma_eps")

                metric_parts: List[str] = []
                for p in pts_p:
                    z_eff = p.get("z_eff")
                    if z_eff is None:
                        continue
                    label = "LRG1" if float(z_eff) < 0.6 else "LRG2"
                    msg_parts: List[str] = [f"{label}: z_eff={_fmt_float(z_eff, digits=3)}"]
                    if p.get("eps") is not None:
                        msg_parts.append(f"pbg: {_fmt_pm(p.get('eps'), p.get('sigma_eps'))}")
                    if p.get("abs_sigma") is not None:
                        msg_parts.append(
                            f"|z|={'≥' if p.get('abs_sigma_is_lower_bound') else ''}{_fmt_float(abs(float(p['abs_sigma'])), digits=2)}σ"
                        )
                    wl = min(pts_l, key=lambda q: abs(float(q.get("z_eff") or 0.0) - float(z_eff))) if pts_l else None
                    if wl and wl.get("eps") is not None:
                        msg_parts.append(f"lcdm: {_fmt_pm(wl.get('eps'), wl.get('sigma_eps'))}")
                        if p.get("eps") is not None and wl.get("eps") is not None:
                            metric_parts.append(
                                " / ".join(msg_parts)
                                + f" / Δε(pbg−lcdm)={_fmt_float(float(p['eps']) - float(wl['eps']), digits=3)}"
                            )
                            continue
                    metric_parts.append(" / ".join(msg_parts))
                cov_label = "jackknife（sky; dv+cov）"
                if "__rascalc_cov" in desi_path.name:
                    cov_label = "RascalC（legendre_projected; dv+cov）"
                metric_parts.append(f"cov={cov_label}")

                metric_public = ""
                if abs_sigma_p is not None:
                    metric_public = (
                        f"幾何の歪み（ε）: max|z|={'≥' if lb_p else ''}{_fmt_float(abs(abs_sigma_p), digits=1)}σ"
                        "（0に近いほど一致）"
                    )

                out_rows.append(
                    TableRow(
                        topic="宇宙論（BAO一次統計）",
                        observable="AP warping ε（ξ0+ξ2; smooth+peak）",
                        data=(
                            "DESI DR1（LRG; combined; y1bins; "
                            + _infer_catalog_sampling_label(root, sample="lrg", caps="combined")
                            + ("; RascalC）" if "__rascalc_cov" in desi_path.name else "; jackknife）")
                        ),
                        n=2,
                        reference="ε=0（距離写像が正しければ）",
                        pmodel=_fmt_pm(eps_p, sig_p),
                        metric=" / ".join([p for p in metric_parts if p]) + f" / source={desi_path.name}",
                        metric_public=metric_public,
                    )
                )

    return out_rows


def _load_cosmology_cmb_polarization_phase_rows(root: Path) -> List[TableRow]:
    path = _first_existing(
        [
            _OUT_PRIVATE / "cosmology" / "cosmology_cmb_polarization_phase_audit_metrics.json",
            _OUT_PUBLIC / "cosmology" / "cosmology_cmb_polarization_phase_audit_metrics.json",
            root / "output" / "cosmology" / "cosmology_cmb_polarization_phase_audit_metrics.json",
        ]
    )
    if path is None:
        return []
    try:
        j = _read_json(path)
    except Exception:
        return []

    phase_fit = j.get("phase_fit") if isinstance(j.get("phase_fit"), dict) else {}
    ee = phase_fit.get("ee") if isinstance(phase_fit.get("ee"), dict) else {}
    te = phase_fit.get("te") if isinstance(phase_fit.get("te"), dict) else {}
    delta_ee = _safe_float(ee.get("delta_fit"))
    delta_te = _safe_float(te.get("delta_fit"))
    abs_ee = _safe_float(ee.get("abs_shift_from_expected"))
    abs_te = _safe_float(te.get("abs_shift_from_expected"))

    gate = j.get("gate") if isinstance(j.get("gate"), dict) else {}
    hard = gate.get("hard_gate") if isinstance(gate.get("hard_gate"), dict) else {}
    hard_pass = bool(hard.get("pass"))
    overall = str(gate.get("overall_status") or "")

    metric_parts: List[str] = []
    if delta_ee is not None:
        metric_parts.append(f"Δϕ_EE={_fmt_float(delta_ee, digits=3)}")
    if delta_te is not None:
        metric_parts.append(f"Δϕ_TE={_fmt_float(delta_te, digits=3)}")
    if abs_ee is not None:
        metric_parts.append(f"|Δϕ_EE−0.5|={_fmt_float(abs_ee, digits=3)}")
    if abs_te is not None:
        metric_parts.append(f"|Δϕ_TE−0.25|={_fmt_float(abs_te, digits=3)}")
    metric_parts.append(f"hard_gate={'pass' if hard_pass else 'fail'}")
    if overall:
        metric_parts.append(f"overall={overall}")

    metric_public = ""
    if abs_ee is not None and abs_te is not None:
        metric_public = (
            f"位相ズレ判定: {'pass' if hard_pass else 'fail'}"
            f"（EE={_fmt_float(abs_ee, digits=3)}, TE={_fmt_float(abs_te, digits=3)}）"
        )

    return [
        TableRow(
            topic="宇宙論（CMB偏極位相）",
            observable="TT/EE/TE 音響位相差（Δϕ_EE, Δϕ_TE）",
            data="Planck 2018 TT/TE/EE（binned）",
            n=None,
            reference="Δϕ_EE=0.5, Δϕ_TE=0.25（位相ロック）",
            pmodel="Θ∝cos(k r_s), Π/E∝sin(k r_s) から位相差を導出",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_cosmology_fsigma8_growth_rows(root: Path) -> List[TableRow]:
    path = _first_existing(
        [
            _OUT_PRIVATE / "cosmology" / "cosmology_fsigma8_growth_mapping_metrics.json",
            _OUT_PUBLIC / "cosmology" / "cosmology_fsigma8_growth_mapping_metrics.json",
            root / "output" / "cosmology" / "cosmology_fsigma8_growth_mapping_metrics.json",
        ]
    )
    if path is None:
        return []
    try:
        j = _read_json(path)
    except Exception:
        return []

    branches = j.get("branches") if isinstance(j.get("branches"), dict) else {}
    delay = branches.get("delay") if isinstance(branches.get("delay"), dict) else {}
    chi2 = _safe_float(delay.get("chi2"))
    dof = int(_safe_float(delay.get("dof")) or 0)
    chi2_dof = _safe_float(delay.get("chi2_per_dof"))
    max_abs_z = _safe_float(delay.get("max_abs_z_score"))
    tau_eff = _safe_float(delay.get("tau_eff_gyr"))
    gates = j.get("gates") if isinstance(j.get("gates"), dict) else {}
    overall = str(gates.get("overall_status") or "")
    n_pts = int(_safe_float((j.get("inputs") or {}).get("n_rsd_points")) or 0) if isinstance(j.get("inputs"), dict) else 0

    metric_parts: List[str] = []
    if chi2 is not None and dof > 0:
        metric_parts.append(f"delay: χ²/dof={_fmt_float(chi2, digits=3)}/{dof}")
    if chi2_dof is not None:
        metric_parts.append(f"χ²/ν={_fmt_float(chi2_dof, digits=3)}")
    if max_abs_z is not None:
        metric_parts.append(f"max|z|={_fmt_float(max_abs_z, digits=3)}")
    if tau_eff is not None:
        metric_parts.append(f"τ_eff={_fmt_float(tau_eff, digits=4)} Gyr")
    if overall:
        metric_parts.append(f"overall={overall}")

    metric_public = ""
    if chi2 is not None and dof > 0:
        metric_public = f"fσ8整合: χ²/dof={_fmt_float(chi2, digits=3)}/{dof}"
        if max_abs_z is not None:
            metric_public += f"（max|z|={_fmt_float(max_abs_z, digits=3)}）"

    return [
        TableRow(
            topic="宇宙論（構造形成 fσ8）",
            observable="成長率 fσ8（遅延枝の実効摩擦 Γ_eff）",
            data="BOSS DR12 consensus fσ8（RSD）",
            n=n_pts if n_pts > 0 else None,
            reference="観測 fσ8(z)（一次統計）",
            pmodel="遅延ポテンシャルから Γ_eff を導出し growth 方程式へ写像",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_cosmology_cmb_acoustic_peak_rows(root: Path) -> List[TableRow]:
    path = _first_existing(
        [
            _OUT_PUBLIC / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction_metrics.json",
            _OUT_PRIVATE / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction_metrics.json",
            root / "output" / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction_metrics.json",
        ]
    )
    if path is None:
        return []
    try:
        j = _read_json(path)
    except Exception:
        return []

    gate = j.get("gate") if isinstance(j.get("gate"), dict) else {}
    first3 = gate.get("first3") if isinstance(gate.get("first3"), dict) else {}
    ext46 = gate.get("extended_4to6") if isinstance(gate.get("extended_4to6"), dict) else {}
    overall = gate.get("overall") if isinstance(gate.get("overall"), dict) else {}
    overall_ext = gate.get("overall_extended") if isinstance(gate.get("overall_extended"), dict) else {}

    model = j.get("model") if isinstance(j.get("model"), dict) else {}
    closure = model.get("first_principles_closure") if isinstance(model.get("first_principles_closure"), dict) else {}
    dm_free = (
        closure.get("third_peak_dm_free_damping_proof")
        if isinstance(closure.get("third_peak_dm_free_damping_proof"), dict)
        else {}
    )

    first3_ell = _safe_float(first3.get("max_abs_delta_ell"))
    first3_amp = _safe_float(first3.get("max_abs_delta_amp_rel"))
    ext46_ell = _safe_float(ext46.get("max_abs_delta_ell"))
    ext46_amp = _safe_float(ext46.get("max_abs_delta_amp_rel"))
    theorem_pass = bool(dm_free.get("attenuation_theorem_pass"))
    a3_pred = _safe_float(dm_free.get("a3_over_a1_pred_dm_free"))
    a3_obs = _safe_float(dm_free.get("a3_over_a1_observed"))

    metric_parts: List[str] = []
    if first3_ell is not None:
        metric_parts.append(f"first3 max|Δℓ|={_fmt_float(first3_ell, digits=3)}")
    if first3_amp is not None:
        metric_parts.append(f"first3 max|ΔA/A|={_fmt_float(first3_amp, digits=3)}")
    if ext46_ell is not None:
        metric_parts.append(f"holdout4-6 max|Δℓ|={_fmt_float(ext46_ell, digits=3)}")
    if ext46_amp is not None:
        metric_parts.append(f"holdout4-6 max|ΔA/A|={_fmt_float(ext46_amp, digits=3)}")
    metric_parts.append(f"dm-free第3ピーク減衰={'pass' if theorem_pass else 'fail'}")
    if a3_pred is not None and a3_obs is not None:
        metric_parts.append(f"A3/A1(pred/obs)={_fmt_float(a3_pred, digits=3)}/{_fmt_float(a3_obs, digits=3)}")
    if isinstance(overall, dict) and overall.get("status"):
        metric_parts.append(f"core={str(overall.get('status'))}")
    if isinstance(overall_ext, dict) and overall_ext.get("status"):
        metric_parts.append(f"extended={str(overall_ext.get('status'))}")

    metric_public = ""
    if first3_ell is not None and ext46_ell is not None:
        metric_public = (
            f"逆同定+holdout整合: first3 max|Δℓ|={_fmt_float(first3_ell, digits=2)}, "
            f"holdout4-6 max|Δℓ|={_fmt_float(ext46_ell, digits=2)}"
        )
        metric_public += f" / DMなし第3ピーク減衰={'成立' if theorem_pass else '未達'}"

    n_obs = len(j.get("observed_peaks") or []) if isinstance(j.get("observed_peaks"), list) else None
    return [
        TableRow(
            topic="宇宙論（CMB音響ピーク）",
            observable="TT音響ピーク（1〜3逆同定, 4〜6 holdout）",
            data="Planck 2018 TT binned spectrum",
            n=n_obs if (n_obs is not None and n_obs > 0) else None,
            reference="第1〜3ピーク位置/振幅（第4〜6はholdout）",
            pmodel="光子-バリオン流体のcos解 + DMなし減衰則",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_sparc_rotation_rows(root: Path) -> List[TableRow]:
    path = _first_existing(
        [
            _OUT_PUBLIC / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json",
            _OUT_PRIVATE / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json",
            root / "output" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json",
        ]
    )
    if path is None:
        return []
    try:
        j = _read_json(path)
    except Exception:
        return []

    counts = j.get("counts") if isinstance(j.get("counts"), dict) else {}
    fit = j.get("fit_results") if isinstance(j.get("fit_results"), dict) else {}
    pm = fit.get("pmodel_corrected") if isinstance(fit.get("pmodel_corrected"), dict) else {}
    baryon = fit.get("baryon_only") if isinstance(fit.get("baryon_only"), dict) else {}
    comp = fit.get("comparison") if isinstance(fit.get("comparison"), dict) else {}
    gal = fit.get("galaxy_level_summary") if isinstance(fit.get("galaxy_level_summary"), dict) else {}

    n_gal = int(_safe_float(counts.get("n_galaxies")) or 0)
    delta_chi2 = _safe_float(comp.get("delta_chi2_baryon_minus_pmodel"))
    ratio = _safe_float(comp.get("chi2_dof_ratio_pmodel_over_baryon"))
    pm_chi2_dof = _safe_float(pm.get("chi2_dof"))
    baryon_chi2_dof = _safe_float(baryon.get("chi2_dof"))
    median_pm = _safe_float(gal.get("median_chi2_dof_pmodel"))
    median_b = _safe_float(gal.get("median_chi2_dof_baryon"))
    upsilon = _safe_float(pm.get("upsilon_best"))

    metric_parts: List[str] = []
    if delta_chi2 is not None:
        metric_parts.append(f"Δχ²(baryon−P)={_fmt_float(delta_chi2, digits=2)}")
    if ratio is not None:
        metric_parts.append(f"χ²/ν比(P/baryon)={_fmt_float(ratio, digits=3)}")
    if pm_chi2_dof is not None and baryon_chi2_dof is not None:
        metric_parts.append(
            f"χ²/ν: P={_fmt_float(pm_chi2_dof, digits=3)}, baryon={_fmt_float(baryon_chi2_dof, digits=3)}"
        )
    if median_pm is not None and median_b is not None:
        metric_parts.append(
            f"銀河別中央値 χ²/ν: P={_fmt_float(median_pm, digits=3)}, baryon={_fmt_float(median_b, digits=3)}"
        )
    if upsilon is not None:
        metric_parts.append(f"single-Υ best={_fmt_float(upsilon, digits=3)}")

    metric_public = ""
    if delta_chi2 is not None and pm_chi2_dof is not None:
        metric_public = (
            f"single-Υで説明力向上: Δχ²={_fmt_float(delta_chi2, digits=1)}, "
            f"P-model χ²/ν={_fmt_float(pm_chi2_dof, digits=2)}"
        )

    return [
        TableRow(
            topic="銀河回転曲線（SPARC）",
            observable="V_obs(R) の回転曲線フィット（single-Υ）",
            data="SPARC 175 galaxies",
            n=n_gal if n_gal > 0 else None,
            reference="baryon-only baseline",
            pmodel="P場補正項 + single-Υ",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_bbn_rows(root: Path) -> List[TableRow]:
    path = _first_existing(
        [
            _OUT_PUBLIC / "quantum" / "bbn_he4_watch_convergence_audit_metrics.json",
            _OUT_PRIVATE / "quantum" / "bbn_he4_watch_convergence_audit_metrics.json",
            root / "output" / "quantum" / "bbn_he4_watch_convergence_audit_metrics.json",
        ]
    )
    if path is None:
        return []
    try:
        j = _read_json(path)
    except Exception:
        return []

    decision = j.get("decision") if isinstance(j.get("decision"), dict) else {}
    lin = j.get("linear_propagation") if isinstance(j.get("linear_propagation"), dict) else {}
    mc = j.get("mc_propagation") if isinstance(j.get("mc_propagation"), dict) else {}
    inputs = j.get("inputs") if isinstance(j.get("inputs"), dict) else {}
    nominal = j.get("freeze_nominal") if isinstance(j.get("freeze_nominal"), dict) else {}

    y_obs = _safe_float(inputs.get("he4_obs"))
    y_sig = _safe_float(inputs.get("he4_sigma_obs"))
    y_pred = _safe_float(nominal.get("y_pred"))
    z_lin = _safe_float(lin.get("z_abs"))
    z_mc = _safe_float(mc.get("z_abs"))
    overall = str(decision.get("overall_status") or "")
    criterion = str(decision.get("criterion") or "")
    q_b = None
    params = j.get("uncertainty_parameters") if isinstance(j.get("uncertainty_parameters"), list) else []
    for p in params:
        if not isinstance(p, dict):
            continue
        if str(p.get("name") or "") == "q_b":
            q_b = _safe_float(p.get("mu"))
            break

    metric_parts: List[str] = []
    if y_pred is not None and y_obs is not None and y_sig is not None:
        metric_parts.append(
            f"Yp(pred/obs)={_fmt_float(y_pred, digits=4)}/{_fmt_float(y_obs, digits=4)}±{_fmt_float(y_sig, digits=4)}"
        )
    if z_lin is not None:
        metric_parts.append(f"|z|_linear={_fmt_float(z_lin, digits=3)}")
    if z_mc is not None:
        metric_parts.append(f"|z|_MC={_fmt_float(z_mc, digits=3)}")
    if q_b is not None:
        metric_parts.append(f"q_B={_fmt_float(q_b, digits=3)}")
    if overall:
        metric_parts.append(f"overall={overall}")
    if criterion:
        metric_parts.append(f"criterion={criterion}")

    metric_public = ""
    if y_pred is not None and y_obs is not None:
        metric_public = f"Yp整合: pred={_fmt_float(y_pred, digits=4)}, obs={_fmt_float(y_obs, digits=4)}"
        if z_lin is not None:
            metric_public += f" / |z|={_fmt_float(z_lin, digits=2)}"

    n_mc = int(_safe_float(inputs.get("mc_samples")) or 0)
    return [
        TableRow(
            topic="BBN（初期熱史）",
            observable="He-4 質量比 Yp（凍結温度導出）",
            data="light-element abundance benchmark",
            n=n_mc if n_mc > 0 else None,
            reference="Yp≈0.25（観測）",
            pmodel="q_B=1/2 枝 + 弱相互作用凍結で導出",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_background_metric_choice_rows(root: Path) -> List[TableRow]:
    out: List[TableRow] = []
    for case_label, relpath in (
        ("caseB", "pmodel_vector_metric_choice_audit_caseB_effective.json"),
        ("caseA", "pmodel_vector_metric_choice_audit_caseA_flat.json"),
    ):
        path = _first_existing(
            [
                _OUT_PUBLIC / "theory" / relpath,
                _OUT_PRIVATE / "theory" / relpath,
                root / "output" / "theory" / relpath,
            ]
        )
        if path is None:
            continue
        try:
            j = _read_json(path)
        except Exception:
            continue

        case_result = j.get("case_result") if isinstance(j.get("case_result"), dict) else {}
        summary = case_result.get("summary") if isinstance(case_result.get("summary"), dict) else {}
        derived = case_result.get("derived") if isinstance(case_result.get("derived"), dict) else {}
        rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
        overall = str(summary.get("overall_status") or "")
        decision = str(summary.get("decision") or "")
        z_gamma = _safe_float(derived.get("z_gamma"))
        mercury_rel = _safe_float(derived.get("mercury_rel_error"))
        nonlinear = bool(derived.get("nonlinear_pde_closure_pass")) if "nonlinear_pde_closure_pass" in derived else None

        metric_parts: List[str] = []
        if z_gamma is not None:
            metric_parts.append(f"γ gate z={_fmt_float(z_gamma, digits=3)}")
        if mercury_rel is not None:
            metric_parts.append(f"Mercury係数残差={_fmt_float(mercury_rel, digits=6)}")
        if nonlinear is not None:
            metric_parts.append(f"nonlinear_pde_closure={'pass' if nonlinear else 'fail'}")
        if decision:
            metric_parts.append(f"decision={decision}")
        if overall:
            metric_parts.append(f"overall={overall}")

        metric_public = ""
        if case_label == "caseB":
            metric_public = "有効計量 g_{μν}(P) を採用（weak-field + 非線形PDE closure）"
        else:
            metric_public = "平坦背景 η_{μν} は weak-field 監査で棄却"

        out.append(
            TableRow(
                topic=f"背景計量（{case_label}）",
                observable="ベクトル拡張の計量選択ゲート",
                data="Mercury + 光偏向 + PDE closure",
                n=len(rows) if rows else None,
                reference="weak-field整合 + closure pass",
                pmodel="caseB: g_{μν}(P) / caseA: η_{μν}",
                metric=" / ".join(metric_parts),
                metric_public=metric_public,
            )
        )
    return out


def _load_gw_polarization_rows(root: Path) -> List[TableRow]:
    path = _first_existing(
        [
            _OUT_PUBLIC / "gw" / "pmodel_vector_gw_polarization_mapping_audit.json",
            _OUT_PRIVATE / "gw" / "pmodel_vector_gw_polarization_mapping_audit.json",
            root / "output" / "gw" / "pmodel_vector_gw_polarization_mapping_audit.json",
        ]
    )
    if path is None:
        return []
    try:
        j = _read_json(path)
    except Exception:
        return []

    summary = j.get("summary") if isinstance(j.get("summary"), dict) else {}
    network = summary.get("network_gate") if isinstance(summary.get("network_gate"), dict) else {}
    overall = str(summary.get("overall_status") or "")
    reason = str(summary.get("overall_reason") or "")
    n_usable = int(_safe_float(summary.get("n_usable_events")) or 0)
    scalar_excl = bool(network.get("scalar_exclusion_pass")) if "scalar_exclusion_pass" in network else None
    tensor_pass = bool(network.get("tensor_consistency_pass")) if "tensor_consistency_pass" in network else None
    scalar_red = bool(network.get("scalar_reduction_pass")) if "scalar_reduction_pass" in network else None

    metric_parts: List[str] = []
    if overall:
        metric_parts.append(f"overall={overall}")
    if scalar_red is not None:
        metric_parts.append(f"scalar_reduction={'pass' if scalar_red else 'fail'}")
    if scalar_excl is not None:
        metric_parts.append(f"scalar_exclusion={'pass' if scalar_excl else 'fail'}")
    if tensor_pass is not None:
        metric_parts.append(f"tensor_consistency={'pass' if tensor_pass else 'fail'}")
    if reason:
        metric_parts.append(f"reason={reason}")

    metric_public = "3検出器(H1/L1/V1)でスカラー縮退は低減したが完全排除は未達"
    return [
        TableRow(
            topic="重力波（偏光モード）",
            observable="ベクトル横波 vs スカラー縮退（H1/L1/V1）",
            data="GW150914 + network gate events",
            n=n_usable if n_usable > 0 else None,
            reference="scalar_exclusion_pass=true（hard pass）",
            pmodel="P_μ ベクトル横波応答",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_cosmology_cluster_collision_rows(root: Path) -> List[TableRow]:
    deriv_path = _first_existing(
        [
            _OUT_PUBLIC / "cosmology" / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
            _OUT_PRIVATE / "cosmology" / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
            root / "output" / "cosmology" / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
        ]
    )
    audit_path = _first_existing(
        [
            _OUT_PUBLIC / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.json",
            _OUT_PRIVATE / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.json",
            root / "output" / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.json",
        ]
    )
    if deriv_path is None and audit_path is None:
        return []

    chi2_dof = None
    max_abs_z = None
    ad_hoc_n = None
    xi_mode = ""
    deriv_status = ""
    n_obs = None
    if deriv_path is not None:
        try:
            jd = _read_json(deriv_path)
            fit = jd.get("fit") if isinstance(jd.get("fit"), dict) else {}
            chi2_dof = _safe_float(fit.get("chi2_dof"))
            max_abs_z = _safe_float(fit.get("max_abs_z_offset"))
            ad_hoc_n = int(_safe_float(fit.get("ad_hoc_parameter_count")) or 0)
            xi_mode = str(fit.get("xi_mode") or "")
            n_obs = int(_safe_float(fit.get("n_observations")) or 0)
            deriv_status = str((jd.get("decision") or {}).get("overall_status") or "")
        except Exception:
            chi2_dof = None

    chi2_dof_audit = None
    max_abs_z_lens = None
    audit_status = ""
    if audit_path is not None:
        try:
            ja = _read_json(audit_path)
            pm = ((ja.get("models") or {}).get("pmodel_corrected") or {}) if isinstance(ja.get("models"), dict) else {}
            chi2_dof_audit = _safe_float(pm.get("chi2_dof"))
            max_abs_z_lens = _safe_float(pm.get("max_abs_z_p_lens"))
            audit_status = str((ja.get("decision") or {}).get("overall_status") or "")
        except Exception:
            chi2_dof_audit = None

    metric_parts: List[str] = []
    if chi2_dof is not None:
        metric_parts.append(f"導出系: χ²/ν={_fmt_float(chi2_dof, digits=3)}")
    if max_abs_z is not None:
        metric_parts.append(f"導出系 max|z|={_fmt_float(max_abs_z, digits=3)}")
    if chi2_dof_audit is not None:
        metric_parts.append(f"オフセット監査: χ²/ν={_fmt_float(chi2_dof_audit, digits=3)}")
    if max_abs_z_lens is not None:
        metric_parts.append(f"レンズ重心 max|z|={_fmt_float(max_abs_z_lens, digits=3)}")
    if ad_hoc_n is not None:
        metric_parts.append(f"ad_hoc={ad_hoc_n}")
    if xi_mode:
        metric_parts.append(f"xi_mode={xi_mode}")
    if deriv_status:
        metric_parts.append(f"derivation={deriv_status}")
    if audit_status:
        metric_parts.append(f"audit={audit_status}")

    metric_public = ""
    if chi2_dof is not None:
        metric_public = f"Bullet系オフセット（導出）: χ²/ν={_fmt_float(chi2_dof, digits=3)}"
    if max_abs_z is not None:
        metric_public += f" / max|z|={_fmt_float(max_abs_z, digits=3)}"

    return [
        TableRow(
            topic="宇宙論（銀河団衝突オフセット）",
            observable="レンズ中心−ガス中心オフセット（Bullet系）",
            data="Bullet Cluster offset observables（main/sub）",
            n=n_obs if (n_obs is not None and n_obs > 0) else None,
            reference="観測オフセット（κ/Σ と X線中心）",
            pmodel="P_μ–J^μ 遅延核から Δx を導出（外部手足しなし）",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _load_strong_field_higher_order_rows(root: Path) -> List[TableRow]:
    path = _first_existing(
        [
            _OUT_PUBLIC / "theory" / "pmodel_strong_field_higher_order_audit.json",
            _OUT_PRIVATE / "theory" / "pmodel_strong_field_higher_order_audit.json",
            root / "output" / "theory" / "pmodel_strong_field_higher_order_audit.json",
        ]
    )
    if path is None:
        return []
    try:
        j = _read_json(path)
    except Exception:
        return []

    fits = j.get("fits") if isinstance(j.get("fits"), dict) else {}
    joint = fits.get("joint") if isinstance(fits.get("joint"), dict) else {}
    delta_aic = _safe_float(joint.get("delta_aic_fit_minus_baseline"))
    lam = _safe_float(joint.get("lambda_fit"))
    lam_sig = _safe_float(joint.get("lambda_sigma"))
    chi2_dof = _safe_float(joint.get("chi2_dof_fit"))
    n_obs = int(_safe_float(joint.get("n_obs")) or 0)
    decision = str((j.get("decision") or {}).get("overall_status") or "")

    metric_parts: List[str] = []
    if delta_aic is not None:
        metric_parts.append(f"ΔAIC(fit−baseline)={_fmt_float(delta_aic, digits=3)}")
    if lam is not None and lam_sig is not None and lam_sig > 0:
        metric_parts.append(f"λ_H={_fmt_float(lam, digits=6)}±{_fmt_float(lam_sig, digits=6)}")
    elif lam is not None:
        metric_parts.append(f"λ_H={_fmt_float(lam, digits=6)}")
    if chi2_dof is not None:
        metric_parts.append(f"χ²/ν={_fmt_float(chi2_dof, digits=3)}")
    if decision:
        metric_parts.append(f"overall={decision}")

    metric_public = ""
    if delta_aic is not None:
        metric_public = f"強場高次項の同時拘束: ΔAIC={_fmt_float(delta_aic, digits=3)}"
    if decision:
        metric_public += f"（{decision}）"

    return [
        TableRow(
            topic="強場（高次項同時拘束）",
            observable="単一 λ_H で EHT+GW+Pulsar+Fe-Kα を同時拘束",
            data="EHT/GW250114/連星パルサー/Fe-Kα（統合監査）",
            n=n_obs if n_obs > 0 else None,
            reference="baseline: λ_H=0（高次項なし）",
            pmodel="fit: 単一 λ_H（チャネル横断）",
            metric=" / ".join(metric_parts),
            metric_public=metric_public,
        )
    ]


def _infer_catalog_sampling_label(root: Path, *, sample: str, caps: str) -> str:
    """
    Best-effort label for catalog sampling (random subsampling) used by ξℓ inputs.

    We infer from the manifest npz filename because some manifests may mark cached files without
    preserving the original sampling metadata.
    """
    try:
        manifest: Path
        if sample in ("lrgpcmass_rec", "qso"):
            manifest = root / "data" / "cosmology" / "eboss_dr16_lss" / "manifest.json"
        elif sample in ("lrg",):
            # Prefer the multi-random reservoir when present (primary-product; Step 4.5B.21.4.4.6).
            cand_res = root / "data" / "cosmology" / "desi_dr1_lss_reservoir_r0to17_mix" / "manifest.json"
            cand_full = root / "data" / "cosmology" / "desi_dr1_lss_full_r0" / "manifest.json"
            if cand_res.exists():
                manifest = cand_res
            elif cand_full.exists():
                manifest = cand_full
            else:
                manifest = root / "data" / "cosmology" / "desi_dr1_lss" / "manifest.json"
        else:
            return "random=?"
        if not manifest.exists():
            return "random=?"
        j = _read_json(manifest)
        items = j.get("items") or {}
        caps_to_use = ["north", "south"] if caps == "combined" else [caps]

        def _desc_from_name(name: str) -> str:
            if ".prefix_" in name:
                try:
                    n = int(name.split(".prefix_", 1)[1].split(".npz", 1)[0])
                    return f"prefix random={n//1000000}M" if (n % 1_000_000 == 0) else f"prefix random={n:,}"
                except Exception:
                    return "prefix random=?"
            if ".reservoir_" in name:
                try:
                    token = name.split(".reservoir_", 1)[1].split(".npz", 1)[0]
                    n = int(token.split("_", 1)[0])
                    return f"reservoir random={n//1000000}M" if (n % 1_000_000 == 0) else f"reservoir random={n:,}"
                except Exception:
                    return "reservoir random=?"
            return "full random"

        descs: List[str] = []
        for cap in caps_to_use:
            it = items.get(f"{sample}:{cap}") or {}
            rnd = it.get("random") or {}
            npz = rnd.get("npz_path")
            if not npz:
                continue
            descs.append(_desc_from_name(Path(str(npz)).name))
        descs = sorted(set([d for d in descs if d]))
        if not descs:
            return "random=?"
        if len(descs) == 1:
            return descs[0]
        return " / ".join(descs)
    except Exception:
        return "random=?"


def _load_frame_dragging_rows(root: Path) -> List[TableRow]:
    out: List[TableRow] = []
    path = _first_existing(
        [
            _OUT_PUBLIC / "theory" / "frame_dragging_experiments.json",
            _OUT_PRIVATE / "theory" / "frame_dragging_experiments.json",
            root / "output" / "theory" / "frame_dragging_experiments.json",
        ]
    )
    if path is not None:
        j = _read_json(path)
        rows = j.get("rows") or []
    else:
        rows = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        label = str(r.get("short_label") or r.get("id") or "")
        mu = _safe_float(r.get("mu"))
        sig = _safe_float(r.get("mu_sigma"))
        z = _safe_float(r.get("z_score"))

        src = r.get("source") or {}
        if not isinstance(src, dict):
            src = {}
        doi = str(src.get("doi") or "")
        year = str(src.get("year") or "")
        src_txt = doi
        if year and doi:
            src_txt = f"{year}, {doi}"
        elif year and not src_txt:
            src_txt = year

        obs_txt = ""
        if mu is not None and sig is not None:
            obs_txt = f"μ={_fmt_float(mu, digits=4)}±{_fmt_float(sig, digits=4)}"
        elif mu is not None:
            obs_txt = f"μ={_fmt_float(mu, digits=4)}"

        metric_parts: List[str] = []
        if z is not None:
            metric_parts.append(f"z={_fmt_float(z, digits=3)}")

        omega_pred = _safe_float(r.get("omega_pred_mas_per_yr"))
        omega_obs = _safe_float(r.get("omega_obs_mas_per_yr"))
        omega_sig = _safe_float(r.get("omega_obs_sigma_mas_per_yr"))
        if omega_pred is not None and omega_obs is not None:
            if omega_sig is not None:
                metric_parts.append(
                    f"|Ω|={_fmt_float(abs(omega_obs), digits=3)}±{_fmt_float(abs(omega_sig), digits=3)} mas/yr（予測 {_fmt_float(abs(omega_pred), digits=3)}）"
                )
            else:
                metric_parts.append(
                    f"|Ω|={_fmt_float(abs(omega_obs), digits=3)} mas/yr（予測 {_fmt_float(abs(omega_pred), digits=3)}）"
                )

        if src_txt:
            metric_parts.append(src_txt)

        metric_public = ""
        if mu is not None and sig is not None and sig > 0:
            metric_public = f"観測との差: {_fmt_float(abs(mu - 1.0) / sig, digits=2)}σ（1に近いほど一致）"

        out.append(
            TableRow(
                topic="回転（フレームドラッグ）",
                observable="比 μ=|Ω_obs|/|Ω_pred|（GR=1）",
                data=label,
                n=None,
                reference=obs_txt or "μ（観測）",
                pmodel="μ=1",
                metric=" / ".join(metric_parts),
                metric_public=metric_public,
            )
        )

    scalar_audit_path = _first_existing(
        [
            _OUT_PUBLIC / "theory" / "frame_dragging_scalar_limit_combined_audit.json",
            _OUT_PRIVATE / "theory" / "frame_dragging_scalar_limit_combined_audit.json",
            root / "output" / "theory" / "frame_dragging_scalar_limit_combined_audit.json",
        ]
    )
    if scalar_audit_path is not None:
        try:
            js = _read_json(scalar_audit_path)
            gate = js.get("gate") if isinstance(js.get("gate"), dict) else {}
            summary = js.get("summary") if isinstance(js.get("summary"), dict) else {}
            rows_scalar = js.get("rows") if isinstance(js.get("rows"), list) else []

            z_gate = _safe_float(gate.get("z_reject"))
            z_by_exp: Dict[str, float] = {}
            for rr in rows_scalar:
                if not isinstance(rr, dict):
                    continue
                observable = str(rr.get("observable") or "")
                if observable != "frame_dragging":
                    continue
                exp = str(rr.get("experiment") or rr.get("label") or rr.get("id") or "")
                z_scalar = _safe_float(rr.get("z_scalar"))
                if exp and z_scalar is not None:
                    z_by_exp[exp] = z_scalar

            n_channels = int(_safe_float(summary.get("frame_dragging_channels_n")) or len(z_by_exp))
            n_reject = int(_safe_float(summary.get("frame_dragging_reject_n")) or 0)
            decision = str(summary.get("decision") or "")
            overall = str(summary.get("overall_status") or "")

            z_gp = z_by_exp.get("GP-B")
            z_lageos = z_by_exp.get("LAGEOS")

            reference_parts: List[str] = []
            if z_gp is not None:
                reference_parts.append(f"GP-B z={_fmt_float(z_gp, digits=2)}")
            if z_lageos is not None:
                reference_parts.append(f"LAGEOS z={_fmt_float(z_lageos, digits=2)}")
            if z_gate is not None:
                reference_parts.append(f"reject if abs(z)>{_fmt_float(z_gate, digits=1)}")

            metric_parts: List[str] = []
            if z_gp is not None or z_lageos is not None:
                zz: List[str] = []
                if z_gp is not None:
                    zz.append(f"GP-B z_scalar={_fmt_float(z_gp, digits=3)}")
                if z_lageos is not None:
                    zz.append(f"LAGEOS z_scalar={_fmt_float(z_lageos, digits=3)}")
                metric_parts.append(" / ".join(zz))
            if n_channels > 0:
                metric_parts.append(f"frame-dragging reject={n_reject}/{n_channels}")
            if z_gate is not None:
                metric_parts.append(f"gate=abs(z)>{_fmt_float(z_gate, digits=1)}")
            if decision:
                metric_parts.append(f"decision={decision}")
            if overall:
                metric_parts.append(f"overall={overall}")

            metric_public = "純スカラー極限は棄却（GP-B/LAGEOS の両方で frame-dragging が reject）"
            if z_gp is not None and z_lageos is not None:
                metric_public += (
                    f" / GP-B z={_fmt_float(z_gp, digits=2)}, "
                    f"LAGEOS z={_fmt_float(z_lageos, digits=1)}"
                )

            out.append(
                TableRow(
                    topic="回転（フレームドラッグ）",
                    observable="純スカラー極限（回転項なし）棄却ゲート",
                    data="GP-B + LAGEOS（統合）",
                    n=n_channels if n_channels > 0 else None,
                    reference=" / ".join(reference_parts) if reference_parts else "統合 scalar-limit 監査",
                    pmodel="回転源 J^i≠0 では P_φ≠0（4元ベクトル P_μ が必須）",
                    metric=" / ".join(metric_parts),
                    metric_public=metric_public,
                )
            )
        except Exception:
            pass
    return out


def _load_delta_rows(root: Path) -> List[TableRow]:
    path = _OUT_PRIVATE / "theory" / "delta_saturation_constraints.json"
    if not path.exists():
        return []
    j = _read_json(path)
    delta = float(j["delta_adopted"])
    gamma_max = float(j["gamma_max_for_delta_adopted"])

    strictest = None
    strictest_label = None
    for r in j.get("rows") or []:
        du = float(r["delta_upper_from_gamma"])
        if strictest is None or du < strictest:
            strictest = du
            strictest_label = str(r.get("label") or r.get("key") or "")

    metric = ""
    if strictest is not None:
        metric = f"観測からの上限（最も厳しい例）: δ < {_fmt_sci(strictest, digits=2)}（{strictest_label}）"

    metric_public = "δが小さいほど γ_max が大きくなり、SR（発散）に近い"
    if strictest is not None:
        metric_public += f" / 観測上限: δ<{_fmt_sci(strictest, digits=2)}"

    return [
        TableRow(
            topic="速度飽和 δ（理論）",
            observable="γ_max（v→cで発散しない）",
            data="モデル仮定（δ>0, 無次元）",
            n=None,
            reference="SR: γ→∞（v→c）",
            pmodel=f"δ={_fmt_sci(delta, digits=1)} -> γ_max~={_fmt_sci(gamma_max, digits=1)}",
            metric=metric,
            metric_public=metric_public,
        )
    ]


def _load_quantum_bell_rows(root: Path) -> List[TableRow]:
    path = _OUT_PUBLIC / "quantum" / "bell" / "table1_row.json"
    if not path.exists():
        return []
    j = _read_json(path)
    n = j.get("n")
    try:
        n_i = int(n) if n is not None else None
    except Exception:
        n_i = None
    return [
        TableRow(
            topic=str(j.get("topic") or "Bell（公開一次データ）"),
            observable=str(j.get("observable") or ""),
            data=str(j.get("data") or ""),
            n=n_i,
            reference=str(j.get("reference") or ""),
            pmodel=str(j.get("pmodel") or ""),
            metric=str(j.get("metric") or ""),
            metric_public=str(j.get("metric_public") or ""),
        )
    ]


def _load_quantum_gravity_quantum_interference_rows(root: Path) -> List[TableRow]:
    rows: List[TableRow] = []

    # COW (neutron interferometry) — magnitude/scaling check.
    path_cow = _OUT_PUBLIC / "quantum" / "cow_phase_shift_metrics.json"
    if path_cow.exists():
        j = _read_json(path_cow)
        res = j.get("results") or {}
        cfg = j.get("config") or {}
        phi0_cycles = _safe_float((res.get("phi0_cycles") if isinstance(res, dict) else None))
        h_m = _safe_float((cfg.get("H_m") if isinstance(cfg, dict) else None))
        v0 = _safe_float((cfg.get("v0_m_per_s") if isinstance(cfg, dict) else None))
        parts: List[str] = []
        if phi0_cycles is not None:
            parts.append(f"|Δφ|≈{_fmt_float(abs(phi0_cycles), digits=3)} cycles")
        if h_m is not None:
            parts.append(f"H={_fmt_float(h_m * 100.0, digits=3)} cm")
        if v0 is not None:
            parts.append(f"v0={_fmt_float(v0, digits=4)} m/s")
        parts.append(f"source={path_cow.name}")
        rows.append(
            TableRow(
                topic="重力×量子干渉",
                observable="COW位相（スケール）",
                data="中性子干渉（代表値）",
                n=None,
                reference="Δφ=-m g H^2/(ħ v0)（文献式）",
                pmodel="弱場写像で同スケール",
                metric=" / ".join(parts),
            )
        )

    # Atom interferometer gravimeter — magnitude/scaling check.
    path_ai = _OUT_PUBLIC / "quantum" / "atom_interferometer_gravimeter_phase_metrics.json"
    if path_ai.exists():
        j = _read_json(path_ai)
        res = j.get("results") or {}
        cfg = j.get("config") or {}
        phi_ref = _safe_float((res.get("phi_ref_rad") if isinstance(res, dict) else None))
        t_s = _safe_float((cfg.get("T_s") if isinstance(cfg, dict) else None))
        parts = []
        if phi_ref is not None:
            parts.append(f"φ≈{_fmt_sci(phi_ref, digits=2)} rad")
        if t_s is not None:
            parts.append(f"T={_fmt_float(t_s, digits=3)} s")
        parts.append(f"source={path_ai.name}")
        rows.append(
            TableRow(
                topic="重力×量子干渉",
                observable="原子干渉計重力計位相（スケール）",
                data="代表値（T固定）",
                n=None,
                reference="φ≈k_eff g T^2（文献式）",
                pmodel="弱場写像で位相スケール固定",
                metric=" / ".join(parts),
            )
        )

    # Optical clock chronometric leveling — consistency check vs geodesy.
    path_clock = _OUT_PUBLIC / "quantum" / "optical_clock_chronometric_leveling_metrics.json"
    if path_clock.exists():
        j = _read_json(path_clock)
        d = j.get("derived") or {}
        z = _safe_float((d.get("z_score") if isinstance(d, dict) else None))
        eps = _safe_float((d.get("epsilon") if isinstance(d, dict) else None))
        eps_sig = _safe_float((d.get("sigma_epsilon") if isinstance(d, dict) else None))
        parts = []
        if z is not None:
            parts.append(f"z={_fmt_float(z, digits=3)}")
        if eps is not None and eps_sig is not None:
            parts.append(f"ε={_fmt_sci(eps, digits=2)}±{_fmt_sci(eps_sig, digits=2)}")
        parts.append(f"source={path_clock.name}")
        rows.append(
            TableRow(
                topic="量子時計（赤方偏移）",
                observable="偏差 ε（GR=0）",
                data="光格子時計（chronometric leveling）",
                n=None,
                reference="ε=0（GR）",
                pmodel="ε=0（弱場）",
                metric=" / ".join(parts),
            )
        )

    return rows


def _load_quantum_matter_wave_rows(root: Path) -> List[TableRow]:
    rows: List[TableRow] = []

    path_ds = _OUT_PUBLIC / "quantum" / "electron_double_slit_interference_metrics.json"
    if path_ds.exists():
        j = _read_json(path_ds)
        d = j.get("derived") or {}
        lam_pm = _safe_float((d.get("electron_wavelength_pm") if isinstance(d, dict) else None))
        fringe_mrad = _safe_float((d.get("fringe_spacing_theta_mrad") if isinstance(d, dict) else None))
        parts: List[str] = []
        if lam_pm is not None:
            parts.append(f"λ={_fmt_float(lam_pm, digits=3)} pm")
        if fringe_mrad is not None:
            parts.append(f"縞間隔={_fmt_float(fringe_mrad, digits=3)} mrad")
        parts.append(f"source={path_ds.name}")
        rows.append(
            TableRow(
                topic="物質波干渉",
                observable="de Broglie（電子二重スリット）",
                data="600 eV（代表値）",
                n=None,
                reference="λ=h/p",
                pmodel="同（波の位相条件）",
                metric=" / ".join(parts),
            )
        )

    path_alpha = _OUT_PUBLIC / "quantum" / "de_broglie_precision_alpha_consistency_metrics.json"
    if path_alpha.exists():
        j = _read_json(path_alpha)
        d = j.get("derived") or {}
        z = _safe_float((d.get("z_score") if isinstance(d, dict) else None))
        eps_mu = _safe_float((d.get("epsilon_mc_mean") if isinstance(d, dict) else None))
        eps_sig = _safe_float((d.get("epsilon_mc_sigma") if isinstance(d, dict) else None))
        parts = []
        if z is not None:
            parts.append(f"z={_fmt_float(z, digits=3)}")
        if eps_mu is not None and eps_sig is not None:
            parts.append(f"ε={_fmt_sci(eps_mu, digits=2)}±{_fmt_sci(eps_sig, digits=2)}（解釈パラメータ）")
        parts.append(f"source={path_alpha.name}")
        rows.append(
            TableRow(
                topic="物質波干渉（精密）",
                observable="de Broglie 精密（α整合）",
                data="recoil（Rb） vs g-2（e−）",
                n=None,
                reference="整合（z=0が理想）",
                pmodel="整合が必要（入口）",
                metric=" / ".join(parts),
            )
        )

    return rows


def _load_quantum_decoherence_rows(root: Path) -> List[TableRow]:
    path = _OUT_PUBLIC / "quantum" / "gravity_induced_decoherence_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)
    derived = j.get("derived") or {}
    ensemble = derived.get("ensemble") if isinstance(derived, dict) else None
    t_parts: List[str] = []
    if isinstance(ensemble, list):
        # Expect a small list: sigma_z_m, t_half_s.
        for item in ensemble:
            if not isinstance(item, dict):
                continue
            sz = _safe_float(item.get("sigma_z_m"))
            th = _safe_float(item.get("t_half_s"))
            if sz is None or th is None:
                continue
            t_parts.append(f"{_fmt_float(sz * 1e3, digits=3)} mm→{_fmt_sci(th, digits=2)} s")

    metric = ""
    if t_parts:
        metric = f"V=0.5: σz→t1/2 = {' / '.join(t_parts)}"
    metric = (metric + " / " if metric else "") + f"source={path.name}"

    return [
        TableRow(
            topic="重力誘起デコヒーレンス",
            observable="可視度Vの低下スケール",
            data="高さ分布 σz（例）",
            n=None,
            reference="time dilation dephasing（文献モデル）",
            pmodel="追加時間構造ノイズ σ_y の必要条件（入口）",
            metric=metric,
        )
    ]


def _load_quantum_photon_interference_rows(root: Path) -> List[TableRow]:
    path = _OUT_PUBLIC / "quantum" / "photon_quantum_interference_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)

    spi = j.get("single_photon_interference") or {}
    hom = j.get("hom") or {}
    sq = j.get("squeezing") or {}

    sigma_l_nm = _safe_float(spi.get("sigma_path_nm_from_visibility") if isinstance(spi, dict) else None)
    eta = _safe_float(sq.get("eta_lower_if_perfect_intrinsic") if isinstance(sq, dict) else None)

    parts: List[str] = []
    if sigma_l_nm is not None:
        parts.append(f"単一光子: σL≈{_fmt_float(sigma_l_nm, digits=3)} nm")

    if isinstance(hom, dict):
        v = hom.get("visibility")
        ep = hom.get("visibility_err_plus")
        em = hom.get("visibility_err_minus")
        d_ns = hom.get("d_ns")
        if isinstance(v, list) and isinstance(ep, list) and isinstance(em, list) and isinstance(d_ns, list):
            vv: List[str] = []
            for di, vi, epi, emi in zip(d_ns, v, ep, em):
                di_f = _safe_float(di)
                vi_f = _safe_float(vi)
                epi_f = _safe_float(epi)
                emi_f = _safe_float(emi)
                if di_f is None or vi_f is None:
                    continue
                if epi_f is not None and emi_f is not None:
                    vv.append(
                        f"D={_fmt_float(di_f, digits=3)} ns: V={_fmt_float(vi_f, digits=4)}(+{_fmt_float(epi_f, digits=4)}/-{_fmt_float(emi_f, digits=4)})"
                    )
                elif epi_f is not None:
                    vv.append(
                        f"D={_fmt_float(di_f, digits=3)} ns: V={_fmt_float(vi_f, digits=4)}±{_fmt_float(epi_f, digits=4)}"
                    )
                else:
                    vv.append(f"D={_fmt_float(di_f, digits=3)} ns: V={_fmt_float(vi_f, digits=4)}")
            if vv:
                parts.append("HOM: " + "; ".join(vv))

    if eta is not None:
        parts.append(f"スクイーズド（10 dB）: η≥{_fmt_float(eta, digits=3)}（loss-only）")

    parts.append(f"source={path.name}")

    return [
        TableRow(
            topic="光の量子干渉",
            observable="可視度/干渉指標（入口）",
            data="単一光子/HOM/スクイーズド光",
            n=None,
            reference="報告値（一次ソース）",
            pmodel="時間構造の制約へ写像（入口）",
            metric=" / ".join(parts),
        )
    ]


def _load_quantum_qed_vacuum_rows(root: Path) -> List[TableRow]:
    path = _OUT_PUBLIC / "quantum" / "qed_vacuum_precision_metrics.json"
    if not path.exists():
        return []
    j = _read_json(path)

    casimir = j.get("casimir") or {}
    r_m = _safe_float(casimir.get("sphere_radius_m") if isinstance(casimir, dict) else None)
    diameter_um = None if r_m is None else (2.0 * r_m * 1e6)

    sources = j.get("sources") if isinstance(j.get("sources"), list) else []
    rel_prec = None
    if isinstance(sources, list):
        for s in sources:
            if not isinstance(s, dict):
                continue
            av = s.get("abstract_value") if isinstance(s.get("abstract_value"), dict) else None
            if isinstance(av, dict):
                rp = _safe_float(av.get("relative_precision_at_closest_separation"))
                if rp is not None:
                    rel_prec = rp
                    break

    parts: List[str] = []
    if rel_prec is not None:
        parts.append(f"Casimir: closestで相対精度~{_fmt_pct(rel_prec, digits=2)}")
    if diameter_um is not None:
        parts.append(f"sphere diam={_fmt_float(diameter_um, digits=4)} μm")
    parts.append("Lamb: Z^4/Z^6 スケーリング固定（入口）")

    h_1s2s = j.get("hydrogen_1s2s") if isinstance(j.get("hydrogen_1s2s"), dict) else None
    if isinstance(h_1s2s, dict):
        f_raw = h_1s2s.get("f_hz")
        try:
            f_hz_int = int(f_raw)
        except Exception:
            f_hz_int = None
        sigma_hz = None
        try:
            sigma_hz = int(h_1s2s.get("sigma_hz"))
        except Exception:
            sigma_hz = None
        frac = _safe_float(h_1s2s.get("fractional_sigma"))
        if f_hz_int is not None and sigma_hz is not None:
            frac_s = "" if frac is None else f"（{frac:.2e}）"
            parts.append(f"H 1S–2S: f={f_hz_int:,} Hz (σ={sigma_hz} Hz){frac_s}")

    alpha_prec = j.get("alpha_precision") if isinstance(j.get("alpha_precision"), dict) else None
    if isinstance(alpha_prec, dict):
        derived = alpha_prec.get("derived") if isinstance(alpha_prec.get("derived"), dict) else None
        if isinstance(derived, dict):
            delta = _safe_float(derived.get("delta_alpha_inv"))
            sigma = _safe_float(derived.get("sigma_delta_alpha_inv"))
            z = _safe_float(derived.get("z_score"))
            eps = _safe_float(derived.get("epsilon_required"))
            if delta is not None and sigma is not None and z is not None:
                extra = "" if eps is None else f", ε≈{eps*1e9:+.2f} ppb"
                parts.append(f"α^-1: Δ={delta:+.3e}±{sigma:.3e} (z={z:+.2f}{extra})")
    parts.append(f"source={path.name}")

    return [
        TableRow(
            topic="真空・QED精密",
            observable="Casimir/Lamb の精度入口",
            data="一次ソース固定（初版）",
            n=None,
            reference="報告精度（一次ソース）",
            pmodel="現象論を再現できなければ棄却",
            metric=" / ".join(parts),
        )
    ]


def _load_quantum_atomic_molecular_rows(root: Path) -> List[TableRow]:
    out: List[TableRow] = []

    def build_row(*, path: Path, observable: str, data: str, name_map: Dict[str, str]) -> TableRow | None:
        if not path.exists():
            return None
        j = _read_json(path)
        rows = j.get("lines") if isinstance(j.get("lines"), list) else []
        if not isinstance(rows, list) or not rows:
            return None

        parts: List[str] = []
        n_lines = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            line_id = str(r.get("id") or "")
            nm = _safe_float(r.get("lambda_vac_nm"))
            if nm is None:
                continue
            label = name_map.get(line_id, line_id)
            parts.append(f"{label}={_fmt_float(nm, digits=3)} nm")
            n_lines += 1

        multiplets = j.get("multiplets") if isinstance(j.get("multiplets"), list) else []
        if isinstance(multiplets, list) and multiplets:
            # Keep the table concise: include only the most visible Balmer fine-structure multiplets.
            include = {"Hα", "Hβ"}
            for m in multiplets:
                if not isinstance(m, dict):
                    continue
                mid = str(m.get("id") or "")
                label = name_map.get(mid, mid)
                if label not in include:
                    continue
                n_raw = m.get("n_components")
                try:
                    n_comp = int(n_raw)
                except Exception:
                    continue
                if n_comp < 2:
                    continue
                mn = _safe_float(m.get("lambda_min_nm"))
                mx = _safe_float(m.get("lambda_max_nm"))
                if mn is None or mx is None:
                    continue
                parts.append(
                    f"{label} multiplet={_fmt_float(mn, digits=3)}–{_fmt_float(mx, digits=3)} nm (N={n_comp})"
                )

        if not parts:
            return None
        parts.append(f"source={path.name}")

        return TableRow(
            topic="原子・分子（基準値）",
            observable=observable,
            data=data,
            n=n_lines if n_lines > 0 else None,
            reference="NIST ASD（line output）",
            pmodel="基準値（ターゲット）を固定→再導出で棄却判定",
            metric=" / ".join(parts),
        )

    row_h = build_row(
        path=_OUT_PUBLIC / "quantum" / "atomic_hydrogen_baseline_metrics.json",
        observable="H I 遷移波長（vacuum）",
        data="NIST ASD（H I）",
        name_map={
            "H_I_Lyα": "Lyα",
            "H_I_Hα": "Hα",
            "H_I_Hβ": "Hβ",
            "H_I_Hγ": "Hγ",
        },
    )
    if row_h is not None:
        out.append(row_h)

    # Hydrogen ground-state hyperfine (21 cm) benchmark.
    path_hf = _OUT_PUBLIC / "quantum" / "atomic_hydrogen_hyperfine_baseline_metrics.json"
    if path_hf.exists():
        j_hf = _read_json(path_hf)
        hf = j_hf.get("hyperfine") if isinstance(j_hf.get("hyperfine"), dict) else None
        if isinstance(hf, dict):
            f_mhz = _safe_float(hf.get("frequency_mhz"))
            sigma_hz = _safe_float(hf.get("sigma_hz"))
            frac = _safe_float(hf.get("fractional_sigma"))
            wl_cm = _safe_float(hf.get("wavelength_cm"))
            parts: List[str] = []
            if f_mhz is not None:
                parts.append(f"f={f_mhz:.10f} MHz")
            if sigma_hz is not None:
                parts.append(f"σ={_fmt_sci(sigma_hz, digits=3)} Hz")
            if frac is not None:
                parts.append(f"frac={_fmt_sci(frac, digits=2)}")
            if wl_cm is not None:
                parts.append(f"λ={_fmt_float(wl_cm, digits=6)} cm")
            if parts:
                parts.append(f"source={path_hf.name}")
                out.append(
                    TableRow(
                        topic="原子・分子（基準値）",
                        observable="H I hyperfine（21 cm）",
                        data="NIST AtSpec（handbook PDF）",
                        n=1,
                        reference="NIST AtSpec（H 21 cm）",
                        pmodel="基準値（ターゲット）を固定→再導出で棄却判定",
                        metric=" / ".join(parts),
                    )
                )

    row_he = build_row(
        path=_OUT_PUBLIC / "quantum" / "atomic_helium_baseline_metrics.json",
        observable="He I 遷移波長（vacuum）",
        data="NIST ASD（He I）",
        name_map={
            "He_I_447.27nm": "447.273",
            "He_I_501.71nm": "501.708",
            "He_I_587.73nm": "587.725",
            "He_I_668.00nm": "668.000",
        },
    )
    if row_he is not None:
        out.append(row_he)

    def add_diatomic_row(*, slug: str, label: str) -> None:
        path = _OUT_PUBLIC / "quantum" / f"molecular_{slug}_baseline_metrics.json"
        if not path.exists():
            return
        j = _read_json(path)
        consts = j.get("constants") if isinstance(j.get("constants"), dict) else {}
        if not isinstance(consts, dict) or not consts:
            return

        omega_e = _safe_float(consts.get("omega_e_cm^-1"))
        omega_exe = _safe_float(consts.get("omega_e_x_e_cm^-1"))
        be = _safe_float(consts.get("B_e_cm^-1"))
        alpha_e = _safe_float(consts.get("alpha_e_cm^-1"))
        de = _safe_float(consts.get("D_e_cm^-1"))
        re_a = _safe_float(consts.get("r_e_A"))

        d0_e_ev = None
        derived = j.get("derived") if isinstance(j.get("derived"), dict) else None
        if isinstance(derived, dict):
            morse = derived.get("morse") if isinstance(derived.get("morse"), dict) else None
            if isinstance(morse, dict):
                d0_e_ev = _safe_float(morse.get("D0_eV"))

        parts: List[str] = []
        n_consts = 0
        if omega_e is not None:
            parts.append(f"ωe={_fmt_float(omega_e, digits=6)} cm^-1")
            n_consts += 1
        if omega_exe is not None:
            parts.append(f"ωexe={_fmt_float(omega_exe, digits=6)} cm^-1")
            n_consts += 1
        if be is not None:
            parts.append(f"Be={_fmt_float(be, digits=7)} cm^-1")
            n_consts += 1
        if alpha_e is not None:
            parts.append(f"αe={_fmt_float(alpha_e, digits=7)} cm^-1")
            n_consts += 1
        if de is not None:
            parts.append(f"D(dist)={_fmt_float(de, digits=7)} cm^-1")
            n_consts += 1
        if re_a is not None:
            parts.append(f"re={_fmt_float(re_a, digits=6)} Å")
            n_consts += 1
        if d0_e_ev is not None:
            parts.append(f"D0(Morse,derived)≈{_fmt_float(d0_e_ev, digits=4)} eV")
        parts.append(f"source={path.name}")

        out.append(
            TableRow(
                topic="原子・分子（基準値）",
                observable=f"{label} 分子定数（ground state）",
                data=f"NIST WebBook（{label}）",
                n=(n_consts if n_consts > 0 else None),
                reference="NIST Chemistry WebBook（Huber & Herzberg）",
                pmodel="基準値（ターゲット）を固定→再導出で棄却判定",
                metric=" / ".join(parts),
            )
        )

    add_diatomic_row(slug="h2", label="H2")
    add_diatomic_row(slug="hd", label="HD")
    add_diatomic_row(slug="d2", label="D2")

    # Dissociation enthalpy (298 K) from thermochemistry (independent baseline vs spectroscopic constants).
    path_diss = _OUT_PUBLIC / "quantum" / "molecular_dissociation_thermochemistry_metrics.json"
    if path_diss.exists():
        j = _read_json(path_diss)
        rows = j.get("rows") if isinstance(j.get("rows"), list) else []
        parts: List[str] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            sp = str(r.get("species") or "")
            ev = _safe_float(r.get("dissociation_eV_298K"))
            kj = _safe_float(r.get("dissociation_kj_per_mol_298K"))
            if not sp or ev is None or kj is None:
                continue
            parts.append(f"{sp}={_fmt_float(ev, digits=4)} eV ({_fmt_float(kj, digits=2)} kJ/mol)")
        if parts:
            parts.append(f"source={path_diss.name}")
            out.append(
                TableRow(
                    topic="原子・分子（基準値）",
                    observable="分子 解離エンタルピー（298 K）",
                    data="NIST WebBook（thermochemistry）",
                    n=len(parts) - 1,
                    reference="NIST WebBook（ΔfH°gas）",
                    pmodel="結合エネルギーの独立ベースライン（D0とは別）",
                    metric=" / ".join(parts),
                )
            )

    # Spectroscopic dissociation energy D0 (0 K; independent baseline vs 298 K thermochemistry).
    path_d0 = _OUT_PUBLIC / "quantum" / "molecular_dissociation_d0_spectroscopic_metrics.json"
    if path_d0.exists():
        j = _read_json(path_d0)
        rows = j.get("rows") if isinstance(j.get("rows"), list) else []
        parts: List[str] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            sp = str(r.get("molecule") or "")
            ev = _safe_float(r.get("d0_eV"))
            cm = _safe_float(r.get("d0_cm^-1"))
            n = r.get("rotational_N")
            n_i = None
            if n is not None:
                try:
                    n_i = int(n)
                except Exception:
                    n_i = None
            if not sp or ev is None or cm is None:
                continue
            tag = sp if n_i is None else f"{sp}(N={n_i})"
            parts.append(f"{tag}={_fmt_float(ev, digits=4)} eV ({_fmt_float(cm, digits=6)} cm^-1)")
        if parts:
            parts.append(f"source={path_d0.name}")
            out.append(
                TableRow(
                    topic="原子・分子（基準値）",
                    observable="分光学的解離エネルギー D0（0 K）",
                    data="高精度分光（ionization→D0）",
                    n=len(parts) - 1,
                    reference="arXiv（H2/D2）+ Phys. Rev. A 108, 022811（HD）",
                    pmodel="結合エネルギーの独立ベースライン（0 K；D0）",
                    metric=" / ".join(parts),
                )
            )

    # Molecular line lists: representative transitions (selected objectively as top-N by Einstein A).
    path_exomol = _OUT_PUBLIC / "quantum" / "molecular_transitions_exomol_baseline_metrics.json"
    if path_exomol.exists():
        j = _read_json(path_exomol)
        datasets = j.get("datasets") if isinstance(j.get("datasets"), list) else []
        rows = j.get("rows") if isinstance(j.get("rows"), list) else []

        parts: List[str] = []
        n_total = 0
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
            mol = str(ds.get("molecule") or "")
            tag = str(ds.get("dataset_tag") or "")
            top_n = ds.get("top_n")
            a_max = _safe_float(ds.get("A_max_s^-1"))
            nu_rng = ds.get("wavenumber_range_cm^-1")
            if not mol or not tag:
                continue
            try:
                top_n_i = int(top_n)
            except Exception:
                top_n_i = None
            n_total += (0 if top_n_i is None else top_n_i)
            nu_rng_str = None
            if isinstance(nu_rng, list) and len(nu_rng) == 2:
                n0 = _safe_float(nu_rng[0])
                n1 = _safe_float(nu_rng[1])
                if n0 is not None and n1 is not None:
                    nu_rng_str = f"{_fmt_float(n0, digits=6)}–{_fmt_float(n1, digits=6)} cm^-1"
            seg = f"{mol}({tag}): top{top_n_i}" if top_n_i is not None else f"{mol}({tag})"
            if a_max is not None:
                seg += f", A_max={_fmt_sci(a_max, digits=4)} s^-1"
            if nu_rng_str:
                seg += f", ν̃={nu_rng_str}"
            parts.append(seg)

        # Include a single representative transition per molecule (rank=1).
        rep_by_mol: dict[str, str] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            mol = str(r.get("molecule") or "")
            rank = r.get("rank_by_A_desc")
            if not mol:
                continue
            try:
                rank_i = int(rank)
            except Exception:
                continue
            if rank_i != 1:
                continue
            nu = _safe_float(r.get("wavenumber_cm^-1"))
            A = _safe_float(r.get("A_s^-1"))
            label = str(r.get("transition_label") or "")
            if nu is None or A is None or not label:
                continue
            rep_by_mol[mol] = f"{mol} #1: {label} (ν̃={_fmt_float(nu, digits=3)} cm^-1, A={_fmt_sci(A, digits=4)} s^-1)"

        rep_parts = [rep_by_mol[k] for k in sorted(rep_by_mol.keys())]
        if rep_parts:
            parts.extend(rep_parts)

        if parts:
            parts.append(f"source={path_exomol.name}")
            out.append(
                TableRow(
                    topic="原子・分子（基準値）",
                    observable="分子 遷移（一次線リスト；代表遷移）",
                    data="ExoMol + MOLAT",
                    n=(n_total if n_total > 0 else None),
                    reference="ExoMol database（H2: RACPPK / HD: ADJSAAM） + MOLAT（D2: ARLSJ1999）",
                    pmodel="基準値（ターゲット）を固定→再導出で棄却判定",
                    metric=" / ".join(parts),
                )
            )

    return out


def _load_quantum_nuclear_rows(root: Path) -> List[TableRow]:
    """
    Phase 7 / Step 7.9 nuclear baseline + minimal effective-equation summary.

    NOTE: Keep this row citation-key-free so that "データ出典/参考文献" stays filtered to citations
    actually used in the paper body.
    """
    path_canonical = _OUT_PUBLIC / "quantum" / "nuclear_effective_potential_canonical_metrics.json"
    path_barrier_tail_kq_v2t = (
        root
        / "output"
        / "quantum"
        / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_v2t_metrics.json"
    )
    path_barrier_tail_channel_split_triplet_barrier_fraction = (
        root
        / "output"
        / "quantum"
        / "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan_metrics.json"
    )
    path_barrier_tail_kq = (
        root
        / "output"
        / "quantum"
        / "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_metrics.json"
    )
    path_repulsive = _OUT_PUBLIC / "quantum" / "nuclear_effective_potential_repulsive_core_two_range_metrics.json"
    path_two_ar = _OUT_PUBLIC / "quantum" / "nuclear_effective_potential_two_range_fit_as_rs_metrics.json"
    path_two = _OUT_PUBLIC / "quantum" / "nuclear_effective_potential_two_range_metrics.json"
    path_finite = _OUT_PUBLIC / "quantum" / "nuclear_effective_potential_finite_core_well_metrics.json"
    path_core = _OUT_PUBLIC / "quantum" / "nuclear_effective_potential_core_well_metrics.json"
    path_sq = _OUT_PUBLIC / "quantum" / "nuclear_effective_potential_square_well_metrics.json"
    path_falsification_pack = _OUT_PUBLIC / "quantum" / "nuclear_binding_energy_frequency_mapping_falsification_pack.json"
    path_wave_interface = _OUT_PUBLIC / "quantum" / "nuclear_binding_energy_frequency_mapping_interface_metrics.json"
    path_wave_all_nuclei = _OUT_PUBLIC / "quantum" / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei_metrics.json"
    prefer_canonical = False
    if path_canonical.exists():
        try:
            j_can = _read_json(path_canonical)
        except Exception:
            j_can = {}
        within = j_can.get("falsification", {}).get("within_envelope")
        if isinstance(within, dict) and within.get("r_s") is True and within.get("v2s") is True:
            prefer_canonical = True
    prefer_barrier_tail_channel_split_triplet_barrier_fraction = False
    if path_barrier_tail_channel_split_triplet_barrier_fraction.exists():
        try:
            j_candidate = _read_json(path_barrier_tail_channel_split_triplet_barrier_fraction)
        except Exception:
            j_candidate = {}
        within = j_candidate.get("falsification", {}).get("within_envelope")
        if isinstance(within, dict) and within.get("r_s") is True and within.get("v2s") is True:
            prefer_barrier_tail_channel_split_triplet_barrier_fraction = True
    prefer_barrier_tail_kq_v2t = False
    if path_barrier_tail_kq_v2t.exists():
        try:
            j_v2t = _read_json(path_barrier_tail_kq_v2t)
        except Exception:
            j_v2t = {}
        within = j_v2t.get("falsification", {}).get("within_envelope")
        if isinstance(within, dict) and within.get("r_s") is True:
            prefer_barrier_tail_kq_v2t = True

    path = (
        path_canonical
        if prefer_canonical
        else (
            path_barrier_tail_channel_split_triplet_barrier_fraction
            if prefer_barrier_tail_channel_split_triplet_barrier_fraction
            else (
                path_barrier_tail_kq_v2t
                if prefer_barrier_tail_kq_v2t
                else (
                    path_barrier_tail_kq
                    if path_barrier_tail_kq.exists()
                    else (
                        path_repulsive
                        if path_repulsive.exists()
                        else (
                            path_two_ar
                            if path_two_ar.exists()
                            else (
                                path_two
                                if path_two.exists()
                                else (path_finite if path_finite.exists() else (path_core if path_core.exists() else path_sq))
                            )
                        )
                    )
                )
            )
        )
    )
    if not path.exists():
        return []

    j = _read_json(path)

    b_mev = None
    consts = j.get("constants") if isinstance(j.get("constants"), dict) else None
    if isinstance(consts, dict):
        b_mev = _safe_float(consts.get("B_MeV"))

    parts: List[str] = []
    if b_mev is not None:
        parts.append(f"B≈{_fmt_float(b_mev, digits=6)} MeV")

    def _append_barrier_tail_rows() -> tuple[Optional[bool], Optional[bool], Optional[bool]]:
        rows = j.get("results_by_dataset")
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("label") or "")

                ft = r.get("fit_triplet") if isinstance(r.get("fit_triplet"), dict) else {}
                geo = ft.get("geometry") if isinstance(ft.get("geometry"), dict) else {}
                ere_t = ft.get("ere") if isinstance(ft.get("ere"), dict) else {}

                fs = r.get("fit_singlet") if isinstance(r.get("fit_singlet"), dict) else {}
                ere_s = fs.get("ere") if isinstance(fs.get("ere"), dict) else {}

                cmp_ = r.get("comparison") if isinstance(r.get("comparison"), dict) else {}
                cmp_t = cmp_.get("triplet") if isinstance(cmp_.get("triplet"), dict) else {}
                cmp_s = cmp_.get("singlet") if isinstance(cmp_.get("singlet"), dict) else {}

                r1 = _safe_float(geo.get("R1_fm"))
                r2 = _safe_float(geo.get("R2_fm"))
                rb = _safe_float(geo.get("Rb_fm"))
                r3 = _safe_float(geo.get("R3_fm"))

                v1t = _safe_float(ft.get("V1_t_MeV"))
                v2t_mev = _safe_float(ft.get("V2_t_MeV"))
                vb_t = _safe_float(ft.get("V3_barrier_t_MeV"))
                vt_t = _safe_float(ft.get("V3_tail_t_MeV"))

                v1s_mev = _safe_float(fs.get("V1_s_MeV"))
                v2s_mev = _safe_float(fs.get("V2_s_MeV"))
                vb_s = _safe_float(fs.get("Vb_s_MeV"))
                vt_s = _safe_float(fs.get("Vt_s_MeV"))

                at = _safe_float(ere_t.get("a_fm"))
                dat = _safe_float(cmp_t.get("a_t_fit_minus_obs_fm"))
                rt = _safe_float(ere_t.get("r_eff_fm"))
                drt = _safe_float(cmp_t.get("r_t_fit_minus_obs_fm"))
                v2t = _safe_float(ere_t.get("v2_fm3"))
                dv2t = _safe_float(cmp_t.get("v2t_fit_minus_obs_fm3"))
                rs = _safe_float(ere_s.get("r_eff_fm"))
                drs = _safe_float(cmp_s.get("r_s_fit_minus_obs_fm"))
                v2s = _safe_float(ere_s.get("v2_fm3"))
                dv2s = _safe_float(cmp_s.get("v2s_pred_minus_obs_fm3"))

                s = label
                if r1 is not None and r2 is not None:
                    s += f": R1≈{_fmt_float(r1, digits=3)} fm, R2≈{_fmt_float(r2, digits=3)} fm"
                    if rb is not None and r3 is not None:
                        s += f", Rb≈{_fmt_float(rb, digits=3)} fm, R3≈{_fmt_float(r3, digits=3)} fm"
                delta_terms: List[str] = []
                if at is not None and dat is not None:
                    delta_terms.append(f"Δa_t≈{_fmt_float(dat, digits=4)} fm")
                if rt is not None and drt is not None:
                    delta_terms.append(f"Δr_t≈{_fmt_float(drt, digits=4)} fm")
                if v2t is not None and dv2t is not None:
                    delta_terms.append(f"Δv2t≈{_fmt_float(dv2t, digits=3)} fm^3")
                if rs is not None and drs is not None:
                    delta_terms.append(f"Δr_s≈{_fmt_float(drs, digits=4)} fm")
                if v2s is not None and dv2s is not None:
                    delta_terms.append(f"Δv2s≈{_fmt_float(dv2s, digits=3)} fm^3")
                if delta_terms:
                    s += ", " + ", ".join(delta_terms)

                parts.append(s.strip(" :"))

        rows = j.get("results_by_dataset")
        rs_ok = None
        v2s_ok = None
        v2t_ok = None
        if isinstance(rows, list):
            rs_obs: List[float] = []
            rs_fit: List[float] = []
            v2s_obs: List[float] = []
            v2s_pred: List[float] = []
            v2t_obs: List[float] = []
            v2t_fit: List[float] = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                inp = r.get("inputs") if isinstance(r.get("inputs"), dict) else {}
                inp_t = inp.get("triplet") if isinstance(inp.get("triplet"), dict) else {}
                inp_s = inp.get("singlet") if isinstance(inp.get("singlet"), dict) else {}
                ft = r.get("fit_triplet") if isinstance(r.get("fit_triplet"), dict) else {}
                fs = r.get("fit_singlet") if isinstance(r.get("fit_singlet"), dict) else {}
                ere_t = ft.get("ere") if isinstance(ft.get("ere"), dict) else {}
                ere_s = fs.get("ere") if isinstance(fs.get("ere"), dict) else {}

                rs_o = _safe_float(inp_s.get("r_s_fm"))
                rs_f = _safe_float(ere_s.get("r_eff_fm"))
                if rs_o is not None and rs_f is not None:
                    rs_obs.append(rs_o)
                    rs_fit.append(rs_f)

                v2s_o = _safe_float(inp_s.get("v2s_fm3"))
                v2s_p = _safe_float(ere_s.get("v2_fm3"))
                if v2s_o is not None and v2s_p is not None:
                    v2s_obs.append(v2s_o)
                    v2s_pred.append(v2s_p)

                v2t_o = _safe_float(inp_t.get("v2t_fm3"))
                v2t_f = _safe_float(ere_t.get("v2_fm3"))
                if v2t_o is not None and v2t_f is not None:
                    v2t_obs.append(v2t_o)
                    v2t_fit.append(v2t_f)

            if rs_obs and len(rs_fit) == len(rs_obs):
                rs_min = min(rs_obs)
                rs_max = max(rs_obs)
                rs_ok = all(rs_min <= x <= rs_max for x in rs_fit)
            if v2s_obs and len(v2s_pred) == len(v2s_obs):
                v2s_min = min(v2s_obs)
                v2s_max = max(v2s_obs)
                v2s_ok = all(v2s_min <= x <= v2s_max for x in v2s_pred)
            if v2t_obs and len(v2t_fit) == len(v2t_obs):
                v2t_min = min(v2t_obs)
                v2t_max = max(v2t_obs)
                v2t_ok = all(v2t_min <= x <= v2t_max for x in v2t_fit)

        return rs_ok, v2s_ok, v2t_ok

    if path.name in (
        "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan_metrics.json",
        "nuclear_effective_potential_canonical_metrics.json",
    ):
        scan = j.get("barrier_tail_channel_split_kq_triplet_barrier_fraction_scan", {})
        frozen_geo = scan.get("frozen_singlet_geometry_from_7_13_8_4", {})
        if isinstance(frozen_geo, dict):
            r1s = _safe_float(frozen_geo.get("r1_s_over_lambda_pi_pm_fixed"))
            r2s = _safe_float(frozen_geo.get("r2_s_over_lambda_pi_pm_fixed"))
            if r1s is not None and r2s is not None:
                parts.append(f"R1_s/λπ={_fmt_float(r1s, digits=3)}, R2_s/λπ={_fmt_float(r2s, digits=3)}")

        frozen_kq = scan.get("selected_channel_split_kq_from_7_13_8_4", {})
        if isinstance(frozen_kq, dict):
            ks = _safe_float(frozen_kq.get("k_s"))
            qs = _safe_float(frozen_kq.get("q_s"))
            qt = _safe_float(frozen_kq.get("q_t_fixed"))
            kt_prev = _safe_float(frozen_kq.get("k_t_prev"))
            if ks is not None and qs is not None:
                parts.append(f"(k_s,q_s)=({_fmt_float(ks, digits=3)}, {_fmt_float(qs, digits=3)})")
            if kt_prev is not None and qt is not None:
                parts.append(f"(k_t_prev,q_t)=({_fmt_float(kt_prev, digits=3)}, {_fmt_float(qt, digits=3)})")

        sel = scan.get("selected", {})
        if isinstance(sel, dict):
            frac_t = _safe_float(sel.get("barrier_len_fraction_t"))
            kt = _safe_float(sel.get("k_t"))
            if frac_t is not None and kt is not None:
                parts.append(f"triplet_scan: barrier_len_fraction_t={_fmt_float(frac_t, digits=3)}, k_t={_fmt_float(kt, digits=3)}")

        tol = scan.get("policy", {}).get("triplet_ar_tolerance", {})
        tol_a = _safe_float(tol.get("tol_a_fm")) if isinstance(tol, dict) else None
        tol_r = _safe_float(tol.get("tol_r_fm")) if isinstance(tol, dict) else None
        if tol_a is not None and tol_r is not None:
            parts.append(f"triplet_tol: |Δa_t|≤{_fmt_float(tol_a, digits=3)} fm, |Δr_t|≤{_fmt_float(tol_r, digits=3)} fm")

        rs_ok, v2s_ok, v2t_ok = _append_barrier_tail_rows()
        rs_s = "?" if not isinstance(rs_ok, bool) else ("yes" if rs_ok else "no")
        v2s_s = "?" if not isinstance(v2s_ok, bool) else ("yes" if v2s_ok else "no")
        v2t_s = "?" if not isinstance(v2t_ok, bool) else ("yes" if v2t_ok else "no")
        parts.append(f"envelope_ok(r_s)={rs_s}")
        parts.append(f"envelope_ok(v2s)={v2s_s}")
        parts.append(f"envelope_ok(v2t)={v2t_s}")

        triplet_tol_ok = None
        if tol_a is not None and tol_r is not None:
            rows = j.get("results_by_dataset")
            if isinstance(rows, list):
                ok_all = True
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    cmp_ = r.get("comparison") if isinstance(r.get("comparison"), dict) else {}
                    cmp_t = cmp_.get("triplet") if isinstance(cmp_.get("triplet"), dict) else {}
                    da = _safe_float(cmp_t.get("a_t_fit_minus_obs_fm"))
                    dr = _safe_float(cmp_t.get("r_t_fit_minus_obs_fm"))
                    if da is None or dr is None:
                        ok_all = False
                        break
                    if abs(da) > tol_a or abs(dr) > tol_r:
                        ok_all = False
                        break
                triplet_tol_ok = ok_all

        triplet_tol_s = "?" if not isinstance(triplet_tol_ok, bool) else ("yes" if triplet_tol_ok else "no")
        parts.append(f"tolerance_ok(triplet_ar)={triplet_tol_s}")

        pmodel = "u-profile → V(r)（λπ-constrained barrier+tail; channel-split (k,q) + singlet geometry split; triplet barrier fraction scan; envelope+tolerance checks）"
    elif path.name in (
        "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_metrics.json",
        "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan_v2t_metrics.json",
    ):
        barrier_sel = j.get("barrier_tail_kq_scan", {}).get("selected", {})
        k = _safe_float(barrier_sel.get("k")) if isinstance(barrier_sel, dict) else None
        q = _safe_float(barrier_sel.get("q")) if isinstance(barrier_sel, dict) else None
        if k is not None and q is not None:
            parts.append(f"(k,q)=({ _fmt_float(k, digits=3) }, { _fmt_float(q, digits=3) })")

        pion_scale = j.get("pion_range_scale", {})
        lambda_pi = _safe_float(pion_scale.get("lambda_pi_pm_fm")) if isinstance(pion_scale, dict) else None
        if lambda_pi is not None:
            parts.append(f"λπ≈{_fmt_float(lambda_pi, digits=4)} fm")

        rs_ok, v2s_ok, v2t_ok = _append_barrier_tail_rows()

        rs_s = "?" if not isinstance(rs_ok, bool) else ("yes" if rs_ok else "no")
        v2s_s = "?" if not isinstance(v2s_ok, bool) else ("yes" if v2s_ok else "no")
        v2t_s = "?" if not isinstance(v2t_ok, bool) else ("yes" if v2t_ok else "no")
        parts.append(f"envelope_ok(r_s)={rs_s}")
        parts.append(f"envelope_ok(v2s)={v2s_s}")
        parts.append(f"envelope_ok(v2t)={v2t_s}")

        pmodel = "u-profile → V(r)（λπ-constrained barrier+tail; global (k,q), fit triplet targets, fit singlet a_s,r_s, envelope check on v2t/v2s）"
    elif path.name in (
        "nuclear_effective_potential_repulsive_core_two_range_metrics.json",
        "nuclear_effective_potential_two_range_fit_as_rs_metrics.json",
    ):
        rows = j.get("results_by_dataset")
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("label") or "")

                ft = r.get("fit_triplet") if isinstance(r.get("fit_triplet"), dict) else {}
                geo = ft.get("geometry") if isinstance(ft.get("geometry"), dict) else {}
                ere_t = ft.get("ere") if isinstance(ft.get("ere"), dict) else {}

                fs = r.get("fit_singlet") if isinstance(r.get("fit_singlet"), dict) else {}
                ere_s = fs.get("ere") if isinstance(fs.get("ere"), dict) else {}

                cmp_ = r.get("comparison") if isinstance(r.get("comparison"), dict) else {}
                cmp_t = cmp_.get("triplet") if isinstance(cmp_.get("triplet"), dict) else {}
                cmp_s = cmp_.get("singlet") if isinstance(cmp_.get("singlet"), dict) else {}

                rc = _safe_float(geo.get("Rc_fm"))
                vc = _safe_float(geo.get("Vc_MeV"))
                r1 = _safe_float(geo.get("R1_fm"))
                r2 = _safe_float(geo.get("R2_fm"))
                v1t = _safe_float(ft.get("V1_t_MeV"))
                v2t_mev = _safe_float(ft.get("V2_t_MeV"))
                v1s_mev = _safe_float(fs.get("V1_s_MeV"))
                v2s_mev = _safe_float(fs.get("V2_s_MeV"))

                v2t = _safe_float(ere_t.get("v2_fm3"))
                dv2t = _safe_float(cmp_t.get("v2t_fit_minus_obs_fm3"))
                rs = _safe_float(ere_s.get("r_eff_fm"))
                drs = _safe_float(cmp_s.get("r_s_fit_minus_obs_fm"))
                v2s = _safe_float(ere_s.get("v2_fm3"))
                dv2s = _safe_float(cmp_s.get("v2s_pred_minus_obs_fm3"))

                s = label
                if rc is not None:
                    s += f": Rc≈{_fmt_float(rc, digits=3)} fm"
                    if vc is not None:
                        s += f", Vc≈{_fmt_float(vc, digits=1)} MeV"
                    if r1 is not None and r2 is not None:
                        s += f", R1≈{_fmt_float(r1, digits=3)} fm, R2≈{_fmt_float(r2, digits=3)} fm"
                elif r1 is not None and r2 is not None:
                    s += f": R1≈{_fmt_float(r1, digits=3)} fm, R2≈{_fmt_float(r2, digits=3)} fm"
                if v1t is not None and v2t_mev is not None:
                    s += f", V1t≈{_fmt_float(v1t, digits=1)} MeV, V2t≈{_fmt_float(v2t_mev, digits=1)} MeV"
                if v1s_mev is not None and v2s_mev is not None:
                    s += f", V1s≈{_fmt_float(v1s_mev, digits=1)} MeV, V2s≈{_fmt_float(v2s_mev, digits=3)} MeV"
                if v2t is not None and dv2t is not None:
                    s += f", v2t≈{_fmt_float(v2t, digits=3)} (Δ≈{_fmt_float(dv2t, digits=3)})"
                if rs is not None and drs is not None:
                    s += f", r_s(fit)≈{_fmt_float(rs, digits=4)} (Δ≈{_fmt_float(drs, digits=4)})"
                if v2s is not None and dv2s is not None:
                    s += f", v2s(pred)≈{_fmt_float(v2s, digits=3)} (Δ≈{_fmt_float(dv2s, digits=3)})"
                parts.append(s.strip(" :"))

        within = j.get("falsification", {}).get("within_envelope")
        rs_ok = None
        v2s_ok = None
        if isinstance(within, dict):
            rs_ok = within.get("r_s")
            v2s_ok = within.get("v2s")
        rs_s = "?" if not isinstance(rs_ok, bool) else ("yes" if rs_ok else "no")
        v2s_s = "?" if not isinstance(v2s_ok, bool) else ("yes" if v2s_ok else "no")
        parts.append(f"envelope_ok(r_s)={rs_s}")
        parts.append(f"envelope_ok(v2s)={v2s_s}")
        if path.name == "nuclear_effective_potential_repulsive_core_two_range_metrics.json":
            pmodel = "u-profile → V(r)（repulsive core + two-range; fit triplet targets, fit singlet a_s,r_s, predict v2s）"
        else:
            pmodel = "u-profile → V(r)（two-range; fit triplet targets, fit singlet a_s,r_s, predict v2s）"
    elif path.name == "nuclear_effective_potential_two_range_metrics.json":
        rows = j.get("results_by_dataset")
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("label") or "")

                ft = r.get("fit_triplet") if isinstance(r.get("fit_triplet"), dict) else {}
                geo = ft.get("geometry") if isinstance(ft.get("geometry"), dict) else {}
                ere_t = ft.get("ere") if isinstance(ft.get("ere"), dict) else {}

                fs = r.get("fit_singlet") if isinstance(r.get("fit_singlet"), dict) else {}
                ere_s = fs.get("ere") if isinstance(fs.get("ere"), dict) else {}

                cmp_ = r.get("comparison") if isinstance(r.get("comparison"), dict) else {}
                cmp_t = cmp_.get("triplet") if isinstance(cmp_.get("triplet"), dict) else {}
                cmp_s = cmp_.get("singlet") if isinstance(cmp_.get("singlet"), dict) else {}

                r1 = _safe_float(geo.get("R1_fm"))
                r2 = _safe_float(geo.get("R2_fm"))
                v1t = _safe_float(ft.get("V1_t_MeV"))
                v2t_mev = _safe_float(ft.get("V2_t_MeV"))
                v2s_mev = _safe_float(fs.get("V2_s_MeV"))

                v2t = _safe_float(ere_t.get("v2_fm3"))
                dv2t = _safe_float(cmp_t.get("v2t_fit_minus_obs_fm3"))
                rs = _safe_float(ere_s.get("r_eff_fm"))
                drs = _safe_float(cmp_s.get("r_s_pred_minus_obs_fm"))
                v2s = _safe_float(ere_s.get("v2_fm3"))
                dv2s = _safe_float(cmp_s.get("v2s_pred_minus_obs_fm3"))

                s = label
                if r1 is not None and r2 is not None:
                    s += f": R1≈{_fmt_float(r1, digits=3)} fm, R2≈{_fmt_float(r2, digits=3)} fm"
                if v1t is not None and v2t_mev is not None:
                    s += f", V1t≈{_fmt_float(v1t, digits=1)} MeV, V2t≈{_fmt_float(v2t_mev, digits=1)} MeV"
                if v2s_mev is not None:
                    s += f", V2s≈{_fmt_float(v2s_mev, digits=3)} MeV"
                if v2t is not None and dv2t is not None:
                    s += f", v2t≈{_fmt_float(v2t, digits=3)} (Δ≈{_fmt_float(dv2t, digits=3)})"
                if rs is not None and drs is not None:
                    s += f", r_s(pred)≈{_fmt_float(rs, digits=4)} (Δ≈{_fmt_float(drs, digits=4)})"
                if v2s is not None and dv2s is not None:
                    s += f", v2s(pred)≈{_fmt_float(v2s, digits=3)} (Δ≈{_fmt_float(dv2s, digits=3)})"
                parts.append(s.strip(" :"))

        within = j.get("falsification", {}).get("within_envelope")
        rs_ok = None
        v2s_ok = None
        if isinstance(within, dict):
            rs_ok = within.get("r_s")
            v2s_ok = within.get("v2s")
        rs_s = "?" if not isinstance(rs_ok, bool) else ("yes" if rs_ok else "no")
        v2s_s = "?" if not isinstance(v2s_ok, bool) else ("yes" if v2s_ok else "no")
        parts.append(f"envelope_ok(r_s)={rs_s}")
        parts.append(f"envelope_ok(v2s)={v2s_s}")
        pmodel = "u-profile → V(r)（two-range; fit triplet targets, fit singlet a_s, predict singlet）"
    elif path.name == "nuclear_effective_potential_finite_core_well_metrics.json":
        rows = j.get("results_by_dataset")
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("label") or "")

                ft = r.get("fit_triplet") if isinstance(r.get("fit_triplet"), dict) else {}
                geo = ft.get("geometry") if isinstance(ft.get("geometry"), dict) else {}
                ere_t = ft.get("ere") if isinstance(ft.get("ere"), dict) else {}

                fs = r.get("fit_singlet") if isinstance(r.get("fit_singlet"), dict) else {}
                ere_s = fs.get("ere") if isinstance(fs.get("ere"), dict) else {}

                cmp_ = r.get("comparison") if isinstance(r.get("comparison"), dict) else {}
                cmp_t = cmp_.get("triplet") if isinstance(cmp_.get("triplet"), dict) else {}
                cmp_s = cmp_.get("singlet") if isinstance(cmp_.get("singlet"), dict) else {}

                rc = _safe_float(geo.get("Rc_fm"))
                rout = _safe_float(geo.get("R_fm"))
                vc = _safe_float(ft.get("Vc_MeV"))
                v0t = _safe_float(ft.get("V0_t_MeV"))
                v0s = _safe_float(fs.get("V0_s_MeV"))

                v2t = _safe_float(ere_t.get("v2_fm3"))
                dv2t = _safe_float(cmp_t.get("v2t_fit_minus_obs_fm3"))
                rs = _safe_float(ere_s.get("r_eff_fm"))
                drs = _safe_float(cmp_s.get("r_s_pred_minus_obs_fm"))
                v2s = _safe_float(ere_s.get("v2_fm3"))
                dv2s = _safe_float(cmp_s.get("v2s_pred_minus_obs_fm3"))

                s = label
                if rc is not None and rout is not None:
                    s += f": Rc≈{_fmt_float(rc, digits=3)} fm, R≈{_fmt_float(rout, digits=3)} fm"
                if vc is not None:
                    s += f", Vc≈{_fmt_float(vc, digits=1)} MeV"
                if v0t is not None:
                    s += f", V0t≈{_fmt_float(v0t, digits=3)} MeV"
                if v0s is not None:
                    s += f", V0s≈{_fmt_float(v0s, digits=3)} MeV"
                if v2t is not None and dv2t is not None:
                    s += f", v2t≈{_fmt_float(v2t, digits=3)} (Δ≈{_fmt_float(dv2t, digits=3)})"
                if rs is not None and drs is not None:
                    s += f", r_s(pred)≈{_fmt_float(rs, digits=4)} (Δ≈{_fmt_float(drs, digits=4)})"
                if v2s is not None and dv2s is not None:
                    s += f", v2s(pred)≈{_fmt_float(v2s, digits=3)} (Δ≈{_fmt_float(dv2s, digits=3)})"
                parts.append(s.strip(" :"))

        within = j.get("falsification", {}).get("within_envelope")
        rs_ok = None
        v2s_ok = None
        if isinstance(within, dict):
            rs_ok = within.get("r_s")
            v2s_ok = within.get("v2s")
        rs_s = "?" if not isinstance(rs_ok, bool) else ("yes" if rs_ok else "no")
        v2s_s = "?" if not isinstance(v2s_ok, bool) else ("yes" if v2s_ok else "no")
        parts.append(f"envelope_ok(r_s)={rs_s}")
        parts.append(f"envelope_ok(v2s)={v2s_s}")
        pmodel = "u-profile → V(r)（finite core+well; fit triplet targets, predict singlet）"
    elif path.name == "nuclear_effective_potential_core_well_metrics.json":
        rows = j.get("results_by_dataset")
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                label = str(r.get("label") or "")
                ft = r.get("fit_triplet") if isinstance(r.get("fit_triplet"), dict) else {}
                geo = ft.get("geometry") if isinstance(ft.get("geometry"), dict) else {}
                ere_t = ft.get("ere") if isinstance(ft.get("ere"), dict) else {}
                fs = r.get("fit_singlet") if isinstance(r.get("fit_singlet"), dict) else {}
                ere_s = fs.get("ere") if isinstance(fs.get("ere"), dict) else {}
                cmp_ = r.get("comparison") if isinstance(r.get("comparison"), dict) else {}
                cmp_t = cmp_.get("triplet") if isinstance(cmp_.get("triplet"), dict) else {}
                cmp_s = cmp_.get("singlet") if isinstance(cmp_.get("singlet"), dict) else {}

                rc = _safe_float(geo.get("Rc_fm"))
                rout = _safe_float(geo.get("R_fm"))
                v0t = _safe_float(ft.get("V0_t_MeV"))
                v0s = _safe_float(fs.get("V0_s_MeV"))
                v2t = _safe_float(ere_t.get("v2_fm3"))
                dv2t = _safe_float(cmp_t.get("v2t_pred_minus_obs_fm3"))
                rs = _safe_float(ere_s.get("r_eff_fm"))
                drs = _safe_float(cmp_s.get("r_s_pred_minus_obs_fm"))
                v2s = _safe_float(ere_s.get("v2_fm3"))
                dv2s = _safe_float(cmp_s.get("v2s_pred_minus_obs_fm3"))

                s = label
                if rc is not None and rout is not None:
                    s += f": Rc≈{_fmt_float(rc, digits=3)} fm, R≈{_fmt_float(rout, digits=3)} fm"
                if v0t is not None:
                    s += f", V0t≈{_fmt_float(v0t, digits=3)} MeV"
                if v0s is not None:
                    s += f", V0s≈{_fmt_float(v0s, digits=3)} MeV"
                if v2t is not None and dv2t is not None:
                    s += f", v2t(pred)≈{_fmt_float(v2t, digits=3)} (Δ≈{_fmt_float(dv2t, digits=3)})"
                if rs is not None and drs is not None:
                    s += f", r_s(pred)≈{_fmt_float(rs, digits=4)} (Δ≈{_fmt_float(drs, digits=4)})"
                if v2s is not None and dv2s is not None:
                    s += f", v2s(pred)≈{_fmt_float(v2s, digits=3)} (Δ≈{_fmt_float(dv2s, digits=3)})"
                parts.append(s.strip(" :"))

        within = j.get("falsification", {}).get("within_envelope")
        v2t_ok = None
        v2s_ok = None
        if isinstance(within, dict):
            v2t_ok = within.get("v2t")
            v2s_ok = within.get("v2s")
        v2t_s = "?" if not isinstance(v2t_ok, bool) else ("yes" if v2t_ok else "no")
        v2s_s = "?" if not isinstance(v2s_ok, bool) else ("yes" if v2s_ok else "no")
        parts.append(f"envelope_ok(v2t)={v2t_s}")
        parts.append(f"envelope_ok(v2s)={v2s_s}")
        pmodel = "u-profile → V(r)（core+well; try fit triplet, predict v2 & singlet）"
    else:
        # Step 7.9.3 fallback (square well).
        fits = j.get("fits")
        if not isinstance(fits, list) or not fits:
            return []
        for f in fits:
            if not isinstance(f, dict):
                continue
            label = str(f.get("label") or "")
            fsw = f.get("fit_square_well") if isinstance(f.get("fit_square_well"), dict) else {}
            ere = f.get("ere_from_phase_shift") if isinstance(f.get("ere_from_phase_shift"), dict) else {}
            cmp_ = f.get("comparison") if isinstance(f.get("comparison"), dict) else {}
            r_fm = _safe_float(fsw.get("R_fm"))
            v0_mev = _safe_float(fsw.get("V0_mev"))
            r_eff = _safe_float(ere.get("r_eff_fm"))
            dr = _safe_float(cmp_.get("r_eff_minus_observed_fm"))
            s = label
            if r_fm is not None and v0_mev is not None:
                s += f": R≈{_fmt_float(r_fm, digits=3)} fm, V0≈{_fmt_float(v0_mev, digits=3)} MeV"
            if r_eff is not None and dr is not None:
                s += f", r_t(pred)≈{_fmt_float(r_eff, digits=4)} (Δ≈{_fmt_float(dr, digits=4)})"
            parts.append(s.strip(" :"))
        within = j.get("falsification", {}).get("predicted_within_envelope")
        within_s = "?" if not isinstance(within, bool) else ("yes" if within else "no")
        parts.append(f"envelope_ok={within_s}")
        pmodel = "u-profile → V(r)（square well; fit B,a_t）"

    if path_falsification_pack.exists():
        try:
            j_pack = _read_json(path_falsification_pack)
        except Exception:
            j_pack = {}
        channels = j_pack.get("differential_channels") if isinstance(j_pack.get("differential_channels"), list) else []
        channel_summaries: List[str] = []
        for c in channels:
            if not isinstance(c, dict):
                continue
            label = str(c.get("channel_id") or "").strip()
            abs_stats = c.get("abs_delta_mev_stats") if isinstance(c.get("abs_delta_mev_stats"), dict) else {}
            rel_stats = c.get("required_relative_sigma_3sigma") if isinstance(c.get("required_relative_sigma_3sigma"), dict) else {}
            med_abs = _safe_float(abs_stats.get("median"))
            med_rel = _safe_float(rel_stats.get("median"))
            if label and med_abs is not None and med_rel is not None:
                channel_summaries.append(
                    f"{label}: median abs(ΔB)≈{_fmt_float(med_abs, digits=2)} MeV, median σ_req,rel≈{_fmt_float(100.0 * med_rel, digits=2)}%"
                )
        if channel_summaries:
            parts.append(" / ".join(channel_summaries))

    if path_wave_interface.exists():
        try:
            j_if = _read_json(path_wave_interface)
        except Exception:
            j_if = {}
        rows_if = j_if.get("rows") if isinstance(j_if.get("rows"), list) else []
        row_d = None
        for rr in rows_if:
            if isinstance(rr, dict) and str(rr.get("label") or "").strip().lower() == "deuteron":
                row_d = rr
                break
        if isinstance(row_d, dict):
            be = _safe_float(row_d.get("B_MeV"))
            dmm = _safe_float(row_d.get("Delta_omega_over_omega0_per_nucleon"))
            if be is not None and dmm is not None:
                parts.append(
                    f"wave-I/F(deuteron): B≈{_fmt_float(be, digits=6)} MeV, Δm/m=Δω/ω≈{_fmt_float(dmm, digits=6)}"
                )
        parts.append("wave-I/F mapping: B=ħΔω, Δm=ħΔω/c^2")

    if path_wave_all_nuclei.exists():
        try:
            j_all = _read_json(path_wave_all_nuclei)
        except Exception:
            j_all = {}
        stats = j_all.get("stats") if isinstance(j_all.get("stats"), dict) else {}
        cr_all = stats.get("collective_ratio_all") if isinstance(stats.get("collective_ratio_all"), dict) else {}
        n_all_f = _safe_float(cr_all.get("n"))
        n_all = int(round(n_all_f)) if n_all_f is not None else None
        med_all = _safe_float(cr_all.get("median"))
        p16_all = _safe_float(cr_all.get("p16"))
        p84_all = _safe_float(cr_all.get("p84"))
        if n_all is not None and med_all is not None and p16_all is not None and p84_all is not None:
            parts.append(
                f"AME2020(all): N={n_all}, median(B_pred/B_obs)≈{_fmt_float(med_all, digits=3)}, p16–p84≈{_fmt_float(p16_all, digits=3)}–{_fmt_float(p84_all, digits=3)}"
            )

    if path_falsification_pack.exists():
        try:
            j_pack = _read_json(path_falsification_pack)
        except Exception:
            j_pack = {}
        models = j_pack.get("models") if isinstance(j_pack.get("models"), list) else []
        model_summaries: List[str] = []
        for mm in models:
            if not isinstance(mm, dict):
                continue
            model_id = str(mm.get("model_id") or "")
            z_med = _safe_float(mm.get("z_median"))
            z_dmed = _safe_float(mm.get("z_delta_median"))
            if z_med is None or z_dmed is None:
                continue
            if "local_spacing" in model_id:
                model_summaries.append(
                    f"local-spacing: z_median≈{_fmt_float(z_med, digits=2)}, z_ΔA≈{_fmt_float(z_dmed, digits=2)}"
                )
            elif "global_R" in model_id:
                model_summaries.append(
                    f"global-R: z_median≈{_fmt_float(z_med, digits=2)}, z_ΔA≈{_fmt_float(z_dmed, digits=2)}"
                )
        if model_summaries:
            parts.append("falsification: " + " / ".join(model_summaries))

    parts.append(f"source={path.name}")

    return [
        TableRow(
            topic="原子核（波動干渉＋deuteron+np散乱）",
            observable="結合エネルギー B.E. + (a_t,r_t,v2)（低エネルギー）",
            data="AME2020（全核種） + CODATA（質量欠損） + np散乱（eq18–19）",
            n=None,
            reference="観測値（一次）",
            pmodel=f"{pmodel} + 波動干渉写像（B=ħΔω, Δm/m=Δω/ω）",
            metric=" / ".join([p for p in parts if p]),
        )
    ]


def _load_quantum_planned_rows(root: Path) -> List[TableRow]:
    # NOTE: Planned rows are included even if the corresponding outputs are not yet available.
    # Avoid citation keys here to keep "データ出典/参考文献" filtered to actually-used citations.
    _ = root  # reserved (future: link to roadmap/outputs if needed)
    return [
        TableRow(
            topic="Bell（計画）",
            observable="photon time-tag 追加データ統合",
            data="Giustina 2015 等（一次公開所在の固定）",
            n=None,
            reference="TBD",
            pmodel="同一I/Fで再解析→統合",
            metric="入手できたものから統合（追加time-tag）",
        ),
        TableRow(
            topic="Bell（計画）",
            observable="共分散＋系統分解＋長期統合＋反証条件",
            data="NIST/Weihs/Delft（横断）",
            n=None,
            reference="TBD",
            pmodel="falsification pack（閾値固定）",
            metric="共分散/系統/長期整合を固定（横断）",
        ),
        TableRow(
            topic="量子（計画）",
            observable="干渉・時計・QED の精密化",
            data="追加一次データ（TBD）",
            n=None,
            reference="TBD",
            pmodel="Part I の凍結値（β）で比較",
            metric="Part II 同様に、共分散＋系統分解＋cross-check＋棄却条件を固定出力",
        ),
    ]


def build_table1_rows(root: Path) -> List[TableRow]:
    rows: List[TableRow] = []
    rows.extend(_load_llr_rows(root))
    rows.extend(_load_cassini_rows(root))
    rows.extend(_load_viking_rows(root))
    rows.extend(_load_mercury_rows(root))
    rows.extend(_load_gps_rows(root))
    rows.extend(_load_solar_light_deflection_rows(root))
    rows.extend(_load_gravitational_redshift_rows(root))
    rows.extend(_load_cosmology_distance_duality_rows(root))
    rows.extend(_load_cosmology_tolman_surface_brightness_rows(root))
    rows.extend(_load_cosmology_independent_probe_rows(root))
    rows.extend(_load_cosmology_jwst_mast_rows(root))
    rows.extend(_load_xrism_rows(root))
    rows.extend(_load_cosmology_bao_primary_rows(root))
    rows.extend(_load_cosmology_cmb_polarization_phase_rows(root))
    rows.extend(_load_cosmology_cmb_acoustic_peak_rows(root))
    rows.extend(_load_cosmology_fsigma8_growth_rows(root))
    rows.extend(_load_cosmology_cluster_collision_rows(root))
    rows.extend(_load_frame_dragging_rows(root))
    rows.extend(_load_gw_polarization_rows(root))
    rows.extend(_load_background_metric_choice_rows(root))
    rows.extend(_load_eht_rows(root))
    rows.extend(_load_sparc_rotation_rows(root))
    rows.extend(_load_pulsar_rows(root))
    rows.extend(_load_gw_rows(root))
    rows.extend(_load_gw250114_rows(root))
    rows.extend(_load_strong_field_higher_order_rows(root))
    rows.extend(_load_bbn_rows(root))
    rows.extend(_load_delta_rows(root))
    return rows


def build_table1_quantum_rows(root: Path) -> List[TableRow]:
    rows: List[TableRow] = []
    rows.extend(_load_quantum_bell_rows(root))
    rows.extend(_load_quantum_gravity_quantum_interference_rows(root))
    rows.extend(_load_quantum_matter_wave_rows(root))
    rows.extend(_load_quantum_decoherence_rows(root))
    rows.extend(_load_quantum_photon_interference_rows(root))
    rows.extend(_load_quantum_qed_vacuum_rows(root))
    rows.extend(_load_quantum_atomic_molecular_rows(root))
    rows.extend(_load_quantum_nuclear_rows(root))
    rows.extend(_load_quantum_planned_rows(root))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paper tables from fixed outputs.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default: output/private/summary).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    out_dir = Path(args.out_dir) if args.out_dir else (root / "output" / "private" / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_table1_rows(root)
    rows_quantum = build_table1_quantum_rows(root)
    frozen_note = "βの既定値は 1.0（PPN: (1+γ)=2β）とする。"
    frozen_path = _OUT_PRIVATE / "theory" / "frozen_parameters.json"
    if frozen_path.exists():
        try:
            fj = _read_json(frozen_path)
            beta = _safe_float(fj.get("beta"))
            beta_sig = _safe_float(fj.get("beta_sigma"))
            if beta is not None:
                frozen_note = f"βは Cassini 2003 の PPN γ から凍結し、β={_fmt_float(beta, digits=7)}"
                if beta_sig is not None:
                    frozen_note += f"±{_fmt_float(beta_sig, digits=3)}"
                frozen_note += " を以後固定する（PPN: (1+γ)=2β）。"
        except Exception:
            pass
    notes = [
        "本表は、各テーマで固定した最終集計値から集計し、再計算は行わない。",
        frozen_note,
        "光偏向（太陽）の γ は VLBI 等で推定された PPNパラメータ（一次ソース）で、P-model は γ=2β-1 で対応づける。",
        "重力赤方偏移の ε は z_obs=(1+ε)ΔU/c^2 の偏差パラメータ（GRとP-model弱場予測は ε=0）。",
        "距離二重性（DDR）は公表制約の形式 `D_L=(1+z)^(2+ε0)D_A` に基づくが、この(1+z)^2にはフラックス側(1+z)と膨張側 `D_A=D_M/(1+z)` が混在する。本表では P-model固有指標 `η^(P)=D_L/((1+z)D_A)`（p=1+ε0）を併記し、距離推定I/F監査を前提に差のスケールを示す（直ちに“物理棄却”とは解釈しない）。",
        "Tolman表面輝度の指数 n は SB∝(1+z)^-n で表す（FRW: n=4、静的背景P最小: n=2）。観測から推定される n には銀河進化（系統）が混入するため、ここでは一次ソースの n を固定入力として扱い、差の符号/スケール（zスコア等）を目安として示す。",
        "宇宙論（独立プローブ）の p_t（SN time dilation）と p_T（CMB T(z)）は距離指標と独立であり、背景Pの最小予測（p_t=1, p_T=1）と整合する。",
        "宇宙論（BAO一次統計）の ε は AP warping パラメータ。銀河+randomから ξ0/ξ2 を再計算し、(i) post-recon は Ross 2016 full covariance（dv=[ξ0,ξ2]）、(ii) pre-recon クロスチェックは Satpathy 2016 full covariance を用いて smooth+peak peakfit で ε を評価する。加えて P(k) multipoles（Beutler et al.）の peakfit（窓関数込み）でも ε をクロスチェックする。",
        "EHTは「リング直径~影直径」の近似で比較している（放射モデル/散乱/スピン依存は別途）。",
        "EHTの指標欄には (i) κ=1仮定の整合性 z(P), (ii) κ_fit（リング/シャドウ比）, (iii) 係数差(P/GR) を併記する。",
        "二重パルサーとGW150914は、動的現象（放射）に対する整合性チェックとして、四重極則に基づく観測量（Pdot_b と chirp）との一致度を示す（双極放射などの追加項は上限として評価）。",
        "GW150914のchirp指標は、公開strainから抽出した f(t) を Newton近似の四重極チャープ則（t=t_c−A f^{-8/3}）へ当てはめた簡易指標であり、公式テンプレート解析の代替ではない。",
    ]
    notes_quantum = [
        "本表は Part III（量子物理編）の検証サマリである。",
        "Bell は selection（coincidence window / event-ready window 等）が統計母集団を変え得る入口の定量化を目的とする。",
        "「計画」行は未実施の追加テスト（TBD）を示す。",
    ]

    payload = {
        "generated_utc": _iso_utc_now(),
        "table1": {
            "title": "検証サマリ（Table 1）",
            "rows": [r.to_dict() for r in rows],
            "notes": notes,
        },
    }

    json_path = out_dir / "paper_table1_results.json"
    csv_path = out_dir / "paper_table1_results.csv"
    md_path = out_dir / "paper_table1_results.md"
    json_path_q = out_dir / "paper_table1_quantum_results.json"
    csv_path_q = out_dir / "paper_table1_quantum_results.csv"
    md_path_q = out_dir / "paper_table1_quantum_results.md"

    _write_json(json_path, payload)
    _write_csv(csv_path, rows)
    _write_markdown(md_path, title="検証サマリ（Table 1）", rows=rows, notes=notes)

    payload_q = {
        "generated_utc": payload["generated_utc"],
        "table1": {
            "title": "検証サマリ（Table 1）",
            "rows": [r.to_dict() for r in rows_quantum],
            "notes": notes_quantum,
        },
    }
    _write_json(json_path_q, payload_q)
    _write_csv(csv_path_q, rows_quantum)
    _write_markdown(md_path_q, title="検証サマリ（Table 1）", rows=rows_quantum, notes=notes_quantum)

    try:
        worklog.append_event(
            {
                "event_type": "paper_tables",
                "argv": list(argv) if argv is not None else None,
                "outputs": {
                    "table1_json": json_path,
                    "table1_csv": csv_path,
                    "table1_md": md_path,
                    "table1_quantum_json": json_path_q,
                    "table1_quantum_csv": csv_path_q,
                    "table1_quantum_md": md_path_q,
                },
            }
        )
    except Exception:
        pass

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path_q}")
    print(f"Wrote: {csv_path_q}")
    print(f"Wrote: {md_path_q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
