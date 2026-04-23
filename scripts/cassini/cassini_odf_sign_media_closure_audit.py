#!/usr/bin/env python3
"""Cassini ODF sign/media closure audit for absolute-beta promotion.

Purpose:
- Execute sign-convention candidate fits on ODF raw direct-fit points.
- Quantify convergence against TDF branches under the same ±10 day window.
- Audit whether record-level media-correction flags are observable from ODF labels.

Inputs:
- output/cassini/cassini_odf_beta_direct_fit_points.csv
- output/cassini/cassini_beta_direct_fit_cross_source_metrics.json
- output/cassini/cassini_odf_raw_if_manifest.json
- data/cassini/pds_sce1/**/odf/*.lbl

Outputs:
- output/cassini/cassini_odf_sign_media_closure_audit.json
- output/cassini/cassini_odf_sign_media_closure_audit.csv
- synced copies in output/public/cassini/
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# クラス: `FitResult` の責務と境界条件を定義する。
@dataclass(frozen=True)
class FitResult:
    candidate: str
    fit_intercept: bool
    delta_beta: float
    delta_beta_sigma_proxy: float
    intercept: float
    corr_obs_fit: float
    rmse: float
    weighted_rms: float
    n_points: int
    zscore_vs_tdf_plasma: Optional[float]
    zscore_vs_tdf_raw: Optional[float]
    zscore_min_vs_tdf: Optional[float]


# 関数: `_repo_root` の入出力契約と処理意図を定義する。

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(value: object) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None

    if not math.isfinite(v):
        return None

    return v


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


# 関数: `_load_points` の入出力契約と処理意図を定義する。

def _load_points(path: Path) -> Tuple[List[float], List[float], List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing points CSV: {path}")

    ys: List[float] = []
    xs: List[float] = []
    ws: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            y = _safe_float(r.get("y_obs"))
            x = _safe_float(r.get("y_unit_beta1"))
            w = _safe_float(r.get("weight"))
            if y is None or x is None:
                continue

            if w is None or w <= 0.0:
                w = 1.0

            ys.append(float(y))
            xs.append(float(x))
            ws.append(float(w))

    if not ys:
        raise RuntimeError(f"No valid rows in points CSV: {path}")

    return ys, xs, ws


# 関数: `_weighted_fit` の入出力契約と処理意図を定義する。

def _weighted_fit(
    y_obs: Sequence[float],
    x_unit: Sequence[float],
    weights: Sequence[float],
    *,
    fit_intercept: bool,
) -> Tuple[float, float, float, float, float, float]:
    n = len(y_obs)
    if n <= 1 or n != len(x_unit) or n != len(weights):
        raise ValueError("Invalid input lengths for weighted fit")

    if fit_intercept:
        s00 = sum(weights)
        s01 = sum(w * x for w, x in zip(weights, x_unit))
        s11 = sum(w * x * x for w, x in zip(weights, x_unit))
        b0 = sum(w * y for w, y in zip(weights, y_obs))
        b1 = sum(w * x * y for w, x, y in zip(weights, x_unit, y_obs))
        det = s00 * s11 - s01 * s01
        if abs(det) <= 0.0:
            raise RuntimeError("Singular normal matrix in weighted fit")

        intercept = (b0 * s11 - b1 * s01) / det
        delta = (s00 * b1 - s01 * b0) / det
        inv11 = s00 / det
        dof = max(1, n - 2)
    else:
        s11 = sum(w * x * x for w, x in zip(weights, x_unit))
        b1 = sum(w * x * y for w, x, y in zip(weights, x_unit, y_obs))
        if abs(s11) <= 0.0:
            raise RuntimeError("Singular denominator in zero-intercept fit")

        intercept = 0.0
        delta = b1 / s11
        inv11 = 1.0 / s11
        dof = max(1, n - 1)

    y_fit = [intercept + delta * x for x in x_unit]
    residuals = [y - yf for y, yf in zip(y_obs, y_fit)]
    wrss = sum(w * r * r for w, r in zip(weights, residuals))
    sigma2 = wrss / float(dof)
    delta_sigma = math.sqrt(max(0.0, sigma2 * inv11))
    rmse = math.sqrt(sum(r * r for r in residuals) / float(n))
    weighted_rms = math.sqrt(wrss / sum(weights))

    y_mean = sum(y_obs) / float(n)
    f_mean = sum(y_fit) / float(n)
    num = sum((y - y_mean) * (f - f_mean) for y, f in zip(y_obs, y_fit))
    den_y = sum((y - y_mean) ** 2 for y in y_obs)
    den_f = sum((f - f_mean) ** 2 for f in y_fit)
    corr = num / math.sqrt(den_y * den_f) if den_y > 0.0 and den_f > 0.0 else math.nan

    return float(delta), float(delta_sigma), float(intercept), float(corr), float(rmse), float(weighted_rms)


# 関数: `_zscore` の入出力契約と処理意図を定義する。

def _zscore(delta_a: float, sigma_a: float, delta_b: float, sigma_b: float) -> Optional[float]:
    if not (math.isfinite(delta_a) and math.isfinite(delta_b) and sigma_a > 0.0 and sigma_b > 0.0):
        return None

    pooled = math.sqrt(sigma_a * sigma_a + sigma_b * sigma_b)
    if pooled <= 0.0:
        return None

    return abs(delta_a - delta_b) / pooled


# 関数: `_label_fields_audit` の入出力契約と処理意図を定義する。

def _label_fields_audit(root: Path, source_files: Sequence[str]) -> Dict[str, object]:
    pds_root = root / "data" / "cassini" / "pds_sce1"
    keywords = ["PLASMA", "TROPO", "IONO", "MEDIA", "CORRECTION", "CALIBRATION", "Z CORRECTION"]

    field_hits: Dict[str, int] = {}
    text_hits: Dict[str, int] = {}
    scanned = 0
    label_missing = 0
    for rel in sorted(set(source_files)):
        if not rel:
            continue

        odf = pds_root / Path(*rel.split("/"))
        lbl = odf.with_suffix(".lbl")
        if not lbl.exists():
            lbl = odf.with_suffix(".LBL")

        if not lbl.exists():
            label_missing += 1
            continue

        scanned += 1
        txt = lbl.read_text(encoding="utf-8", errors="replace")

        for m in re.finditer(r'^\s*NAME\s*=\s*"([^"]+)"\s*$', txt, flags=re.IGNORECASE | re.MULTILINE):
            nm = m.group(1).strip()
            up = nm.upper()
            for kw in keywords:
                if kw in up:
                    field_hits[nm] = field_hits.get(nm, 0) + 1

        up_txt = txt.upper()
        for kw in keywords:
            c = up_txt.count(kw)
            if c > 0:
                text_hits[kw] = text_hits.get(kw, 0) + int(c)

    media_flag_detected = len(field_hits) > 0
    status = "pass" if media_flag_detected else "watch"
    return {
        "status": status,
        "record_level_media_flag_detected": bool(media_flag_detected),
        "scanned_labels": int(scanned),
        "missing_labels": int(label_missing),
        "field_hits": field_hits,
        "description_keyword_hits": text_hits,
        "watch_reason": (
            ""
            if media_flag_detected
            else "ODF labels contain correction-related descriptions but no explicit record-level media flag field."
        ),
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(
    path: Path,
    rows: Sequence[FitResult],
    best: Optional[FitResult],
    media: Dict[str, object],
    media_manifest: Dict[str, object],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate",
                "fit_intercept",
                "delta_beta",
                "delta_beta_sigma_proxy",
                "corr_obs_fit",
                "rmse",
                "weighted_rms",
                "zscore_vs_tdf_plasma",
                "zscore_vs_tdf_raw",
                "zscore_min_vs_tdf",
                "is_best_candidate",
            ]
        )
        best_name = best.candidate if best else ""
        for r in rows:
            writer.writerow(
                [
                    r.candidate,
                    "yes" if r.fit_intercept else "no",
                    f"{r.delta_beta:.15g}",
                    f"{r.delta_beta_sigma_proxy:.15g}",
                    f"{r.corr_obs_fit:.15g}",
                    f"{r.rmse:.15e}",
                    f"{r.weighted_rms:.15e}",
                    "" if r.zscore_vs_tdf_plasma is None else f"{r.zscore_vs_tdf_plasma:.15g}",
                    "" if r.zscore_vs_tdf_raw is None else f"{r.zscore_vs_tdf_raw:.15g}",
                    "" if r.zscore_min_vs_tdf is None else f"{r.zscore_min_vs_tdf:.15g}",
                    "yes" if r.candidate == best_name else "no",
                ]
            )

        writer.writerow([])
        writer.writerow(["media_audit_status", str(media.get("status") or "")])
        writer.writerow(["media_record_level_flag_detected", str(bool(media.get("record_level_media_flag_detected")))])
        writer.writerow(["media_scanned_labels", str(int(media.get("scanned_labels") or 0))])
        writer.writerow(["media_missing_labels", str(int(media.get("missing_labels") or 0))])
        writer.writerow(["media_watch_reason", str(media.get("watch_reason") or "")])
        writer.writerow([])
        writer.writerow(["media_manifest_extractable", str(bool(media_manifest.get("record_level_media_state_extractable")))])
        writer.writerow(["media_manifest_hook_mode", str(media_manifest.get("parser_media_hook_mode") or "")])
        writer.writerow(["media_manifest_external_csp_state", str(media_manifest.get("external_csp_public_reproducibility_state") or "")])
        writer.writerow(["media_manifest_external_csp_hard_watch", str(bool(media_manifest.get("external_csp_hard_watch_required")))])
        writer.writerow(["media_manifest_external_csp_watch_reason", str(media_manifest.get("external_csp_hard_watch_reason") or "")])


# 関数: `_sync_public` の入出力契約と処理意図を定義する。

def _sync_public(root: Path, names: Sequence[str]) -> None:
    src_dir = root / "output" / "cassini"
    dst_dir = root / "output" / "public" / "cassini"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = src_dir / n
        if src.exists():
            shutil.copy2(src, dst_dir / n)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    points_csv = out_dir / "cassini_odf_beta_direct_fit_points.csv"
    cross_metrics = out_dir / "cassini_beta_direct_fit_cross_source_metrics.json"
    if_manifest = out_dir / "cassini_odf_raw_if_manifest.json"

    ys, xs, ws = _load_points(points_csv)
    cm = _read_json(cross_metrics)
    rows0 = cm.get("rows") if isinstance(cm.get("rows"), list) else []
    by = {str(r.get("scenario") or ""): r for r in rows0 if isinstance(r, dict)}
    tdf_plasma = by.get("tdf_plasma", {})
    tdf_raw = by.get("tdf_raw", {})
    d_plasma = _safe_float(tdf_plasma.get("delta_beta")) or math.nan
    s_plasma = _safe_float(tdf_plasma.get("delta_beta_sigma_proxy")) or math.nan
    d_raw = _safe_float(tdf_raw.get("delta_beta")) or math.nan
    s_raw = _safe_float(tdf_raw.get("delta_beta_sigma_proxy")) or math.nan

    candidates: List[Tuple[str, List[float], bool]] = []
    y_med = float(median(ys))
    candidates.append(("same_sign", list(ys), True))
    candidates.append(("invert_sign", [-y for y in ys], True))
    candidates.append(("same_sign_offset_removed", [y - y_med for y in ys], False))
    candidates.append(("invert_sign_offset_removed", [-(y - y_med) for y in ys], False))

    fit_rows: List[FitResult] = []
    for name, y_cand, fit_intercept in candidates:
        delta, sig, intercept, corr, rmse, wrms = _weighted_fit(y_cand, xs, ws, fit_intercept=fit_intercept)
        z_p = _zscore(delta, sig, d_plasma, s_plasma)
        z_r = _zscore(delta, sig, d_raw, s_raw)
        z_vals = [z for z in [z_p, z_r] if z is not None and math.isfinite(z)]
        z_min = min(z_vals) if z_vals else None
        fit_rows.append(
            FitResult(
                candidate=name,
                fit_intercept=fit_intercept,
                delta_beta=delta,
                delta_beta_sigma_proxy=sig,
                intercept=intercept,
                corr_obs_fit=corr,
                rmse=rmse,
                weighted_rms=wrms,
                n_points=len(y_cand),
                zscore_vs_tdf_plasma=z_p,
                zscore_vs_tdf_raw=z_r,
                zscore_min_vs_tdf=z_min,
            )
        )

    # 関数: `_rank_key` の入出力契約と処理意図を定義する。

    def _rank_key(r: FitResult) -> Tuple[float, float]:
        z = r.zscore_min_vs_tdf if r.zscore_min_vs_tdf is not None and math.isfinite(r.zscore_min_vs_tdf) else float("inf")
        c = r.corr_obs_fit if math.isfinite(r.corr_obs_fit) else -float("inf")
        return (z, -c)

    best = min(fit_rows, key=_rank_key) if fit_rows else None
    ifm = _read_json(if_manifest) if if_manifest.exists() else {}
    keys0 = ifm.get("keys") if isinstance(ifm.get("keys"), dict) else {}
    sign_key = keys0.get("doppler_sign_convention") if isinstance(keys0.get("doppler_sign_convention"), dict) else {}
    media_key = keys0.get("media_correction_state") if isinstance(keys0.get("media_correction_state"), dict) else {}
    sign_terminal_gate_pass = bool(sign_key.get("terminal_gate_pass"))
    sign_watch_codes = sign_key.get("watch_reason_codes") if isinstance(sign_key.get("watch_reason_codes"), list) else []
    media_manifest_extractable = bool(media_key.get("record_level_media_state_extractable"))
    media_manifest_mode = str(media_key.get("parser_media_hook_mode") or "")
    media_manifest_crosswalk = str(media_key.get("crosswalk_json") or "")
    media_manifest_external_csp_state = str(media_key.get("external_csp_public_reproducibility_state") or "")
    media_manifest_external_csp_hard_watch = bool(media_key.get("external_csp_hard_watch_required"))
    media_manifest_external_csp_watch_reason = str(media_key.get("external_csp_hard_watch_reason") or "")
    src_files = ifm.get("source_files") if isinstance(ifm.get("source_files"), list) else []
    src_files = [str(x) for x in src_files if str(x)]
    media = _label_fields_audit(root, src_files)

    gate_sign = bool(
        best is not None
        and best.zscore_min_vs_tdf is not None
        and best.zscore_min_vs_tdf <= 3.0
        and math.isfinite(best.corr_obs_fit)
        and best.corr_obs_fit >= 0.25
    )
    gate_media_label = bool(media.get("record_level_media_flag_detected"))
    gate_media_manifest = bool(media_manifest_extractable)
    gate_media_external_csp = not media_manifest_external_csp_hard_watch
    gate_media_label_or_manifest = bool(gate_media_label or gate_media_manifest)
    gate_media = bool(gate_media_label_or_manifest and gate_media_manifest and gate_media_external_csp)
    overall_status = "pass_candidate" if gate_sign and gate_media else "watch"
    reasons: List[str] = []
    if not gate_sign:
        reasons.append("sign_hypothesis_not_closed")

    if not sign_terminal_gate_pass:
        reasons.append("odf_sign_terminal_gate_not_closed")

    if not gate_media_label and not gate_media_manifest:
        reasons.append("media_flag_not_observable")

    if not gate_media_manifest:
        reasons.append("media_non_extractable_confirmed")

    if not gate_media_external_csp:
        reasons.append("external_csp_payload_unavailable_hard_watch")

    media_manifest_assessment = {
        "sign_terminal_gate_pass": bool(sign_terminal_gate_pass),
        "sign_watch_reason_codes": [str(v) for v in sign_watch_codes if str(v).strip()],
        "record_level_media_state_extractable": bool(media_manifest_extractable),
        "parser_media_hook_mode": media_manifest_mode,
        "crosswalk_json": media_manifest_crosswalk,
        "external_csp_public_reproducibility_state": media_manifest_external_csp_state,
        "external_csp_hard_watch_required": bool(media_manifest_external_csp_hard_watch),
        "external_csp_hard_watch_reason": media_manifest_external_csp_watch_reason,
    }

    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "points_csv": str(points_csv),
            "cross_source_metrics": str(cross_metrics),
            "odf_if_manifest": str(if_manifest),
        },
        "candidate_fit_results": [
            {
                "candidate": r.candidate,
                "fit_intercept": bool(r.fit_intercept),
                "delta_beta": float(r.delta_beta),
                "delta_beta_sigma_proxy": float(r.delta_beta_sigma_proxy),
                "intercept": float(r.intercept),
                "corr_obs_fit": float(r.corr_obs_fit),
                "rmse": float(r.rmse),
                "weighted_rms": float(r.weighted_rms),
                "n_points": int(r.n_points),
                "zscore_vs_tdf_plasma": r.zscore_vs_tdf_plasma,
                "zscore_vs_tdf_raw": r.zscore_vs_tdf_raw,
                "zscore_min_vs_tdf": r.zscore_min_vs_tdf,
            }
            for r in fit_rows
        ],
        "best_candidate": (
            {
                "candidate": best.candidate,
                "fit_intercept": bool(best.fit_intercept),
                "delta_beta": float(best.delta_beta),
                "delta_beta_sigma_proxy": float(best.delta_beta_sigma_proxy),
                "corr_obs_fit": float(best.corr_obs_fit),
                "zscore_min_vs_tdf": best.zscore_min_vs_tdf,
            }
            if best is not None
            else {}
        ),
        "media_label_audit": media,
        "media_manifest_assessment": media_manifest_assessment,
        "promotion_gates": {
            "sign_hypothesis_closed": bool(gate_sign),
            "media_label_flag_observable": bool(gate_media_label),
            "media_label_or_manifest_observable": bool(gate_media_label_or_manifest),
            "media_manifest_extractable": bool(gate_media_manifest),
            "media_external_csp_payload_available": bool(gate_media_external_csp),
            "media_gate": bool(gate_media),
        },
        "recommended_status": overall_status,
        "reasons": reasons,
    }

    out_json = out_dir / "cassini_odf_sign_media_closure_audit.json"
    out_csv = out_dir / "cassini_odf_sign_media_closure_audit.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, fit_rows, best, media, media_manifest_assessment)
    _sync_public(root, [out_json.name, out_csv.name])
    print("Wrote:", out_json)
    print("Wrote:", out_csv)
    print("Synced:", root / "output" / "public" / "cassini")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
