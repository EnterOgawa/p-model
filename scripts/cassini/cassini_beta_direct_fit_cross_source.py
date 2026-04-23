#!/usr/bin/env python3
"""Cassini beta direct-fit cross-source comparison.

Purpose:
- Run direct beta fit for three Cassini source modes under the same fit window:
  1) TDF processed plasma path (source=pds_tdf)
  2) TDF raw path (source=pds_tdf_raw)
  3) ODF raw path (source=pds_odf_raw)
- Aggregate metrics into fixed-name CSV/JSON artifacts.
- Generate a vector-first comparison figure (PDF canonical, PNG optional).

Inputs:
- Existing Cassini geometry/model CSV used by cassini_fig2_overlay.py
- PDS SCE1 cached data under data/cassini/pds_sce1

Outputs:
- output/cassini/cassini_beta_direct_fit_cross_source_summary.csv
- output/cassini/cassini_beta_direct_fit_cross_source_metrics.json
- output/cassini/cassini_beta_direct_fit_cross_source_convergence.csv
- output/cassini/cassini_beta_direct_fit_cross_source.pdf
- output/cassini/cassini_beta_direct_fit_cross_source.png
- output/cassini/cassini_odf_raw_if_manifest.json
- output/cassini/cassini_odf_raw_if_manifest.csv
- synced copies in output/public/cassini/
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `WAVEP_FIGURE_LANG` から英語 surface かどうかを判定する。
def _is_en_figure() -> bool:
    return str(os.getenv("WAVEP_FIGURE_LANG", "ja")).strip().lower().startswith("en")


# 関数: public 出力先を locale ごとに解決する。
def _public_output_dir(root: Path) -> Path:
    base = root / "output" / "public" / "cassini"
    if _is_en_figure():
        return base / "locales" / "en"

    return base


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        from matplotlib import font_manager, rcParams  # type: ignore
    except Exception:
        return

    candidates = [
        "Yu Gothic",
        "MS Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "TakaoGothic",
        "VL Gothic",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            rcParams["font.family"] = name
            break

    rcParams["axes.unicode_minus"] = False


# 関数: `run_scenario` の入出力契約と処理意図を定義する。

def run_scenario(root: Path, scenario: Dict[str, object]) -> Dict[str, object]:
    cmd = ["python", "-B"] + list(scenario["command"])
    subprocess.run(cmd, cwd=str(root), check=True)

    metrics_path = root / str(scenario["metrics_relpath"])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    fit = payload.get("fit_result") if isinstance(payload.get("fit_result"), dict) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}

    row = {
        "scenario": str(scenario["key"]),
        "label": str(scenario["label"]),
        "source_effective": str(inputs.get("source_effective") or ""),
        "window_days": _safe_float(fit.get("window_days")),
        "n_points": int(_safe_float(fit.get("n_points")) or 0),
        "delta_beta": _safe_float(fit.get("delta_beta")),
        "delta_beta_sigma_proxy": _safe_float(fit.get("delta_beta_sigma_proxy")),
        "beta_est": _safe_float(fit.get("beta_est")),
        "gamma_est": _safe_float(fit.get("gamma_est")),
        "corr_obs_fit": _safe_float(fit.get("corr_obs_fit")),
        "rmse": _safe_float(fit.get("rmse")),
        "weighted_rms": _safe_float(fit.get("weighted_rms")),
        "metrics_json": str(metrics_path),
    }
    return row


# 関数: `run_odf_if_manifest` の入出力契約と処理意図を定義する。

def run_odf_if_manifest(root: Path) -> Dict[str, object]:
    cmd = ["python", "-B", "scripts/cassini/cassini_odf_raw_if_manifest.py"]
    subprocess.run(cmd, cwd=str(root), check=True)
    manifest_path = root / "output" / "cassini" / "cassini_odf_raw_if_manifest.json"
    if not manifest_path.exists():
        return {}

    return _read_json(manifest_path)


# 関数: `write_summary_csv` の入出力契約と処理意図を定義する。

def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "label",
                "source_effective",
                "window_days",
                "n_points",
                "delta_beta",
                "delta_beta_sigma_proxy",
                "beta_est",
                "gamma_est",
                "corr_obs_fit",
                "rmse",
                "weighted_rms",
                "metrics_json",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["scenario"],
                    r["label"],
                    r["source_effective"],
                    f"{_safe_float(r.get('window_days')):.6g}",
                    int(r.get("n_points", 0)),
                    f"{_safe_float(r.get('delta_beta')):.15g}",
                    f"{_safe_float(r.get('delta_beta_sigma_proxy')):.15g}",
                    f"{_safe_float(r.get('beta_est')):.15g}",
                    f"{_safe_float(r.get('gamma_est')):.15g}",
                    f"{_safe_float(r.get('corr_obs_fit')):.15g}",
                    f"{_safe_float(r.get('rmse')):.15e}",
                    f"{_safe_float(r.get('weighted_rms')):.15e}",
                    r["metrics_json"],
                ]
            )


# 関数: `build_spread_metrics` の入出力契約と処理意図を定義する。

def build_spread_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    delta_values = [float(r["delta_beta"]) for r in rows if math.isfinite(_safe_float(r.get("delta_beta")))]
    corr_values = [float(r["corr_obs_fit"]) for r in rows if math.isfinite(_safe_float(r.get("corr_obs_fit")))]

    out: Dict[str, object] = {}
    # 条件分岐: `delta_values` を満たす経路を評価する。
    if delta_values:
        out["delta_beta_min"] = float(min(delta_values))
        out["delta_beta_max"] = float(max(delta_values))
        out["delta_beta_range"] = float(max(delta_values) - min(delta_values))
        mean = sum(delta_values) / len(delta_values)
        var = sum((v - mean) ** 2 for v in delta_values) / len(delta_values)
        out["delta_beta_mean"] = float(mean)
        out["delta_beta_std"] = float(math.sqrt(var))

    # 条件分岐: `corr_values` を満たす経路を評価する。

    if corr_values:
        out["corr_min"] = float(min(corr_values))
        out["corr_max"] = float(max(corr_values))
        out["corr_range"] = float(max(corr_values) - min(corr_values))

    by_key = {str(r["scenario"]): r for r in rows}
    ref = by_key.get("tdf_plasma")
    raw = by_key.get("tdf_raw")
    odf = by_key.get("odf_raw")

    # 条件分岐: `ref is not None and raw is not None` を満たす経路を評価する。
    if ref is not None and raw is not None:
        out["delta_beta_diff_tdf_plasma_minus_tdf_raw"] = float(
            _safe_float(ref.get("delta_beta")) - _safe_float(raw.get("delta_beta"))
        )

    # 条件分岐: `ref is not None and odf is not None` を満たす経路を評価する。

    if ref is not None and odf is not None:
        out["delta_beta_diff_tdf_plasma_minus_odf_raw"] = float(
            _safe_float(ref.get("delta_beta")) - _safe_float(odf.get("delta_beta"))
        )

    return out


# 関数: `build_convergence_audit` の入出力契約と処理意図を定義する。

def build_convergence_audit(rows: Sequence[Dict[str, object]], odf_manifest: Dict[str, object]) -> Dict[str, object]:
    by_key = {str(r.get("scenario") or ""): r for r in rows}
    r_plasma = by_key.get("tdf_plasma", {})
    r_raw = by_key.get("tdf_raw", {})
    r_odf = by_key.get("odf_raw", {})

    delta_plasma = _safe_float(r_plasma.get("delta_beta"))
    delta_raw = _safe_float(r_raw.get("delta_beta"))
    delta_odf = _safe_float(r_odf.get("delta_beta"))
    sigma_plasma = _safe_float(r_plasma.get("delta_beta_sigma_proxy"))
    sigma_raw = _safe_float(r_raw.get("delta_beta_sigma_proxy"))
    sigma_odf = _safe_float(r_odf.get("delta_beta_sigma_proxy"))
    corr_odf = _safe_float(r_odf.get("corr_obs_fit"))

    tdf_internal_threshold = 0.10
    odf_z_threshold = 3.0
    odf_corr_threshold = 0.25
    manifest_gate = bool(odf_manifest.get("overall_if_ready_for_absolute_beta"))
    keys0 = odf_manifest.get("keys") if isinstance(odf_manifest.get("keys"), dict) else {}
    sign_key = keys0.get("doppler_sign_convention") if isinstance(keys0.get("doppler_sign_convention"), dict) else {}
    media_key = keys0.get("media_correction_state") if isinstance(keys0.get("media_correction_state"), dict) else {}
    sign_terminal_gate = bool(sign_key.get("terminal_gate_pass"))
    sign_watch_codes = sign_key.get("watch_reason_codes") if isinstance(sign_key.get("watch_reason_codes"), list) else []
    external_csp_hard_watch = bool(media_key.get("external_csp_hard_watch_required"))
    gate_external_csp = not external_csp_hard_watch

    tdf_internal_delta = math.nan
    if math.isfinite(delta_plasma) and math.isfinite(delta_raw):
        tdf_internal_delta = abs(delta_plasma - delta_raw)

    gate_tdf_internal = bool(math.isfinite(tdf_internal_delta) and tdf_internal_delta <= tdf_internal_threshold)

    z_plasma = math.nan
    if math.isfinite(delta_odf) and math.isfinite(delta_plasma) and sigma_odf > 0.0 and sigma_plasma > 0.0:
        pooled = math.sqrt(sigma_odf * sigma_odf + sigma_plasma * sigma_plasma)
        if pooled > 0.0:
            z_plasma = abs(delta_odf - delta_plasma) / pooled

    z_raw = math.nan
    if math.isfinite(delta_odf) and math.isfinite(delta_raw) and sigma_odf > 0.0 and sigma_raw > 0.0:
        pooled = math.sqrt(sigma_odf * sigma_odf + sigma_raw * sigma_raw)
        if pooled > 0.0:
            z_raw = abs(delta_odf - delta_raw) / pooled

    z_candidates = [z for z in [z_plasma, z_raw] if math.isfinite(z)]
    z_best = min(z_candidates) if z_candidates else math.nan
    gate_odf_z = bool(math.isfinite(z_best) and z_best <= odf_z_threshold)
    gate_odf_corr = bool(math.isfinite(corr_odf) and corr_odf >= odf_corr_threshold)

    gates = {
        "if_manifest_all_pass": {
            "status": "pass" if manifest_gate else "watch",
            "value": bool(manifest_gate),
            "threshold": "True",
            "note": "ODF raw I/F 4項目（band/sign/media/time）が全てpass",
        },
        "external_csp_payload_gate": {
            "status": "pass" if gate_external_csp else "watch",
            "value": bool(gate_external_csp),
            "threshold": "True",
            "note": "IONCAL/TROPCAL/PLSMCAL payload available locally for record-level media join",
        },
        "odf_sign_terminal_gate": {
            "status": "pass" if sign_terminal_gate else "watch",
            "value": bool(sign_terminal_gate),
            "threshold": "True",
            "note": "record-level contiguous-arc sign/corr gate with minimum coverage is closed",
        },
        "tdf_internal_delta_beta": {
            "status": "pass" if gate_tdf_internal else "watch",
            "value": float(tdf_internal_delta) if math.isfinite(tdf_internal_delta) else None,
            "threshold": float(tdf_internal_threshold),
            "note": "|Δβ(tdf_plasma)-Δβ(tdf_raw)| <= 0.10",
        },
        "odf_vs_tdf_z_best": {
            "status": "pass" if gate_odf_z else "watch",
            "value": float(z_best) if math.isfinite(z_best) else None,
            "threshold": float(odf_z_threshold),
            "note": "min z-score between ODF raw and TDF branches <= 3",
        },
        "odf_corr_floor": {
            "status": "pass" if gate_odf_corr else "watch",
            "value": float(corr_odf) if math.isfinite(corr_odf) else None,
            "threshold": float(odf_corr_threshold),
            "note": "ODF raw corr >= 0.25",
        },
    }

    all_pass = all(str((g or {}).get("status", "")).lower() == "pass" for g in gates.values())
    recommended_status = "pass_candidate" if all_pass else "watch"
    reasons: List[str] = []
    if not manifest_gate:
        reasons.append("if_manifest_not_closed")

    if not sign_terminal_gate:
        reasons.append("odf_sign_terminal_gate_not_closed")

    if not gate_external_csp:
        reasons.append("external_csp_payload_unavailable_hard_watch")

    if not gate_tdf_internal:
        reasons.append("tdf_internal_spread_large")

    if not gate_odf_z:
        reasons.append("odf_delta_beta_not_converged")

    if not gate_odf_corr:
        reasons.append("odf_shape_corr_too_low")

    return {
        "recommended_status": recommended_status,
        "all_pass": bool(all_pass),
        "reasons": reasons,
        "gates": gates,
        "diagnostics": {
            "zscore_odf_vs_tdf_plasma": float(z_plasma) if math.isfinite(z_plasma) else None,
            "zscore_odf_vs_tdf_raw": float(z_raw) if math.isfinite(z_raw) else None,
            "delta_beta_tdf_plasma": float(delta_plasma) if math.isfinite(delta_plasma) else None,
            "delta_beta_tdf_raw": float(delta_raw) if math.isfinite(delta_raw) else None,
            "delta_beta_odf_raw": float(delta_odf) if math.isfinite(delta_odf) else None,
            "corr_odf_raw": float(corr_odf) if math.isfinite(corr_odf) else None,
            "sign_watch_reason_codes": [str(v) for v in sign_watch_codes if str(v).strip()],
        },
    }


# 関数: `write_convergence_csv` の入出力契約と処理意図を定義する。

def write_convergence_csv(path: Path, audit: Dict[str, object]) -> None:
    gates = audit.get("gates") if isinstance(audit.get("gates"), dict) else {}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gate", "status", "value", "threshold", "note"])
        for gate_name, gate in gates.items():
            g = gate if isinstance(gate, dict) else {}
            writer.writerow(
                [
                    str(gate_name),
                    str(g.get("status") or ""),
                    "" if g.get("value") is None else str(g.get("value")),
                    str(g.get("threshold") or ""),
                    str(g.get("note") or ""),
                ]
            )

        writer.writerow(
            [
                "recommended_status",
                str(audit.get("recommended_status") or ""),
                str(bool(audit.get("all_pass"))),
                "all gates pass",
                ",".join([str(r) for r in (audit.get("reasons") or [])]),
            ]
        )


# 関数: `plot_comparison` の入出力契約と処理意図を定義する。

def plot_comparison(out_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    is_en = _is_en_figure()
    font_scale = 1.22 if is_en else 1.0
    title_font = 13.8 * font_scale
    axis_font = 12.6 * font_scale
    tick_font = 11.2 * font_scale
    if not is_en:
        _set_japanese_font()
    xs = list(range(len(rows)))
    labels = [str(r["label"]) for r in rows]
    delta = [_safe_float(r.get("delta_beta")) for r in rows]
    sigma = [_safe_float(r.get("delta_beta_sigma_proxy")) for r in rows]
    corr = [_safe_float(r.get("corr_obs_fit")) for r in rows]

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.6),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex=True,
    )
    ax0.errorbar(xs, delta, yerr=sigma, fmt="o", capsize=4, color="tab:blue")
    ax0.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax0.set_ylabel("Δβ", fontsize=axis_font)
    ax0.set_title(
        "Cassini direct β-fit preprocessing comparison (same window, same formula)"
        if is_en
        else "Cassini 直接βfitの前処理依存比較（同一窓・同一式）",
        fontsize=title_font,
    )
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(labelsize=tick_font)

    ax1.bar(xs, corr, color=["tab:green", "tab:orange", "tab:red"], alpha=0.85)
    ax1.set_ylabel("corr", fontsize=axis_font)
    ax1.set_ylim(-0.05, 1.0)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=tick_font)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.tick_params(labelsize=tick_font)

    for axis in (ax0, ax1):
        for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick.set_fontsize(tick_font)

    fig.tight_layout()
    fig.savefig(out_dir / "cassini_beta_direct_fit_cross_source.png", dpi=180)
    fig.savefig(out_dir / "cassini_beta_direct_fit_cross_source.pdf")
    plt.close(fig)


# 関数: `sync_public` の入出力契約と処理意図を定義する。

def sync_public(root: Path, out_dir: Path) -> None:
    public_dir = _public_output_dir(root)
    public_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "cassini_beta_direct_fit_cross_source_summary.csv",
        "cassini_beta_direct_fit_cross_source_metrics.json",
        "cassini_beta_direct_fit_cross_source_convergence.csv",
        "cassini_beta_direct_fit_cross_source.png",
        "cassini_beta_direct_fit_cross_source.pdf",
        "cassini_odf_raw_if_manifest.json",
        "cassini_odf_raw_if_manifest.csv",
    ]:
        src = out_dir / name
        # 条件分岐: `src.exists()` を満たす経路を評価する。
        if src.exists():
            shutil.copy2(src, public_dir / name)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios: List[Dict[str, object]] = [
        {
            "key": "tdf_plasma",
            "label": "TDF processed plasma",
            "command": [
                "scripts/cassini/cassini_fig2_overlay.py",
                "--source",
                "pds_tdf",
                "--no-tdf-reconstruct-shapiro",
                "--tdf-direct-beta-fit",
                "--tdf-direct-beta-ref",
                "1.0",
                "--tdf-direct-beta-window-days",
                "10",
                "--no-sweep",
                "--no-plots",
            ],
            "metrics_relpath": "output/cassini/cassini_tdf_delta_beta_fit_metrics.json",
        },
        {
            "key": "tdf_raw",
            "label": "TDF raw",
            "command": [
                "scripts/cassini/cassini_fig2_overlay.py",
                "--source",
                "pds_tdf_raw",
                "--no-tdf-reconstruct-shapiro",
                "--tdf-direct-beta-fit",
                "--tdf-direct-beta-ref",
                "1.0",
                "--tdf-direct-beta-window-days",
                "10",
                "--no-sweep",
                "--no-plots",
            ],
            "metrics_relpath": "output/cassini/cassini_tdf_delta_beta_fit_metrics.json",
        },
        {
            "key": "odf_raw",
            "label": "ODF raw",
            "command": [
                "scripts/cassini/cassini_fig2_overlay.py",
                "--source",
                "pds_odf_raw",
                "--odf-direct-beta-fit",
                "--odf-direct-beta-ref",
                "1.0",
                "--odf-direct-beta-window-days",
                "10",
                "--no-sweep",
                "--no-plots",
            ],
            "metrics_relpath": "output/cassini/cassini_odf_beta_direct_fit_metrics.json",
        },
    ]

    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        rows.append(run_scenario(root, scenario))

    odf_if_manifest = run_odf_if_manifest(root)
    convergence_audit = build_convergence_audit(rows, odf_if_manifest)

    summary_csv = out_dir / "cassini_beta_direct_fit_cross_source_summary.csv"
    metrics_json = out_dir / "cassini_beta_direct_fit_cross_source_metrics.json"
    convergence_csv = out_dir / "cassini_beta_direct_fit_cross_source_convergence.csv"
    write_summary_csv(summary_csv, rows)
    write_convergence_csv(convergence_csv, convergence_audit)
    plot_comparison(out_dir, rows)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fit_model": "y_obs = a0 + delta_beta * y_unit(beta=1)",
        "window_days": 10.0,
        "beta_ref": 1.0,
        "rows": rows,
        "spread_metrics": build_spread_metrics(rows),
        "odf_if_manifest": {
            "path": str(out_dir / "cassini_odf_raw_if_manifest.json"),
            "overall_status": str(odf_if_manifest.get("overall_status") or ""),
            "overall_if_ready_for_absolute_beta": bool(odf_if_manifest.get("overall_if_ready_for_absolute_beta")),
        },
        "convergence_audit": convergence_audit,
        "outputs": {
            "summary_csv": str(summary_csv),
            "convergence_csv": str(convergence_csv),
            "plot_png": str(out_dir / "cassini_beta_direct_fit_cross_source.png"),
            "plot_pdf": str(out_dir / "cassini_beta_direct_fit_cross_source.pdf"),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sync_public(root, out_dir)
    print("Wrote:", summary_csv)
    print("Wrote:", convergence_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", out_dir / "cassini_beta_direct_fit_cross_source.png")
    print("Wrote:", out_dir / "cassini_beta_direct_fit_cross_source.pdf")
    print("Synced:", root / "output" / "public" / "cassini")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
