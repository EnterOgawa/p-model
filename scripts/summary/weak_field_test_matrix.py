#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weak_field_test_matrix.py

Phase 6 / Step 6.2.1:
弱場テスト（太陽系）を長期・多系統で統合するための「統合I/F」を固定し、
入力・期間・推定量・補正（knobs）・再現コマンドを JSON にまとめる。

出力（固定）:
  - output/private/summary/weak_field_test_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_relpath` の入出力契約と処理意図を定義する。

def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


# 関数: `_path_item` の入出力契約と処理意図を定義する。

def _path_item(rel: str, *, required: bool, note: str) -> Dict[str, Any]:
    p = (_ROOT / rel).resolve()
    return {"path": rel.replace("\\", "/"), "required": bool(required), "exists": p.exists(), "note": str(note)}


# 関数: `_script_item` の入出力契約と処理意図を定義する。

def _script_item(rel: str, *, role: str, repro: str, note: str = "") -> Dict[str, Any]:
    p = (_ROOT / rel).resolve()
    return {
        "path": rel.replace("\\", "/"),
        "role": str(role),
        "exists": p.exists(),
        "repro": str(repro),
        "note": str(note),
    }


# 関数: `_output_item` の入出力契約と処理意図を定義する。

def _output_item(rel: str, *, note: str = "") -> Dict[str, Any]:
    p = (_ROOT / rel).resolve()
    return {"path": rel.replace("\\", "/"), "exists": p.exists(), "note": str(note)}


# 関数: `_try_load_frozen_parameters` の入出力契約と処理意図を定義する。

def _try_load_frozen_parameters() -> Dict[str, Any]:
    p = _ROOT / "output" / "private" / "theory" / "frozen_parameters.json"
    # 条件分岐: `not p.exists()` を満たす経路を評価する。
    if not p.exists():
        return {"path": _relpath(p), "exists": False}

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"path": _relpath(p), "exists": True, "parse_error": True}

    out: Dict[str, Any] = {"path": _relpath(p), "exists": True}
    for k in ("beta", "beta_sigma", "gamma_pmodel", "gamma_pmodel_sigma", "delta"):
        # 条件分岐: `k in data` を満たす経路を評価する。
        if k in data:
            out[k] = data.get(k)

    policy = data.get("policy")
    # 条件分岐: `isinstance(policy, dict)` を満たす経路を評価する。
    if isinstance(policy, dict):
        out["policy"] = {kk: policy.get(kk) for kk in ("fit_predict_separation", "beta_source", "delta_source", "note")}

    return out


# 関数: `build_matrix` の入出力契約と処理意図を定義する。

def build_matrix() -> Dict[str, Any]:
    frozen = _try_load_frozen_parameters()

    tests: List[Dict[str, Any]] = []

    tests.append(
        {
            "id": "cassini_sce1_doppler",
            "topic": "cassini",
            "name": "Cassini SCE1（太陽会合）: ドップラー y(t)",
            "observable": {"name": "two-way fractional frequency y(t)", "unit": "dimensionless"},
            "period": {
                "kind": "event_window",
                "label": "SCE1（Fig.2 window）",
                "note": "主に ±10日（zoom）と全期間（full）で比較。期間は cassini_fig2_overlay.py の設定で管理。",
            },
            "inputs": [
                _path_item(
                    "output/private/cassini/cassini_shapiro_y_full.csv",
                    required=True,
                    note="幾何（Earth/Cassini ベクトル由来）のモデルCSV。再生成にはネットワーク（HORIZONS）が必要。",
                ),
                _path_item(
                    "data/cassini/pds_sce1/",
                    required=True,
                    note="PDS一次データ（SCE1 TDF/ODF）キャッシュ。",
                ),
                _path_item(
                    "data/cassini/cassini_fig2_digitized_raw.csv",
                    required=False,
                    note="論文 Fig.2 のデジタイズ点列（参考/フォールバック）。",
                ),
            ],
            "scripts": [
                _script_item(
                    "scripts/cassini/cassini_fig2_overlay.py",
                    role="main_offline",
                    repro="python -B scripts/cassini/cassini_fig2_overlay.py --beta <frozen>",
                    note="PDS TDFの疑似残差から y(t) を復元し、モデルと重ね合わせ。",
                ),
                _script_item(
                    "scripts/cassini/cassini_shapiro_y_full.py",
                    role="optional_online",
                    repro="python -B scripts/cassini/cassini_shapiro_y_full.py --beta <frozen>",
                    note="幾何CSVの再生成（HORIZONSアクセスが必要）。",
                ),
            ],
            "outputs": [
                _output_item("output/private/cassini/cassini_fig2_overlay_full.png", note="観測 vs モデル（全期間）"),
                _output_item("output/private/cassini/cassini_fig2_overlay_zoom10d.png", note="観測 vs モデル（±10日）"),
                _output_item("output/private/cassini/cassini_fig2_residuals.png", note="残差（観測-モデル）"),
                _output_item("output/private/cassini/cassini_fig2_metrics.csv", note="RMSE/MAE/corr 等の指標"),
                _output_item("output/private/cassini/cassini_fig2_run_metadata.json", note="実行条件・入力・パラメータ"),
                _output_item("output/private/cassini/cassini_beta_sweep_rmse.csv", note="βスイープのRMSE"),
                _output_item("output/private/cassini/cassini_beta_sweep_rmse.png", note="βスイープの可視化"),
            ],
            "frozen_parameters_policy": {
                "beta": "Use output/private/theory/frozen_parameters.json (no per-dataset retuning).",
                "note": "Cassini は β 拘束の主力。Phase 6.2 では長期・多系統でも同一βで自己矛盾しないことを確認する。",
            },
            "systematics_knobs": [
                {"key": "source", "values": ["pds_tdf", "pds_tdf_raw", "digitized", "pds_odf_raw"], "note": "観測系列の取り方"},
                {"key": "tdf_bin_seconds", "default": 3600, "note": "TDF binning"},
                {"key": "tdf_detrend_poly_order", "default": 0, "note": "detrend polynomial order"},
                {"key": "tdf_detrend_exclude_inner_days", "default": 5.0, "note": "中心部を除外してdetrend"},
                {"key": "tdf_reconstruct_shapiro", "default": True, "note": "疑似残差 + 参照Shapiroの足し戻し"},
                {"key": "tdf_y_sign", "default": +1, "note": "TDF定義により符号反転が必要な場合がある"},
            ],
            "refs": [
                _path_item("doc/cassini/README.md", required=False, note="再現手順とオプション一覧。"),
            ],
        }
    )

    tests.append(
        {
            "id": "viking_shapiro_peak",
            "topic": "viking",
            "name": "Viking（地球-火星）: Shapiro遅延（代表ピーク）",
            "observable": {"name": "two-way Shapiro delay (peak)", "unit": "microsecond"},
            "period": {
                "kind": "scan_window",
                "label": "指定 start/stop/step のスキャン",
                "note": "現段階では観測時系列は未導入で、文献の代表ピーク（約250 μs）と比較する。",
            },
            "inputs": [
                _path_item(
                    "doc/slides/時間波密度Pで捉える「重力・時間・光」.pptx",
                    required=False,
                    note="スライド雛形（update_slides.py）。",
                ),
                _path_item(
                    "output/private/viking/horizons_cache/",
                    required=False,
                    note="HORIZONS幾何キャッシュ（初回はネットワークが必要）。",
                ),
            ],
            "scripts": [
                _script_item(
                    "scripts/viking/viking_shapiro_check.py",
                    role="main",
                    repro="python -B scripts/viking/viking_shapiro_check.py --beta <frozen> [--start ... --stop ... --step ...]",
                    note="HORIZONS幾何から Shapiro 遅延を計算し、CSVを生成。",
                ),
                _script_item(
                    "scripts/viking/update_slides.py",
                    role="optional",
                    repro="python -B scripts/viking/update_slides.py",
                    note="結果図をスライドへ反映。",
                ),
            ],
            "outputs": [
                _output_item("output/private/viking/viking_shapiro_result.csv"),
                _output_item("output/private/viking/viking_p_model_vs_measured_no_arrow.png"),
                _output_item("output/private/viking/P_model_Verification_Full.pptx"),
            ],
            "systematics_knobs": [
                {"key": "start/stop/step", "note": "期間スキャン（幾何ピークの取り方も含む）"},
                {"key": "offline", "note": "HORIZONSキャッシュの有無（--offline または HORIZONS_OFFLINE=1）"},
            ],
            "refs": [
                _path_item("doc/viking/README.md", required=False, note="再現手順（キャッシュ運用）。"),
            ],
        }
    )

    tests.append(
        {
            "id": "mercury_perihelion_precession",
            "topic": "mercury",
            "name": "水星：近日点移動（簡易数値積分）",
            "observable": {"name": "perihelion shift", "unit": "arcsec/century (derived)"},
            "period": {"kind": "simulation", "label": "fixed configuration", "note": "簡易モデル（精密暦ではない）。"},
            "inputs": [],
            "scripts": [
                _script_item(
                    "scripts/mercury/mercury_precession_v3.py",
                    role="main_offline",
                    repro="python -B scripts/mercury/mercury_precession_v3.py",
                    note="簡易数値積分で近日点移動を可視化。",
                ),
            ],
            "outputs": [
                _output_item("output/private/mercury/mercury_orbit.png"),
                _output_item("output/private/mercury/mercury_precession_metrics.json", note="数値（角秒/世紀など）"),
                _output_item("output/private/mercury/mercury_perihelion_shifts.csv", note="周回ごとの累積値"),
            ],
            "systematics_knobs": [
                {"key": "unmodeled_perturbations", "note": "他惑星摂動/太陽四重極/小天体摂動等を入れていない限界を明示"},
            ],
            "refs": [
                _path_item("doc/mercury/README.md", required=False, note="再現手順。"),
            ],
        }
    )

    tests.append(
        {
            "id": "gps_satellite_clock",
            "topic": "gps",
            "name": "GPS：衛星時計（IGS精密クロック vs BRDC/P-model）",
            "observable": {"name": "clock residuals (IGS - model)", "unit": "ns"},
            "period": {
                "kind": "daily",
                "label": "default: 2025-10-01 (DOY=274; GPS Week 2386)",
                "note": "別日に切替可能（fetch_igs_bkg.py）。",
            },
            "inputs": [
                _path_item(
                    "data/gps/BRDC00IGS_R_20252740000_01D_MN.rnx",
                    required=True,
                    note="RINEX NAV（固定ファイル名）",
                ),
                _path_item(
                    "data/gps/IGS0OPSFIN_20252740000_01D_15M_ORB.SP3",
                    required=True,
                    note="IGS orbit (SP3; 固定ファイル名)",
                ),
                _path_item(
                    "data/gps/IGS0OPSFIN_20252740000_01D_05M_CLK.CLK",
                    required=True,
                    note="IGS clock (CLK; 固定ファイル名)",
                ),
            ],
            "scripts": [
                _script_item(
                    "scripts/gps/fetch_igs_bkg.py",
                    role="optional_online",
                    repro="python -B scripts/gps/fetch_igs_bkg.py --date YYYY-MM-DD",
                    note="IGS/BKG から取得して data/gps の既定ファイル名へ保存。",
                ),
                _script_item(
                    "scripts/gps/compare_clocks.py",
                    role="main_offline",
                    repro="python -B scripts/gps/compare_clocks.py",
                    note="IGS CLK に対して BRDC と P-model を比較（dt_rel除去を含む）。",
                ),
                _script_item(
                    "scripts/gps/plot.py",
                    role="postprocess",
                    repro="python -B scripts/gps/plot.py",
                    note="残差図と summary 指標を生成。",
                ),
            ],
            "outputs": [
                _output_item("output/private/gps/summary_batch.csv", note="衛星ごとのRMSなど"),
                _output_item("output/private/gps/gps_clock_residuals_all_31.png", note="全衛星重ね描き"),
                _output_item("output/private/gps/gps_residual_compare_G01.png", note="G01例"),
                _output_item("output/private/gps/gps_rms_compare.png", note="全衛星RMS比較"),
                _output_item("output/private/gps/gps_relativistic_correction_G02.png", note="dt_rel（近日点効果）比較"),
                _output_item("output/private/gps/gps_compare_metrics.json", note="指標JSON（Table 1入力）"),
            ],
            "systematics_knobs": [
                {"key": "ref_include_earth_rotation", "note": "地上基準点速度に地球自転を含める（--ref-include-earth-rotation）"},
                {"key": "no_sat_earth_rotation", "note": "衛星速度の ω×r 補正を無効化（--no-sat-earth-rotation; デバッグ）"},
                {"key": "file_naming", "note": "固定ファイル名で回す（別日に切替する場合は同名差替え or compare_clocks.py編集）"},
            ],
            "refs": [
                _path_item("doc/gps/README.md", required=False, note="再現手順（固定ファイル名の注意）。"),
            ],
        }
    )

    tests.append(
        {
            "id": "llr_batch",
            "topic": "llr",
            "name": "LLR：バッチ検証（複数局×複数反射器）",
            "observable": {"name": "range residuals (obs - model)", "unit": "ns / m (derived)"},
            "period": {
                "kind": "multi_year",
                "label": "EDC月次NP2（拡張可能）",
                "note": "coverage（局/年/反射器）偏りを明示し、外れ値に敏感な統計を運用で固定する。",
            },
            "inputs": [
                _path_item("data/llr/llr_edc_batch_manifest.json", required=False, note="EDCバッチ取得マニフェスト"),
                _path_item("data/llr/edc/", required=False, note="EDC取得データ（キャッシュ）"),
                _path_item("data/llr/llr_primary.np2", required=False, note="解析既定入力（単体overlay等）"),
                _path_item("data/llr/stations/", required=False, note="観測局 site log キャッシュ"),
                _path_item("data/llr/reflectors_de421_pa.json", required=False, note="反射器座標（PA）"),
                _path_item("data/llr/kernels/naif/", required=False, note="SPICE kernels（MOON_PA_DE421 等）"),
            ],
            "scripts": [
                _script_item(
                    "scripts/llr/fetch_llr_edc_batch.py",
                    role="optional_online",
                    repro="python -B scripts/llr/fetch_llr_edc_batch.py",
                    note="EDC（月次NP2）を一括取得し data/llr/edc/ にキャッシュ。",
                ),
                _script_item(
                    "scripts/llr/llr_batch_eval.py",
                    role="main",
                    repro="python -B scripts/llr/llr_batch_eval.py --offline --chunk 50",
                    note="複数局×複数反射器でP-model残差を評価（Horizonsキャッシュ運用）。",
                ),
                _script_item(
                    "scripts/llr/llr_pmodel_overlay_horizons_noargs.py",
                    role="single_overlay",
                    repro="python -B scripts/llr/llr_pmodel_overlay_horizons_noargs.py",
                    note="単体の重ね合わせ（入力は data/llr/llr_primary.* を優先）。",
                ),
            ],
            "outputs": [
                _output_item("output/private/llr/batch/llr_batch_summary.json"),
                _output_item("output/private/llr/batch/llr_batch_metrics.csv"),
                _output_item("output/private/llr/batch/llr_outliers.csv"),
                _output_item("output/private/llr/batch/llr_outliers_diagnosis.csv"),
                _output_item("output/private/llr/batch/llr_residual_distribution.png"),
                _output_item("output/private/llr/batch/llr_rms_improvement_overall.png"),
                _output_item("output/private/llr/batch/llr_rms_by_station_target.png"),
            ],
            "systematics_knobs": [
                {"key": "time_tag_mode", "note": "tx/rx/mid/auto（局別最適を採用するか）"},
                {"key": "outlier_policy", "note": "MADベース外れ値除外（ns級で必須）"},
                {"key": "station_metadata_source", "note": "pos+eop vs site log 等"},
                {"key": "include_daily", "note": "月次と日次の重複排除（include_daily=true時）"},
            ],
            "refs": [
                _path_item("doc/llr/README.md", required=False, note="再現手順（offline + optional network）。"),
                _path_item("doc/llr/MODEL_SPEC.md", required=False, note="モデル構成と運用上の規約。"),
            ],
        }
    )

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 6, "step": "6.2.1", "name": "弱場統合I/F固定"},
        "purpose": (
            "弱場（太陽系）テストを長期・多系統で統合するために、"
            "対象テスト、入力、推定量、補正（knobs）、再現コマンド、主要出力の一覧（統合I/F）を固定する。"
        ),
        "policy": {
            "fit_predict_separation": True,
            "beta_frozen_across_tests": True,
            "note": "Phase 6.2 は『弱場の逃げ道封鎖』として、同一βなどの凍結パラメータを前提に長期・多系統で自己矛盾しないことを確認する。",
        },
        "common": {"frozen_parameters": frozen},
        "tests": tests,
        "outputs": {"weak_field_test_matrix_json": "output/private/summary/weak_field_test_matrix.json"},
    }

    return payload


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build weak-field test matrix (Phase 6 / Step 6.2.1).")
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "weak_field_test_matrix.json"),
        help="Output JSON path (default: output/private/summary/weak_field_test_matrix.json).",
    )
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_matrix()
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {_relpath(out_path)}")

    try:
        worklog.append_event(
            {
                "event_type": "summary_weak_field_test_matrix",
                "argv": list(sys.argv),
                "outputs": {"weak_field_test_matrix_json": out_path},
                "metrics": {"tests_n": len(payload.get("tests") or [])},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
