#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weak_field_systematics_templates.py

Phase 6 / Step 6.2.2:
弱場（太陽系）テストの「系統分解テンプレート（何を動かしたら何が動くか）」を固定し、
長期・多系統統合（Step 6.2.3）に向けた入力（JSON）を生成する。

出力（固定）:
  - output/private/summary/weak_field_systematics_templates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_main_script(test: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    scripts = test.get("scripts") or []
    # 条件分岐: `not isinstance(scripts, list)` を満たす経路を評価する。
    if not isinstance(scripts, list):
        return None

    for s in scripts:
        # 条件分岐: `not isinstance(s, dict)` を満たす経路を評価する。
        if not isinstance(s, dict):
            continue

        role = str(s.get("role") or "")
        # 条件分岐: `role.startswith("main")` を満たす経路を評価する。
        if role.startswith("main"):
            return s

    for s in scripts:
        # 条件分岐: `isinstance(s, dict)` を満たす経路を評価する。
        if isinstance(s, dict):
            return s

    return None


def _paths_from_outputs(test: Dict[str, Any], *, limit: int = 8) -> List[str]:
    outs = test.get("outputs") or []
    # 条件分岐: `not isinstance(outs, list)` を満たす経路を評価する。
    if not isinstance(outs, list):
        return []

    paths: List[str] = []
    for item in outs:
        # 条件分岐: `not isinstance(item, dict)` を満たす経路を評価する。
        if not isinstance(item, dict):
            continue

        p = item.get("path")
        # 条件分岐: `not isinstance(p, str) or not p` を満たす経路を評価する。
        if not isinstance(p, str) or not p:
            continue

        paths.append(p)
        # 条件分岐: `len(paths) >= limit` を満たす経路を評価する。
        if len(paths) >= limit:
            break

    return paths


def _sys_item(
    *,
    sys_id: str,
    title: str,
    category: str,
    status: str,
    knobs: List[Dict[str, Any]],
    expected_effect: List[Dict[str, Any]],
    diagnostics_outputs: List[str],
    what_to_check: List[str],
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "id": str(sys_id),
        "title": str(title),
        "category": str(category),
        "status": str(status),
        "knobs": knobs,
        "expected_effect": expected_effect,
        "diagnostics": {"outputs": diagnostics_outputs, "what_to_check": what_to_check},
        "notes": str(notes),
    }


def _knob(
    *,
    key: str,
    knob_type: str,
    apply: Dict[str, Any],
    values: Optional[List[Any]] = None,
    default: Optional[Any] = None,
    note: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"key": str(key), "type": str(knob_type), "apply": apply, "note": str(note)}
    # 条件分岐: `values is not None` を満たす経路を評価する。
    if values is not None:
        out["values"] = values

    # 条件分岐: `default is not None` を満たす経路を評価する。

    if default is not None:
        out["default"] = default

    return out


def build_templates(matrix: Dict[str, Any]) -> Dict[str, Any]:
    tests = matrix.get("tests") or []
    # 条件分岐: `not isinstance(tests, list) or not tests` を満たす経路を評価する。
    if not isinstance(tests, list) or not tests:
        raise ValueError("Invalid matrix: missing tests[]")

    test_by_id: Dict[str, Dict[str, Any]] = {}
    for t in tests:
        # 条件分岐: `not isinstance(t, dict)` を満たす経路を評価する。
        if not isinstance(t, dict):
            continue

        tid = t.get("id")
        # 条件分岐: `isinstance(tid, str) and tid` を満たす経路を評価する。
        if isinstance(tid, str) and tid:
            test_by_id[tid] = t

    def _t(tid: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        # 条件分岐: `tid not in test_by_id` を満たす経路を評価する。
        if tid not in test_by_id:
            raise KeyError(f"Missing test id in matrix: {tid}")

        test = test_by_id[tid]
        main_script = _pick_main_script(test) or {}
        primary_outputs = _paths_from_outputs(test)
        return test, main_script, primary_outputs

    templates: List[Dict[str, Any]] = []

    # --- Cassini ---
    test, main_script, primary_outputs = _t("cassini_sce1_doppler")
    cassini_out = [
        "output/private/cassini/cassini_fig2_metrics.csv",
        "output/private/cassini/cassini_fig2_residuals.png",
        "output/private/cassini/cassini_fig2_overlay_zoom10d.png",
    ]
    cassini_sys: List[Dict[str, Any]] = []
    cassini_sys.append(
        _sys_item(
            sys_id="cassini_band_plasma",
            title="周波数帯と太陽プラズマ補正（Ka/X, dual-frequency）",
            category="instrument_propagation",
            status="supported",
            knobs=[
                _knob(
                    key="tdf_downlink_band",
                    knob_type="categorical",
                    apply={"mode": "cli_flag", "flag": "--tdf-downlink-band", "script": main_script.get("path")},
                    values=["ka", "x", "any"],
                    default="ka",
                    note="PDS TDF: downlink band filter",
                ),
                _knob(
                    key="tdf_plasma_correct",
                    knob_type="bool",
                    apply={"mode": "cli_flag", "flag": "--tdf-plasma-correct/--no-tdf-plasma-correct", "script": main_script.get("path")},
                    default=True,
                    note="PDS TDF: Ka+X が揃う区間でコロナ（プラズマ）成分を除去",
                ),
                _knob(
                    key="tdf_plasma_fill_mode",
                    knob_type="categorical",
                    apply={"mode": "cli_flag", "flag": "--tdf-plasma-fill-mode", "script": main_script.get("path")},
                    values=["dual", "merged"],
                    default="merged",
                    note="dual=共通ビンのみ / merged=欠測を単独バンドで補完",
                ),
            ],
            expected_effect=[
                {
                    "observable": "y(t)",
                    "metric": "RMSE / residual structure near conjunction",
                    "direction": "ambiguous",
                    "note": "プラズマ補正やバンド選択により、中心付近の形状・ノイズが変化し得る。",
                }
            ],
            diagnostics_outputs=cassini_out,
            what_to_check=[
                "中心（太陽会合）付近で残差が系統的に曲がる/非対称になるか",
                "βスイープRMSEの最小位置が安定か（ノイズ・選別の付け替えでβが動かないか）",
            ],
            notes="Cassini は β 拘束の主力。Phase 6.2 では『バンド/補正/選別』の付け替えで整合を作れないことを確認する。",
        )
    )
    cassini_sys.append(
        _sys_item(
            sys_id="cassini_data_selection",
            title="データ選別（source, 時間窓, station/filter）",
            category="data_selection",
            status="supported",
            knobs=[
                _knob(
                    key="source",
                    knob_type="categorical",
                    apply={"mode": "cli_flag", "flag": "--source", "script": main_script.get("path")},
                    values=["pds_tdf", "pds_tdf_raw", "pds_odf_raw", "digitized"],
                    default="pds_tdf",
                    note="観測系列の取り方（PDS一次データTDF/ODF or 論文図デジタイズ）",
                ),
                _knob(
                    key="tdf_sample_type",
                    knob_type="categorical",
                    apply={"mode": "cli_flag", "flag": "--tdf-sample-type", "script": main_script.get("path")},
                    values=["high", "low", "any"],
                    default="high",
                    note="high-rate/low-rate Doppler",
                ),
                _knob(
                    key="tdf_stations",
                    knob_type="string",
                    apply={"mode": "cli_flag", "flag": "--tdf-stations", "script": main_script.get("path")},
                    note="comma-separated station IDs（未指定=全て）",
                ),
            ],
            expected_effect=[
                {
                    "observable": "y(t)",
                    "metric": "coverage / RMSE / correlation",
                    "direction": "ambiguous",
                    "note": "データ系列の定義やフィルタでノイズ/欠測/バイアスが変わる。",
                }
            ],
            diagnostics_outputs=cassini_out,
            what_to_check=[
                "source 切替で形状が本質的に変わるか（digitized は参照用）",
                "station/filter 切替で β 推定が大きく動かないか",
            ],
        )
    )
    cassini_sys.append(
        _sys_item(
            sys_id="cassini_noise_model",
            title="ノイズモデル（binning/stat, outlier filters, detrend）",
            category="noise_model",
            status="supported",
            knobs=[
                _knob(
                    key="tdf_bin_seconds",
                    knob_type="int",
                    apply={"mode": "cli_flag", "flag": "--tdf-bin-seconds", "script": main_script.get("path")},
                    default=3600,
                    note="平滑化の時間ビン幅 [秒]",
                ),
                _knob(
                    key="tdf_bin_stat",
                    knob_type="categorical",
                    apply={"mode": "cli_flag", "flag": "--tdf-bin-stat", "script": main_script.get("path")},
                    values=["mean", "median"],
                    default="median",
                    note="時間ビン内の代表値",
                ),
                _knob(
                    key="tdf_detrend_poly_order",
                    knob_type="int",
                    apply={"mode": "cli_flag", "flag": "--tdf-detrend-poly-order", "script": main_script.get("path")},
                    default=0,
                    note="デトレンドの多項式次数",
                ),
                _knob(
                    key="tdf_detrend_exclude_inner_days",
                    knob_type="float",
                    apply={"mode": "cli_flag", "flag": "--tdf-detrend-exclude-inner-days", "script": main_script.get("path")},
                    default=5.0,
                    note="中心部を除外してデトレンド（|t|<days）",
                ),
            ],
            expected_effect=[
                {
                    "observable": "residuals",
                    "metric": "RMSE / residual width",
                    "direction": "decrease",
                    "note": "binning を強くすると見かけのノイズは減るが、系統形状の破壊にも注意。",
                }
            ],
            diagnostics_outputs=cassini_out,
            what_to_check=[
                "binning や detrend の強さで中心の形状（時間対称性/尖り）が崩れていないか",
                "“良く見せる”方向の付け替えでβが動いていないか",
            ],
        )
    )
    templates.append(
        {
            "test_id": test.get("id"),
            "topic": test.get("topic"),
            "name": test.get("name"),
            "main_script": {k: main_script.get(k) for k in ("path", "repro", "role") if k in main_script},
            "primary_outputs": primary_outputs,
            "systematics": cassini_sys,
        }
    )

    # --- Viking ---
    test, main_script, primary_outputs = _t("viking_shapiro_peak")
    viking_out = [
        "output/private/viking/viking_shapiro_result.csv",
        "output/private/viking/viking_p_model_vs_measured_no_arrow.png",
    ]
    viking_sys: List[Dict[str, Any]] = []
    viking_sys.append(
        _sys_item(
            sys_id="viking_peak_selection",
            title="幾何ピークの選択（start/stop/step のスキャン）",
            category="data_selection",
            status="supported",
            knobs=[
                _knob(
                    key="start",
                    knob_type="date_string",
                    apply={"mode": "cli_flag", "flag": "--start", "script": main_script.get("path")},
                    note="スキャン開始日",
                ),
                _knob(
                    key="stop",
                    knob_type="date_string",
                    apply={"mode": "cli_flag", "flag": "--stop", "script": main_script.get("path")},
                    note="スキャン終了日",
                ),
                _knob(
                    key="step",
                    knob_type="string",
                    apply={"mode": "cli_flag", "flag": "--step", "script": main_script.get("path")},
                    note="サンプリング間隔（例: '4 h'）",
                ),
            ],
            expected_effect=[
                {
                    "observable": "peak Shapiro delay",
                    "metric": "max_delay_us",
                    "direction": "ambiguous",
                    "note": "サンプリング分解能が粗いとピークを外し、最大値が下がる（系統的に過小評価）。",
                }
            ],
            diagnostics_outputs=viking_out,
            what_to_check=[
                "step を細かくしたとき最大値が収束するか（ピーク取りの系統）",
                "HORIZONSキャッシュの有無で結果が変わらないか（再現性）",
            ],
        )
    )
    viking_sys.append(
        _sys_item(
            sys_id="viking_time_sync_baseline",
            title="時刻同期/基線の定義（中心系、光行時間反復）",
            category="model_assumption",
            status="todo",
            knobs=[
                _knob(
                    key="light_time_iteration",
                    knob_type="bool",
                    apply={"mode": "not_parameterized", "note": "現状は同時刻位置で近似（往復光行時間の反復は未実装）"},
                    note="厳密には反復で解く必要がある（ただしオーダー確認では小さい）",
                )
            ],
            expected_effect=[
                {
                    "observable": "Shapiro delay",
                    "metric": "systematic bias",
                    "direction": "ambiguous",
                    "note": "同時刻近似によりピーク時刻/最大値がわずかにずれる可能性。",
                }
            ],
            diagnostics_outputs=viking_out,
            what_to_check=[
                "反復光行時間を入れた場合の差（μsオーダーで十分小さいか）",
            ],
            notes="Step 6.2.3 の自動バッチ前に、必要なら viking_shapiro_check.py へ追加する。",
        )
    )
    templates.append(
        {
            "test_id": test.get("id"),
            "topic": test.get("topic"),
            "name": test.get("name"),
            "main_script": {k: main_script.get(k) for k in ("path", "repro", "role") if k in main_script},
            "primary_outputs": primary_outputs,
            "systematics": viking_sys,
        }
    )

    # --- Mercury ---
    test, main_script, primary_outputs = _t("mercury_perihelion_precession")
    mercury_out = [
        "output/private/mercury/mercury_precession_metrics.json",
        "output/private/mercury/mercury_orbit.png",
    ]
    mercury_sys: List[Dict[str, Any]] = []
    mercury_sys.append(
        _sys_item(
            sys_id="mercury_unmodeled_perturbations",
            title="未モデル摂動の境界（他惑星/J2/小天体など）",
            category="model_assumption",
            status="partial",
            knobs=[
                _knob(
                    key="unmodeled_perturbations",
                    knob_type="enum",
                    apply={"mode": "documentation_only", "note": "現状の簡易積分は摂動を入れていない（精密暦ではない）"},
                    values=["planets", "solar_J2", "asteroids", "post_newtonian_higher_order"],
                    note="どの摂動が支配的かを分解し、比較の限界を明示する。",
                )
            ],
            expected_effect=[
                {
                    "observable": "perihelion shift",
                    "metric": "arcsec/century",
                    "direction": "ambiguous",
                    "note": "未モデル摂動は追加の近日点移動を与え、P-model/GRの差分をマスクし得る。",
                }
            ],
            diagnostics_outputs=mercury_out,
            what_to_check=[
                "『このスクリプト単体では何を言えて、何を言えないか』の境界を固定する",
            ],
        )
    )
    mercury_sys.append(
        _sys_item(
            sys_id="mercury_numerics",
            title="数値積分の分解能（dt, integrator）",
            category="numerics",
            status="todo",
            knobs=[
                _knob(
                    key="dt_seconds",
                    knob_type="float",
                    apply={"mode": "not_parameterized", "note": "現状の dt はスクリプト内定数（CLI化は未実装）"},
                    note="dt を変えて近日点移動が収束することを確認する。",
                )
            ],
            expected_effect=[
                {
                    "observable": "perihelion shift",
                    "metric": "stability",
                    "direction": "ambiguous",
                    "note": "dt が粗いと数値誤差が系統として混入する。",
                }
            ],
            diagnostics_outputs=mercury_out,
            what_to_check=["dt を半分にして値が変わらないか（収束性）"],
        )
    )
    templates.append(
        {
            "test_id": test.get("id"),
            "topic": test.get("topic"),
            "name": test.get("name"),
            "main_script": {k: main_script.get(k) for k in ("path", "repro", "role") if k in main_script},
            "primary_outputs": primary_outputs,
            "systematics": mercury_sys,
        }
    )

    # --- GPS ---
    test, main_script, primary_outputs = _t("gps_satellite_clock")
    gps_out = [
        "output/private/gps/summary_batch.csv",
        "output/private/gps/gps_rms_compare.png",
        "output/private/gps/gps_compare_metrics.json",
    ]
    gps_sys: List[Dict[str, Any]] = []
    gps_sys.append(
        _sys_item(
            sys_id="gps_bias_drift_policy",
            title="バイアス+ドリフト除去の前提（デトレンドの取り方）",
            category="estimator_policy",
            status="partial",
            knobs=[
                _knob(
                    key="detrend_policy",
                    knob_type="enum",
                    apply={"mode": "documentation_only", "note": "現状 compare_clocks.py は bias+linear drift を除去（固定）"},
                    values=["none", "bias_only", "bias_plus_drift"],
                    default="bias_plus_drift",
                    note="衛星ごとの定数オフセットと周波数オフセットを除去する前提。",
                )
            ],
            expected_effect=[
                {
                    "observable": "clock residuals (IGS - model)",
                    "metric": "RMS",
                    "direction": "decrease",
                    "note": "強いデトレンドほど RMS は下がるが、物理的差分も吸収し得る。",
                }
            ],
            diagnostics_outputs=gps_out,
            what_to_check=[
                "同じ detrend で BRDC と P-model を比較しているか（片側だけ有利になっていないか）",
                "detrend を弱めた場合に衛星依存が顕在化するか",
            ],
            notes="Step 6.2.3 に向け、detrend を CLI 化して感度試験できる形にするのが望ましい。",
        )
    )
    gps_sys.append(
        _sys_item(
            sys_id="gps_rotation_corrections",
            title="地球自転補正（基準点/衛星速度の ω×r）",
            category="model_assumption",
            status="supported",
            knobs=[
                _knob(
                    key="ref_include_earth_rotation",
                    knob_type="bool",
                    apply={"mode": "cli_flag", "flag": "--ref-include-earth-rotation", "script": main_script.get("path")},
                    default=False,
                    note="地上基準点速度に自転を含める",
                ),
                _knob(
                    key="no_sat_earth_rotation",
                    knob_type="bool",
                    apply={"mode": "cli_flag", "flag": "--no-sat-earth-rotation", "script": main_script.get("path")},
                    default=False,
                    note="衛星速度の ω×r 補正を無効化（デバッグ/系統切り分け）",
                ),
            ],
            expected_effect=[
                {
                    "observable": "rate_rel(t)",
                    "metric": "systematic drift",
                    "direction": "ambiguous",
                    "note": "速度推定が変わり、P-modelの相対レートに系統が入る可能性。",
                }
            ],
            diagnostics_outputs=gps_out,
            what_to_check=["補正ON/OFFで系統的な差がどの衛星で出るか（衛星依存の切り分け）"],
        )
    )
    gps_sys.append(
        _sys_item(
            sys_id="gps_satellite_dependence",
            title="衛星依存（衛星ごとのRMS分布、外れ衛星）",
            category="data_selection",
            status="supported",
            knobs=[
                _knob(
                    key="satellite_selection",
                    knob_type="enum",
                    apply={"mode": "not_parameterized", "note": "現状は共通集合の全G衛星。除外/限定はコード変更が必要。"},
                    note="外れ衛星が支配しない評価指標（中央値/分位）も併用する。",
                )
            ],
            expected_effect=[
                {
                    "observable": "summary_batch",
                    "metric": "RMS distribution width",
                    "direction": "ambiguous",
                    "note": "衛星固有のクロック特性や軌道誤差でRMSが偏る。",
                }
            ],
            diagnostics_outputs=gps_out,
            what_to_check=[
                "summary_batch.csv の分布（外れ衛星が結論を支配していないか）",
                "G01など代表衛星の時系列で、位相/季節性/欠測があるか",
            ],
        )
    )
    templates.append(
        {
            "test_id": test.get("id"),
            "topic": test.get("topic"),
            "name": test.get("name"),
            "main_script": {k: main_script.get(k) for k in ("path", "repro", "role") if k in main_script},
            "primary_outputs": primary_outputs,
            "systematics": gps_sys,
        }
    )

    # --- LLR ---
    test, main_script, primary_outputs = _t("llr_batch")
    llr_out = [
        "output/private/llr/batch/llr_batch_summary.json",
        "output/private/llr/batch/llr_batch_metrics.csv",
        "output/private/llr/batch/llr_data_coverage.csv",
    ]
    llr_sys: List[Dict[str, Any]] = []
    llr_sys.append(
        _sys_item(
            sys_id="llr_coverage_bias",
            title="coverage（年/局/反射器）の偏りとRMSへの影響",
            category="data_selection",
            status="supported",
            knobs=[
                _knob(
                    key="min_points",
                    knob_type="int",
                    apply={"mode": "cli_flag", "flag": "--min-points", "script": main_script.get("path")},
                    default=6,
                    note="station×target グループを含める最小点数",
                ),
                _knob(
                    key="max_groups",
                    knob_type="int",
                    apply={"mode": "cli_flag", "flag": "--max-groups", "script": main_script.get("path")},
                    default=0,
                    note="グループ数の上限（0=全て）",
                ),
            ],
            expected_effect=[
                {
                    "observable": "RMS metrics",
                    "metric": "overall RMS / station-target RMS",
                    "direction": "ambiguous",
                    "note": "点数が少ないグループは外れ値に敏感で、RMSを不安定化させる。",
                }
            ],
            diagnostics_outputs=llr_out,
            what_to_check=[
                "llr_data_coverage.csv で『どの局/反射器が支配しているか』を固定表示",
                "min_points を上げても結論が変わらないか（サンプル不足の切り分け）",
            ],
        )
    )
    llr_sys.append(
        _sys_item(
            sys_id="llr_time_tag",
            title="時刻タグ解釈（tx/rx/mid/auto）",
            category="estimator_policy",
            status="supported",
            knobs=[
                _knob(
                    key="time_tag_mode",
                    knob_type="categorical",
                    apply={"mode": "cli_flag", "flag": "--time-tag-mode", "script": main_script.get("path")},
                    values=["tx", "rx", "mid", "auto"],
                    note="局別に最適を採るか（auto）/固定か",
                )
            ],
            expected_effect=[
                {
                    "observable": "range residuals",
                    "metric": "bias/RMS",
                    "direction": "ambiguous",
                    "note": "時刻タグの取り方で幾何と補正の位相が変わり、残差が変化する。",
                }
            ],
            diagnostics_outputs=llr_out,
            what_to_check=["局ごとの time_tag_mode 最適化が一貫しているか（恣意性を排除）"],
        )
    )
    llr_sys.append(
        _sys_item(
            sys_id="llr_outliers",
            title="外れ値ゲート（MAD×sigma と絶対ns閾値）",
            category="noise_model",
            status="supported",
            knobs=[
                _knob(
                    key="outlier_clip_sigma",
                    knob_type="float",
                    apply={"mode": "cli_flag", "flag": "--outlier-clip-sigma", "script": main_script.get("path")},
                    default=8.0,
                    note="MAD倍率",
                ),
                _knob(
                    key="outlier_clip_ns",
                    knob_type="float",
                    apply={"mode": "cli_flag", "flag": "--outlier-clip-ns", "script": main_script.get("path")},
                    default=100.0,
                    note="最低絶対閾値 [ns]",
                ),
            ],
            expected_effect=[
                {
                    "observable": "RMS",
                    "metric": "RMS",
                    "direction": "decrease",
                    "note": "強い外れ値除外ほど RMS は下がるが、データ品質の違いを隠す危険もある。",
                }
            ],
            diagnostics_outputs=llr_out,
            what_to_check=[
                "llr_outliers.csv が局/年/反射器に偏っていないか",
                "外れ値ポリシーを動かしても“改善の有無”が反転しないか",
            ],
        )
    )
    templates.append(
        {
            "test_id": test.get("id"),
            "topic": test.get("topic"),
            "name": test.get("name"),
            "main_script": {k: main_script.get(k) for k in ("path", "repro", "role") if k in main_script},
            "primary_outputs": primary_outputs,
            "systematics": llr_sys,
        }
    )

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 6, "step": "6.2.2", "name": "弱場テスト系統分解テンプレ固定"},
        "source_matrix": {
            "path": _relpath(_ROOT / "output" / "private" / "summary" / "weak_field_test_matrix.json"),
            "generated_utc": matrix.get("generated_utc"),
        },
        "templates": templates,
        "outputs": {"weak_field_systematics_templates_json": "output/private/summary/weak_field_systematics_templates.json"},
    }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate weak-field systematics decomposition templates (Phase 6 / Step 6.2.2).")
    ap.add_argument(
        "--matrix",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "weak_field_test_matrix.json"),
        help="Input matrix JSON (default: output/private/summary/weak_field_test_matrix.json).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "weak_field_systematics_templates.json"),
        help="Output JSON path (default: output/private/summary/weak_field_systematics_templates.json).",
    )
    args = ap.parse_args(argv)

    matrix_path = Path(args.matrix)
    # 条件分岐: `not matrix_path.is_absolute()` を満たす経路を評価する。
    if not matrix_path.is_absolute():
        matrix_path = (_ROOT / matrix_path).resolve()

    # 条件分岐: `not matrix_path.exists()` を満たす経路を評価する。

    if not matrix_path.exists():
        print(f"[err] missing matrix: {matrix_path}")
        return 2

    out_path = Path(args.out)
    # 条件分岐: `not out_path.is_absolute()` を満たす経路を評価する。
    if not out_path.is_absolute():
        out_path = (_ROOT / out_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = _read_json(matrix_path)
    payload = build_templates(matrix)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {_relpath(out_path)}")

    worklog.append_event(
        {
            "event_type": "summary_weak_field_systematics_templates",
            "phase": "6.2.2",
            "inputs": {"weak_field_test_matrix_json": _relpath(matrix_path)},
            "outputs": {"weak_field_systematics_templates_json": _relpath(out_path)},
        }
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
