# 付録：LLR（月レーザー測距） 全グラフ

本文（4.2）では代表図のみを掲載した。ここでは、LLR（Lunar Laser Ranging）の検証で生成している主要グラフを **全て** まとめて掲載する（診断・切り分けを含む）。

再現（固定バッチ）：
- 再現コマンドと固定アーティファクト一覧は、検証用資料（Verification Materials）を正とする。[PModelVerificationMaterials]

## time-tag / 入力データの確認

- `output/llr/batch/llr_time_tag_selection_by_station.png`
- `output/llr/llr_primary_tof_timeseries.png`
- `output/llr/llr_primary_range_timeseries.png`

## バッチ要約（RMS）

- `output/llr/batch/llr_rms_improvement_overall.png`
- `output/llr/batch/llr_rms_by_station_target.png`
- `output/llr/batch/llr_rms_by_station_month.png`
- `output/llr/batch/llr_grsm_rms_by_target_month.png`
- `output/llr/batch/llr_grsm_rms_by_month_models.png`

## 残差 vs 仰角（系統の入口）

- `output/llr/batch/llr_apol_residual_vs_elevation.png`
- `output/llr/batch/llr_grsm_residual_vs_elevation.png`
- `output/llr/batch/llr_matm_residual_vs_elevation.png`
- `output/llr/batch/llr_wetl_residual_vs_elevation.png`

## 局座標ソースの影響

- `output/llr/batch/llr_station_coord_delta_pos_eop.png`
- `output/llr/coord_compare/llr_station_coords_rms_compare.png`
- `output/llr/coord_compare/llr_grsm_monthly_rms_pos_eop_vs_slrlog.png`

## 寄与の切り分け（追加図）

- `output/llr/batch/llr_shapiro_ablations_overall.png`
- `output/llr/batch/llr_tide_ablations_overall.png`
