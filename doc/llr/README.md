# LLR REPRO (offline + optional network)
ASCII preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

Offline replay (for Horizons-based overlay):
- First run downloads Horizons vectors and saves them to output/llr/horizons_cache/.
- Later runs auto-use cache (no network). To force offline, set HORIZONS_OFFLINE=1.

# 月レーザー測距（LLR: Lunar Laser Ranging）検証：再現手順

目的：CRDの Normal Point（record "11"）から、観測TOF/距離の簡易可視化（オフライン）と、HORIZONS幾何を使ったP-model重ね合わせ（ネットワーク）を行う。

---

## 入力

- サンプルCRD：`data/llr/demo_llr_like.crd`
- 実データを使う場合：`data/llr/` にCRD/NPTを置く（`.crd`/`.npt`/`.np2`/`.gz`対応）
- Webから取得する場合（推奨）：
  - `python -B scripts/llr/fetch_llr_edc.py --target apollo11 --latest`
  - 取得した実データは `data/llr/edc/...` にキャッシュされ、解析既定入力として `data/llr/llr_primary.np2` にコピーされる。
  - 追加（局情報）：`python -B scripts/llr/fetch_station_edc.py`（観測局の site log を取得して `data/llr/stations/` にキャッシュ）

---

## 1) オフライン quicklook（推奨）

`python -B scripts/llr/llr_crd_quicklook.py data/llr/llr_primary.np2 --assume-two-way`

（デモ入力で試す場合）
`python -B scripts/llr/llr_crd_quicklook.py data/llr/demo_llr_like.crd --assume-two-way`

生成物（output/llr）
- `output/llr/demo_llr_like_npt11.csv`
- `output/llr/demo_llr_like_summary.json`
- `output/llr/demo_llr_like_range_timeseries.png`
- `output/llr/demo_llr_like_tof_timeseries.png`

---

## 2) P-model重ね合わせ（HORIZONS幾何、ネットワーク必要）

`python -B scripts/llr/llr_pmodel_overlay_horizons_noargs.py`

生成物（output/llr/out_llr）
- `output/llr/out_llr/<stem>_table.csv`
- `output/llr/out_llr/<stem>_overlay_tof.png`
- `output/llr/out_llr/<stem>_residual.png`

補足：
- 入力CRDは `data/llr/llr_primary.*` があればそれを優先し、無ければ `data/llr/` 配下を探索して先頭1件を使う。
- time-tag（観測時刻の解釈）は既定で `auto`。`output/llr/batch/llr_time_tag_best_by_station.json` があれば局別最適（tx/rx/mid）を自動で使う。強制する場合は `LLR_TIME_TAG=tx|rx|mid|auto` を指定する。
- βは `scripts/llr/llr_pmodel_overlay_horizons_noargs.py` の `DEFAULT_BETA` を編集する。

---

## 3) ns級モデルの追加依存（SPICE / 反射器カタログ）

### 3.1 月の回転（MOON_PA_DE421）
ns級の整合では、月回転を近似（IAU式など）ではなく SPICE の `MOON_PA_DE421` で扱う。

- 取得：`python -B scripts/llr/fetch_moon_kernels_naif.py`
- 保存：`data/llr/kernels/naif/`

### 3.2 反射器座標（DE421 principal-axis）
反射器の月固定座標は、一次ソース（Murphy et al., arXiv:1009.5720 / Icarus DOI:10.1016/j.icarus.2010.11.010）の Table 4/5 から生成する。

- 生成：`python -B scripts/llr/fetch_reflector_catalog.py`
- 出力：`data/llr/reflectors_de421_pa.json`
- 注意：Table 4 の座標には「静的潮汐成分」が既に含まれる。未変形Moonへ戻すときは Table 5 を差し引く。

### 3.3 海洋潮汐荷重（TOC, IMLS HARPOS）
観測局座標の微小変位（cm級）が、ns級では無視できない場合があるため、海洋潮汐荷重（TOC）を追加できるようにする。

- 取得：`python -B scripts/llr/fetch_ocean_loading_imls.py`
- 保存：`data/llr/ocean_loading/toc_fes2014b_harmod.hps`

---

## 4) バッチ検証（複数局×複数反射器）

### 4.1 EDC（月次NP2）の一括取得（推奨）
`python -B scripts/llr/fetch_llr_edc_batch.py`

- 保存: `data/llr/edc/`
- マニフェスト: `data/llr/llr_edc_batch_manifest.json`
- 観測局ログ（slrlog）も同時に `data/llr/stations/` へキャッシュする

### 4.2 バッチ評価（P-model vs 観測）
初回（オンライン、Horizonsキャッシュ作成）:
`python -B scripts/llr/llr_batch_eval.py --chunk 50`

2回目以降（オフライン再現）:
`python -B scripts/llr/llr_batch_eval.py --offline --chunk 50`

生成物（output/llr/batch）:
- `llr_batch_metrics.csv`
- `llr_batch_summary.json`
- `llr_batch_points.csv`（全点診断：inlier/outlier列つき）
- `llr_outliers.csv`（外れ値一覧：問題ファイル/期間の特定用）
- `llr_outliers_diagnosis.csv`（外れ値の詳細診断：tx/rx/mid感度＋ターゲット混入推定、source_file:linenoで一次データ行へ追跡）
- `llr_rms_improvement_overall.png`
- `llr_rms_by_station_target.png`
- `llr_rms_ablations_overall.png`
- `llr_outliers_time_tag_sensitivity.png`（外れ値ごとの time-tag 感度：tx/rx/mid）
- `llr_outliers_target_mixing_sensitivity.png`（外れ値ごとの ターゲット混入感度：現ターゲット vs 推定ターゲット）
- `llr_residual_distribution.png`（残差分布：観測 - P-model、inlierのみ）

補足：
- ns級の統計は外れ値に非常に弱いので、station×targetごとに MAD（中央値絶対偏差）ベースで外れ値を除外してRMSを算出する。
- 外れ値が多い場合は time-tag（tx/rx/mid）の混在、局メタデータ（座標/epoch）、観測品質（低仰角/大気）、ファイル混入などを疑い、`llr_outliers.csv` から原因を特定する。
- time-tag の混在が疑わしい場合は `--time-tag-mode auto` で局別に tx/rx/mid を総当たりして最小RMSのモードを選ぶ（結果は `output/llr/batch/llr_time_tag_best_by_station.json`）。
- 「ターゲット混入（反射器取り違え）」が疑われる外れ値は、診断として `best_target_guess` / `suspected_target_mixing` を `llr_outliers_diagnosis.csv` に出力するが、**観測データのラベルは自動では書き換えない**（統計から除外し、source_file:lineno で一次データ行を確認する）。
- `include_daily=true` の場合、月次ファイルと日次ファイルで同一データが重複することがあるため、`llr_batch_eval.py` は（station,target,epoch,tof）の完全一致行を自動で重複排除する。
