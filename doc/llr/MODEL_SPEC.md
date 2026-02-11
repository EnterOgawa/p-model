# 月レーザー測距（LLR: Lunar Laser Ranging）モデル仕様（waveP）

目的：月レーザー測距（LLR: Lunar Laser Ranging）（CRD Normal Point / record 11）の観測TOFに対して、P-model（現状：幾何 + Shapiro）で再現可能な変動成分を比較し、モデルの誤差源を段階的に切り分ける。

---

## 1. 入力（一次データ・参照）

### 1.1 観測データ（EDC）
- DGFI-TUM EDC `npt_crd_v2`（`.np2`）
- 取得：`scripts/llr/fetch_llr_edc.py` / `scripts/llr/fetch_llr_edc_batch.py`
- 保存：`data/llr/edc/<target>/<year>/<file>.np2`

### 1.2 観測局（site log）
- DGFI-TUM EDC `slrlog`（`.log`）
- 取得：`scripts/llr/fetch_station_edc.py`
- 保存：`data/llr/stations/<station>.json`（lat/lon/height）

### 1.3 反射器座標（一次ソース確定済み）
- Murphy et al., “Laser Ranging to the Lost Lunokhod 1 Reflector”
  - arXiv:1009.5720
  - Icarus DOI: 10.1016/j.icarus.2010.11.010
  - Table 4（DE421 principal-axis 座標）、Table 5（静的潮汐オフセット）
- 生成：`scripts/llr/fetch_reflector_catalog.py`
- 保存：`data/llr/reflectors_de421_pa.json`

### 1.4 月回転（SPICE / DE421）
- NAIF kernels（MOON_PA_DE421）
- 取得：`scripts/llr/fetch_moon_kernels_naif.py`
- 保存：`data/llr/kernels/naif/`

### 1.5 エフェメリス（JPL Horizons）
- Earth→Moon / Earth→Sun：地心
- 観測局→Moon：topocentric（観測局の緯度経度高度を Horizons に渡す）
- キャッシュ：`output/llr/horizons_cache/`
- オフライン再現：`HORIZONS_OFFLINE=1`（キャッシュがある場合）

---

## 2. 観測量の定義

- 観測は CRD の record `11` の `TOF`（往復飛行時間、秒）
- CRDの `range_type` が未取得/不明な場合、LLRとして **two-way を仮定**（既定）

### 2.1 時刻タグ（重要）
観測時刻は record 11 の時刻（UTC）だが、実務上「送信/受信/中点」いずれのタグかが混在しうるため、まずはモードを固定して評価する。

- `LLR_TIME_TAG=auto`（既定）: 局別最適（`output/llr/batch/llr_time_tag_best_by_station.json`）を使う。無い場合は `tx`。
- `LLR_TIME_TAG=tx` : 観測時刻 = 送信時刻
- `LLR_TIME_TAG=rx` : 観測時刻 = 受信時刻
- `LLR_TIME_TAG=mid` : 観測時刻 = 反射（中点）時刻

---

## 3. 座標系の定義

### 3.1 反射器（Moon固定）
- `data/llr/reflectors_de421_pa.json` は **Moon DE421 principal-axis**（= SPICE `MOON_PA_DE421`）の月中心直交座標（m）

### 3.2 反射器→ICRF(J2000)
- `MOON_PA_DE421 -> J2000` の回転は SPICE の `pxform` を使用
  - カーネル：`moon_pa_de421_1900-2050.bpc` + `moon_080317.tf` + `naif0012.tls` + `pck00010.tpc`

### 3.3 観測局（地球固定）
- 観測局の緯度経度高度は site log から取得（WGS84）
- EOP（UT1-UTC/極運動等）は自前実装せず、**Horizons の topocentric 出力に委ねる**（現段階の整合方針）
- 局座標（ECEF, ITRF）は pos+eop（SINEX）が利用できる場合があり、バッチ評価では `--station-coords auto|slrlog|pos_eop` で切り替えて影響を定量化する。

---

## 4. モデル（現状）

`scripts/llr/llr_pmodel_overlay_horizons_noargs.py` のモデル構成（現時点）：

1) 幾何（two-way）  
  - geocenter → Moon center  
  - station → Moon center（topocentric）  
  - station → reflector（Moon固定点 + 月回転）

2) Shapiro遅延（太陽、two-way）  
  - 観測の送信脚 + 受信脚の和  
  - 係数：`2*β`（P-model側のパラメータとして扱う）

3) 対流圏遅延（two-way, 簡易）  
  - `scripts/llr/llr_batch_eval.py` で追加（段階的導入）  
  - Saastamoinen（hydrostatic + wet(任意)） + Niell Mapping Function（NMF）  
  - 観測側ですでに補正済みの可能性があるため、**ON/OFFして寄与を定量化**する  

4) 潮汐（暫定, ns級で寄与しうる）  
  - `scripts/llr/llr_batch_eval.py` に暫定実装  
    - 観測局：固体地球潮汐（Moon+Sun, h2/l2 の簡易式; radial+horizontal）  
    - 観測局：海洋潮汐荷重（TOC, IMLS HARPOS / FES2014b; UTC→TTで合成）  
    - 反射器：月体潮汐（Earth-driven, 簡易）  
  - 観測側ですでに補正済みの可能性が高いので、**必ずON/OFFして効果を確認**する  
  - 海洋荷重データ取得：`python -B scripts/llr/fetch_ocean_loading_imls.py`（保存: `data/llr/ocean_loading/`）

5) 定数オフセット整列  
  - 観測TOFとモデルTOFの平均差（定数）を差し引き、**変動成分の一致**を評価する

---

## 5. 出力（固定）

- quicklook：`output/llr/<stem>_tof_timeseries.png` 等（観測の可視化）
- overlay：`output/llr/out_llr/<stem>_overlay_tof.png`
- residual：`output/llr/out_llr/<stem>_residual.png`
- compare：`output/llr/out_llr/<stem>_residual_compare.png`
- metrics：`output/llr/out_llr/<stem>_metrics.json`

### 5.1 バッチ（複数局×複数反射器）
- 集計：`output/llr/batch/llr_batch_metrics.csv` / `output/llr/batch/llr_batch_summary.json`
- 診断（全点）：`output/llr/batch/llr_batch_points.csv`
  - `inlier_best` / `outlier_best`：station×targetごとの外れ値判定（MADベース）
  - `delta_best_raw_ns`：外れ値判定に使った（obs-model）の生差分（ns）
- 外れ値一覧：`output/llr/batch/llr_outliers.csv`（原因切り分け用：問題ファイル・期間の特定）
- 外れ値診断（詳細）：`output/llr/batch/llr_outliers_diagnosis.csv`（source_file:lineno で一次データ行へ追跡）
  - time-tag 感度：`output/llr/batch/llr_outliers_time_tag_sensitivity.png`（tx/rx/mid）
  - ターゲット混入感度：`output/llr/batch/llr_outliers_target_mixing_sensitivity.png`（現ターゲット vs 推定ターゲット）
  - 方針：混入が疑われても観測データのターゲットラベルは自動で書き換えない（統計から除外し、一次データ行で確認する）

--- 

## 6. 段階的な検証（ON/OFF切り分け）

「どの要素が残差を減らしたか」を説明できるよう、以下を段階的にON/OFFできる状態にする。

推奨の比較軸（最低限）：
- 地球中心 → 観測局（topocentric）にする効果
- 月中心 → 反射器（Moon固定点）にする効果
- 月回転 IAU近似 → SPICE（MOON_PA_DE421）にする効果
- 太陽Shapiro のON/OFF（将来：地球Shapiro等も追加）
- 地球Shapiro のON/OFF（寄与は小さい想定だが、ns級での定量化用）
- 対流圏（簡易）のON/OFF（station依存の系統残差の切り分け）
- 潮汐（地球固体潮, 簡易）のON/OFF（観測側補正との二重計上チェック）
- 潮汐（海洋荷重, TOC）のON/OFF（ns級での追加寄与の確認）

---

## 7. 現状ベースライン（2025-12-31）

この節は「今どこまで合っているか」を固定点として残すためのメモ。数値は `output/llr/batch/llr_batch_summary.json` の中央値（station×target のRMS中央値）をそのまま転記する。

再現コマンド（2回目以降オフライン）：

`python -B scripts/llr/llr_batch_eval.py --out-dir output/llr/batch --offline --time-tag-mode auto --chunk 50 --min-points 30`

条件：
- `β=1.0`（Shapiro係数は `2*β`）
- time-tag：`auto`（局別に tx/rx/mid を最小RMSで自動選択、無ければ tx）
- station coords：`auto`（pos+eop(SINEX)があれば優先）
- 外れ値ゲート：`sigma=8.0`、`min=100 ns`（station×target ごとに MAD ベース）
- 潮汐：固体地球潮汐（Moon+Sun, 簡易）＋月体潮汐（Earth-driven, 簡易）
- 海洋荷重（ocean loading）：未導入（次段階）

主要結果（中央値RMS, ns）：
- station→reflector（幾何＋太陽Shapiro）: **6.8145**
- + 対流圏（簡易）: **4.6026**
- + 潮汐（固体地球＋月体）: **4.3300**

追加（海洋荷重を導入した場合, 2026-01-01）：
- + 潮汐（固体地球＋海洋荷重＋月体）: **4.3245**

参考（切り分け）：
- Shapiro OFF（対流圏あり）: **5.6722**
- + 地球Shapiro（対流圏あり）: **4.6016**
