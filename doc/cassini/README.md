# CASSINI REPRO (offline-first)
ASCII preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

# Cassini（太陽会合）検証：再現手順

目的：Cassini 太陽会合（SCE1）のドップラー `y(t)` を **PDS一次データ**から復元し、P-model（β）で計算した `y(t)` と比較する。
補助：PDS TDF の `DOPPLER PSEUDORESIDUAL` は「疑似残差（観測-予測）」として扱い、参照Shapiro（β=1.0）を足し戻して論文図スケールの `y(t)` に復元して比較する。
さらに、復元系列が論文 Fig.2 デジタイズ点列と整合するかも確認する。

---

## 入力

- 幾何（Earth/Cassini のベクトル由来、モデルCSV）：`output/cassini/cassini_shapiro_y_full.csv`
- PDS一次データ（SCE1 TDF/ODF キャッシュ）：`data/cassini/pds_sce1/`
- 参考：デジタイズ点列（Fig.2）：`data/cassini/cassini_fig2_digitized_raw.csv`

注意：`output/cassini/cassini_shapiro_y_full.csv` は `scripts/cassini/cassini_shapiro_y_full.py` で再生成できる（HORIZONSアクセスが必要）。

---

## 手順

### 1) （必要なら）モデルCSVを再生成（ネットワーク必要）

`output/cassini/cassini_shapiro_y_full.csv` が無い／作り直したい場合：

`python scripts/cassini/cassini_shapiro_y_full.py --beta 1.0`

（βを変えたい場合も `--beta` で指定。内部で `gamma = 2*beta - 1` を使う。）

補足：すでに幾何CSV（`output/cassini/cassini_shapiro_y_full.csv`）がある場合、ネットワーク無しで `dt/y` を再計算したいときは
`--in-csv` を使える（出力は別名にするのが安全）：

`python -B scripts/cassini/cassini_shapiro_y_full.py --in-csv output/cassini/cassini_shapiro_y_full.csv --beta 1.0 --out output/cassini/cassini_shapiro_y_full_recomputed.csv`

### 2) Fig.2 と重ね合わせ（オフラインで可）

`python -B scripts/cassini/cassini_fig2_overlay.py`

主なオプション：
- `--model eq2|full`（既定：`eq2`。Cassini Eq(2) 近似／または Eq(1) の厳密微分）
- `--beta <float>`（既定：1.0）
- `--source pds_tdf|pds_tdf_raw|digitized|pds_odf_raw`（既定：`pds_tdf`）
  - `pds_tdf`: PDS一次データ（TDF）→ 平滑化＋デトレンド（推奨）
  - `pds_tdf_raw`: PDS TDFの生値（デバッグ用）
  - `digitized`: 論文図デジタイズ（参考/フォールバック）
- `--no-sweep`（βスイープを省略）
- `--no-plots`（CSVだけ出してPNGを省略）
- `--tdf-bin-seconds <int>`（既定：3600秒）
- `--tdf-detrend-poly-order <int>`（既定：0）
- `--tdf-detrend-exclude-inner-days <float>`（既定：5.0日）
- `--tdf-reconstruct-shapiro/--no-tdf-reconstruct-shapiro`（既定：enabled。疑似残差 + 参照Shapiro足し戻し）
- `--tdf-reconstruct-beta <float>`（既定：1.0。参照Shapiro復元で使う β）
- `--tdf-fixed-frac-bits <int>`（既定：0。固定小数点に見える場合のみ 18 等を試す）
- `--tdf-y-sign ±1`（既定：+1。TDF側の定義により反転が必要な場合がある）

---

## 生成物（output/cassini）

- `output/cassini/cassini_sce1_tdf_extracted.csv`（PDS TDF 抽出: y・符号/復元情報つき）
- `output/cassini/cassini_sce1_tdf_paperlike.csv`（平滑化＋デトレンド後の y(t)）
- `output/cassini/cassini_pds_vs_digitized.png`（整合チェック: PDS処理後 vs 論文図デジタイズ）
- `output/cassini/cassini_pds_vs_digitized_metrics.csv`（整合チェック指標）
- `output/cassini/cassini_fig2_matched_points.csv`（観測点列 + y_model + residual）
- `output/cassini/cassini_fig2_metrics.csv`（RMSE/MAE/corr など）
- `output/cassini/cassini_fig2_run_metadata.json`（実行条件・入力・パラメータ）
- `output/cassini/cassini_fig2_overlay_full.png`
- `output/cassini/cassini_fig2_overlay_zoom10d.png`
- `output/cassini/cassini_fig2_residuals.png`
- `output/cassini/cassini_beta_sweep_rmse.csv`
- `output/cassini/cassini_beta_sweep_rmse.png`
- `output/cassini/cassini_fig2_overlay_bestbeta_zoom10d.png`
