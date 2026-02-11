# GPS REPRO (offline)
ASCII preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

# GPS（衛星時計）検証：再現手順

目的：IGSの精密クロック（CLK, 準実測）に対して、放送暦（BRDC）とP-modelの両方を比較し、残差の時系列を出す。

注意：GNSSでは相対補正（近日点効果、`dt_rel = -2 r·v / c^2`）を別項として扱う慣例があるため、
本リポジトリでは **IGS CLK と比較する際は P-model 側でも dt_rel を除去**して整合させている。

---

## 入力（data/gps）

このスクリプトは固定ファイル名を参照する。
- `data/gps/BRDC00IGS_R_20252740000_01D_MN.rnx`（RINEX NAV）
- `data/gps/IGS0OPSFIN_20252740000_01D_15M_ORB.SP3`（SP3 orbit）
- `data/gps/IGS0OPSFIN_20252740000_01D_05M_CLK.CLK`（IGS clock）

Webから取得する場合（推奨）
- `python -B scripts/gps/fetch_igs_bkg.py`（既定: 2025-10-01 / DOY=274）
- 別日を指定する例：`python -B scripts/gps/fetch_igs_bkg.py --date 2025-10-01`

別日のデータで回す場合は、同名で差し替えるか、`scripts/gps/compare_clocks.py` 内のファイル名を編集する。

---

## ワンコマンド再現（PowerShell）

`python -B scripts/gps/compare_clocks.py; python -B scripts/gps/plot.py`

任意の設定（例）
- 地上基準点の速度に地球自転を含める：`python -B scripts/gps/compare_clocks.py --ref-include-earth-rotation`
- 衛星速度の ω×r 補正を無効化（デバッグ用）：`python -B scripts/gps/compare_clocks.py --no-sat-earth-rotation`

---

## 生成物（output/gps）

- `output/gps/summary_batch.csv`（衛星ごとのRMSなど）
- `output/gps/residual_precise_Gxx.csv`（衛星ごとの残差時系列）
- `output/gps/gps_clock_residuals_all_31.png`（全衛星重ね描き）
- `output/gps/gps_residual_compare_G01.png`（G01の例：BRDC-IGS と P-model-IGS）
- `output/gps/gps_rms_compare.png`（全衛星：残差RMSの比較）
- `output/gps/gps_relativistic_correction_G02.png`（相対補正（近日点効果）：標準式 vs P-model）
- `output/gps/gps_compare_metrics.json`（指標JSON）
