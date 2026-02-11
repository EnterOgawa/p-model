# Step 7.4：ベル不等式の棄却（一次データ再解析）

## 目的（このStepで固定したいこと）

ベル不等式（CHSH/CH/Eberhard など）は、観測された統計量を「同一母集団の平均」として扱う前提を暗黙に置く。
しかし time-tag（検出時刻）に基づく事象選別（coincidence window, pairing algorithm 等）が入ると、
採用される事象集合が測定条件（settings）に依存し得る。

このStepでは、**一次データ**（time-tag 生ログ）を用いて、
「設定に依存する時間構造」が実データに現れているかを、再現可能な形で確認する。

注意：ここで行うのは **NIST公式の p-value 再現ではない**（公式解析の全チェーンを再構築するのは別作業）。
本Stepの狙いは「独立確率仮定（中立な選別）が破れ得る入口」を、一次データで可視化し、定量指標を固定すること。

---

## 一次データ（NIST belltestdata）

NIST が公開している time-tag データ（Shalm et al. 2015 実験のリポジトリ）を使用する。
ファイル/フォルダ構造と raw record 仕様は以下のPDFにまとまっている：

- `data/quantum/sources/nist_belltestdata/File_Folder_Descriptions.pdf`
- `data/quantum/sources/nist_belltestdata/File_Folder_Descriptions_Addendum_2017_02.pdf`

本リポジトリでは **Windows で扱えるサイズ**として、training run（NIST記述では mode-lock が悪い）をまず固定し、
解析パイプラインの再現性を優先する：

- `.../03_31_CH_pockel_100kHz.run4.afterTimingfix2_training.*.dat.compressed.zip`

（この選択は “結論” ではなく “再解析の入口固定” を目的とする。より良い run への拡張は次段で行う。）

---

## 解析の方針（最小）

1. Alice/Bob の time base は絶対値が異なるため、GPS PPS（channel 5）でオフセットを推定して整列する。
2. 各 side の click event（channel 0）について、
   - RNG setting（channel 2/4）を付与する
   - sync（channel 6）からの遅延（click delay）を計算する
3. click delay の分布が setting に依存するかを定量化する（KS距離）。
4. click timestamp の単純な greedy pairing を用い、coincidence window を掃引してペア数・設定別内訳がどう変わるかを出力する。

---

## 再現手順（コマンド）

### 1) データ取得（オンライン）

```powershell
python -B scripts/quantum/fetch_nist_belltestdata.py
```

（既に存在する場合は skip。`data/quantum/sources/nist_belltestdata/manifest.json` を更新。）

### 1b) run 一覧と「training 以外」への拡張（任意）

training run は「入口の固定」には便利だが、NIST 自身が mode-lock の悪さを明記しているため、頑健性確認には不十分である。
本リポジトリでは **より良い run を追加で取得し、同じ指標（delay分布/KS/window sweep）を再計算**できるようにする。

```powershell
# 利用可能な run base を列挙（alice/bob 両方が揃うものだけ表示）
python -B scripts/quantum/fetch_nist_belltestdata.py --list --date 2015_09_18
```

```powershell
# 例：より良い run を追加取得（大容量になり得るので safety limit を上げる）
python -B scripts/quantum/fetch_nist_belltestdata.py --run 03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking --max-gib 2.0
```

```powershell
# 参考：trial-based（公式解析に近い）入力となる build(hdf5) を列挙
python -B scripts/quantum/fetch_nist_belltestdata.py --list-hdf5 --date 2015_09_18
```

注意：
- `--list` で `[multipart]` と出る run は `.z01/.z02...` を含み、そのままでは扱えない（本スクリプトは現状非対応）。
- 大容量 run は回線とストレージを消費する。必要なら `--max-seconds` を使い、解析側で先頭のみでまず QC する。

### 2) 再解析（オフライン可）

```powershell
python -B scripts/quantum/nist_belltest_time_tag_reanalysis.py
```

（別runを解析する場合は `--alice-zip/--bob-zip` を指定し、`--out-tag` で出力を分ける。）

### 3) trial-based（公式解析の trial 定義に寄せる；任意）

processed_compressed/hdf5（build）には、sync×slot（16bit）で click が整理され、click 定義（radius等）が config として同梱されている。
これを用いて trial-based 集計を行い、coincidence-based（greedy window）とのアルゴリズム依存を定量化する：

```powershell
# build(hdf5) の取得（例：03_43 run）
python -B scripts/quantum/fetch_nist_belltestdata.py --run 03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking --hdf5 --max-gib 2.0
```

```powershell
# 依存：h5py（未導入なら）
python -m pip install -U h5py
```

```powershell
# trial-based 再解析（既定 out-tag は 03_43_afterfixingModeLocking_s3600）
python -B scripts/quantum/nist_belltest_trial_based_reanalysis.py
```

---

## 一次データ（Weihs et al. 1998；photon time-tag）

Weihs et al. 1998（光子）について、Alice/Bob 別 zip の time-tag 公開データ（Zenodo）を一次ソースとして固定し、
coincidence window の選択が CHSH |S| に与える感度（selection の入口）を再現可能な形で出力する。

### 1) データ取得（オンライン）

```powershell
python -B scripts/quantum/fetch_weihs1998_zenodo_7185335.py
```

（保存先：`data/quantum/sources/zenodo_7185335/`）

### 2) 再解析（オフライン可）

```powershell
# 既定：longdist/longdist1 を解析して window sweep を出力
python -B scripts/quantum/weihs1998_time_tag_reanalysis.py
```

出力：
- 図：`output/quantum/weihs1998_chsh_sweep__weihs1998_longdist_longdist1.png`
- 数値：`output/quantum/weihs1998_chsh_sweep_metrics__weihs1998_longdist_longdist1.json`
- sweep：`output/quantum/weihs1998_chsh_sweep__weihs1998_longdist_longdist1.csv`

注意：
- `*_C.dat` の bit encoding（setting/outcome）は一次資料の記述が揺れているため、本リポジトリの再解析では
  `--encoding bit0-setting` を既定とし、参照 window（`--ref-window-ns`）で固定した CHSH variant を全 window に適用する。

---

## 固定した成果物

- 図：`output/quantum/nist_belltest_time_tag_bias.png`
- 数値：`output/quantum/nist_belltest_time_tag_bias_metrics.json`
- window sweep：`output/quantum/nist_belltest_coincidence_sweep.csv`

### 追加（training 以外の run）

afterfixingModeLocking（03_43 run）の先頭 3600 秒（max_seconds=3600）でも、同じ指標が再現できることを固定した：

- 図：`output/quantum/nist_belltest_time_tag_bias__03_43_afterfixingModeLocking_s3600.png`
- 数値：`output/quantum/nist_belltest_time_tag_bias_metrics__03_43_afterfixingModeLocking_s3600.json`
- window sweep：`output/quantum/nist_belltest_coincidence_sweep__03_43_afterfixingModeLocking_s3600.csv`

（指標例：KS(A)≈0.029、KS(B)≈0.041。click delay の中央値が setting により ~150–220 ns 程度ずれる。）

### 追加（trial-based；build(hdf5)）

afterfixingModeLocking（03_43 run）について、build(hdf5) の click 定義（radius等）に従った trial-based 集計を固定した：

- 図：`output/quantum/nist_belltest_trial_based__03_43_afterfixingModeLocking_s3600.png`
- 数値：`output/quantum/nist_belltest_trial_based_metrics__03_43_afterfixingModeLocking_s3600.json`
- 集計：`output/quantum/nist_belltest_trial_based_counts__03_43_afterfixingModeLocking_s3600.csv`

### 追加（NIST 追加run；装置条件差）

追加runとして、completeblind（17_04 run；multipart z01+zip）でも同じ指標を再計算し、run条件への依存を切り分ける入口を固定した：

- coincidence-based（先頭3600秒）：
  - 図：`output/quantum/nist_belltest_time_tag_bias__17_04_completeblind_s3600.png`
  - 数値：`output/quantum/nist_belltest_time_tag_bias_metrics__17_04_completeblind_s3600.json`
  - sweep：`output/quantum/nist_belltest_coincidence_sweep__17_04_completeblind_s3600.csv`
- trial-based（build；sync×slot）：
  - 図：`output/quantum/nist_belltest_trial_based__17_04_completeblind_s3600.png`
  - 数値：`output/quantum/nist_belltest_trial_based_metrics__17_04_completeblind_s3600.json`
  - 集計：`output/quantum/nist_belltest_trial_based_counts__17_04_completeblind_s3600.csv`

追加runとして、light cone shift + 200ns addition delay（21_15 run）でも同じ指標を再計算し、run条件への依存を切り分ける入口を固定した：

- coincidence-based（先頭3600秒）：
  - 図：`output/quantum/nist_belltest_time_tag_bias__21_15_200nsadditiondelay_lightconeshift_s3600.png`
  - 数値：`output/quantum/nist_belltest_time_tag_bias_metrics__21_15_200nsadditiondelay_lightconeshift_s3600.json`
  - sweep：`output/quantum/nist_belltest_coincidence_sweep__21_15_200nsadditiondelay_lightconeshift_s3600.csv`
- trial-based（build；sync×slot）：
  - 図：`output/quantum/nist_belltest_trial_based__21_15_200nsadditiondelay_lightconeshift_s3600.png`
  - 数値：`output/quantum/nist_belltest_trial_based_metrics__21_15_200nsadditiondelay_lightconeshift_s3600.json`
  - 集計：`output/quantum/nist_belltest_trial_based_counts__21_15_200nsadditiondelay_lightconeshift_s3600.csv`

追加runとして、light cone shift + 200ns reduced delay（22_20 run）でも同様に先頭3600秒の coincidence-based と trial-based(build) を固定した：

- coincidence-based（先頭3600秒）：
  - 図：`output/quantum/nist_belltest_time_tag_bias__22_20_200nsreduceddelay_lightconeshift_s3600.png`
  - 数値：`output/quantum/nist_belltest_time_tag_bias_metrics__22_20_200nsreduceddelay_lightconeshift_s3600.json`
  - sweep：`output/quantum/nist_belltest_coincidence_sweep__22_20_200nsreduceddelay_lightconeshift_s3600.csv`
- trial-based（build；sync×slot）：
  - 図：`output/quantum/nist_belltest_trial_based__22_20_200nsreduceddelay_lightconeshift_s3600.png`
  - 数値：`output/quantum/nist_belltest_trial_based_metrics__22_20_200nsreduceddelay_lightconeshift_s3600.json`
  - 集計：`output/quantum/nist_belltest_trial_based_counts__22_20_200nsreduceddelay_lightconeshift_s3600.csv`

追加runとして、Classical RNG XOR（23_55 run）でも同様に先頭3600秒の coincidence-based と trial-based(build) を固定した：

- coincidence-based（先頭3600秒）：
  - 図：`output/quantum/nist_belltest_time_tag_bias__23_55_ClassicalRNGXOR_s3600.png`
  - 数値：`output/quantum/nist_belltest_time_tag_bias_metrics__23_55_ClassicalRNGXOR_s3600.json`
  - sweep：`output/quantum/nist_belltest_coincidence_sweep__23_55_ClassicalRNGXOR_s3600.csv`
- trial-based（build；sync×slot）：
  - 図：`output/quantum/nist_belltest_trial_based__23_55_ClassicalRNGXOR_s3600.png`
  - 数値：`output/quantum/nist_belltest_trial_based_metrics__23_55_ClassicalRNGXOR_s3600.json`
  - 集計：`output/quantum/nist_belltest_trial_based_counts__23_55_ClassicalRNGXOR_s3600.csv`

注意：
- build(hdf5) の構造は run により異なり、`badSyncInfo` 等が存在しない場合がある（本リポジトリの trial-based 再解析は欠損時に「bad sync=0」として扱う）。

---

## 読み方（本Stepが示すこと）

- setting ごとに click delay 分布が異なるなら、time-tag に基づく選別が **条件依存**になり得る。
- coincidence window や pairing の選択は、採用される事象集合（≒統計母集団）を変えるため、
  「その統計量をベル不等式へ直接代入してよいか」は自明ではない。

本プロジェクトの立場では、これは「非局所性の断定」ではなく、
**観測手続きに含まれる時間構造の前提検証が必要**という主張へ接続する。
