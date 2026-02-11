# EHT（Event Horizon Telescope）検証（ブラックホール「影」サイズ）

このフォルダは、P-model の **強重力（ブラックホール近傍）** を、EHT の観測値と比較するための資料です。

---

## 結論（現時点でどこまで「証明」できたか）

EHT は強重力領域への重要な窓口ですが、**現状のEHT公表値（リング直径）だけで「P-modelが正しい／GRが正しい」を強く判別することはできません**。

本リポジトリにおける EHT の位置づけは次の2つです。

- **整合性チェック（Phase 3）**：P-model（β≈1）と GR（Schwarzschild近似）が、観測レンジから大きく外れていないかを確認する
- **差分予測の入口（Phase 4）**：β を固定しても残る係数差（数%）を、将来の判別可能条件として提示する

判別力が限定的な主因は大きく2点です。

1) **EHTが推定しているのは「影（シャドウ）直径」ではなく、画像上の“リング”の代表径**である  
   リングとシャドウの対応には系統因子 `κ`（後述）が入り、`κ=1` が成り立つとは限りません。
2) **GR側も Kerr（スピンあり）では影形状が非円となり、“直径係数”に幅が出る**  
   さらに放射モデル・散乱・画像再構成が観測側の系統誤差を支配し得ます（特に Sgr A*）。

出力（強重力/EHT）：
- 整合チェック：`output/eht/eht_shadow_compare.png` / `output/eht/eht_shadow_zscores.png`
- 系統誤差（κ・Kerrレンジ）：`output/eht/eht_shadow_systematics.png`
- κ_fit（観測リング直径と理論シャドウ直径から逆算した κ）：`output/eht/eht_kappa_fit.png`
- κ精度要求（3σ判別に必要な κ の相対精度の目安）：`output/eht/eht_kappa_precision_required.png`
- δ精度要求（参考：GR論文の派生量 δ の精度スケール）：`output/eht/eht_delta_precision_required.png`
- κσ–ringσ トレードオフ（3σ判別の許容域；κ≈1）：`output/eht/eht_kappa_tradeoff.png`
- 差分予測：`output/eht/eht_shadow_differential.png`
- 形状指標（一次ソースレンジ）：`output/eht/eht_ring_morphology.png`
- 散乱のスケール感：`output/eht/eht_scattering_budget.png` / `output/eht/eht_refractive_scattering_limits.png`
- 散乱レンジの定量（比：min/max ÷ 必要σ）：`output/eht/eht_refractive_scattering_limits_metrics.json`
- 一次ソース整合（sha256）：`output/eht/eht_sources_integrity.json`
- 一次ソースの値アンカー（TeX内ヒット）：`output/eht/eht_primary_value_anchors.json`
- S2（GRAVITY）観測制約（f, f_SP）抽出：`output/eht/gravity_s2_constraints.json`（`scripts/eht/gravity_s2_constraints.py`）
- S2（GRAVITY）P-model差分予測（f, f_SP の必要精度の見積もり）：`output/eht/gravity_s2_pmodel_projection.json`（`scripts/eht/gravity_s2_pmodel_projection.py`）
- Sgr A* ring fitting tables（Paper III; appendix_ringfits + image_analysis tab:SgrA_ringfit）パース：`output/eht/eht_sgra_ringfit_table_metrics.json`（図：`output/eht/eht_sgra_ringfit_table_diameter_by_pipeline_descattered.png` / `output/eht/eht_sgra_ringfit_table_diameter_by_pipeline_on_sky.png`）
- Sgr A* 校正/非閉包系統（Paper III; observations + appendix_synthetic の gain table）抽出：`output/eht/eht_sgra_calibration_systematics_metrics.json`（`scripts/eht/eht_sgra_calibration_systematics_metrics.py`）
- Sgr A* variability noise model（Paper III; eq:PSD_noise / tab:premodeling）抽出：`output/eht/eht_sgra_variability_noise_model_metrics.json`（`scripts/eht/eht_sgra_variability_noise_model_metrics.py`）
- Sgr A* m-ring fits table（Paper V; tab:mringfits）抽出：`output/eht/eht_sgra_mringfits_table_metrics.json`（`scripts/eht/eht_sgra_mringfits_table_metrics.py`）
- Sgr A* Paper IV（ring→θ_g の GRMHD 校正；tab:alphacal）抽出：`output/eht/eht_sgra_paper4_alpha_calibration_metrics.json`（`scripts/eht/eht_sgra_paper4_alpha_calibration_metrics.py`）
- Sgr A* Paper IV（θ_g 推定；tab:thetag）抽出：`output/eht/eht_sgra_paper4_thetag_table_metrics.json`（`scripts/eht/eht_sgra_paper4_thetag_table_metrics.py`）
- Sgr A* Paper IV（形状パラメータ；tab:SgrAMorphology）抽出：`output/eht/eht_sgra_paper4_morphology_table_metrics.json`（`scripts/eht/eht_sgra_paper4_morphology_table_metrics.py`）
- Sgr A* Paper VI（Testing the Black Hole Metric）の制約表（shadow diameter / δ）抽出：`output/eht/eht_sgra_paper6_metric_constraints.json`（`scripts/eht/eht_sgra_paper6_metric_constraints.py`）
- κ誤差予算（必要精度 vs 現状スケール；散乱proxyも含む）：`output/eht/eht_kappa_error_budget.json`（図：`output/eht/eht_kappa_error_budget.png`；`scripts/eht/eht_kappa_error_budget.py`）
- 反証条件パック側は κ誤差予算の「採用κσ（現状スケール）」を優先して使用：`output/summary/decisive_falsification.json`（`kappa_systematics_source=budget`）

補足（δ：Schwarzschild shadow deviation; 参考）：
- Sgr A* Paper I は、Schwarzschild からの影直径のずれを表す δ を報告しています（Table「Measured Parameters of Sgr A*」）。
- δ は GR側の解析手順（校正データセット、放射モデル、散乱、再構成など）に依存する派生量であり、本プロジェクトでは「現状のGR解析の精度スケール」を示す参考指標として扱います（P-model検証は κ を明示して進める）。

---

## 何を比較するか（リング直径とシャドウ直径）

EHT は M87* と Sgr A* の「リング状の明るい構造」の角直径（µas）を公表しています。  
一般向けには「ブラックホールの影の大きさ」に近い量として解釈されることが多い一方、厳密には次の点に注意が必要です。

- EHTが推定する「リング直径」は、複数の画像再構成法と幾何学モデル（リング/ガウス等）を併用した **“リング状構造の代表サイズ”** である
- これは「事象の地平面そのもの」を直接見ているわけではなく、**発光領域と相対論的レンズ効果が作る見かけの構造**である

本プロジェクトでは第一歩として

- **リング直径 ≒ シャドウ直径（κ=1）**

と近似して比較し、次段階として `κ` と GR（Kerr）のスピン依存を系統誤差として扱います。

補足（Sgr A* のリングは“薄い円”ではない）：
- Sgr A* Paper I は、リングの **fractional width（W/d）がおよそ 30–50%** と報告しています。
- また **brightness asymmetry（A）もおよそ 0.04–0.3** とされ、リングの“形”は系統（再構成・放射モデル等）の影響を受けやすいことが示唆されます。
- M87* Paper I の幾何モデル（asymmetric emission ring/crescent）では、**fractional width（W/d）が <0.5（上限）** と報告されています。

補足（Sgr A* の散乱：一次ソース）：
- Sgr A* Paper I は、thin ring（直径 54 μas）を **Gaussian kernel（FWHM 23 μas）で畳み込んだ比較**を示し、散乱が無視できない系統要因であることを明示しています。
- 同論文は「descattered imaging」で **散乱カーネルと散乱パワースペクトル**を用いると述べ、その一次ソースとして Psaltis et al. (2018) / Johnson et al. (2018) を参照しています。
- Zhu et al. (2018) は、230 GHz における散乱カーネルのスケールとして **FWHM（長軸）≈23 μas、（短軸）≈12 μas** を示しています。
- Johnson et al. (2018) は、散乱カーネルを異方性ガウスとしてパラメータ化し、軸サイズが λ^2 でスケールすることを用いて 230 GHz（λ=1.3 mm）換算で **約 23.3±0.2 μas × 11.9±0.2 μas（≈1%）** 程度になることが読み取れます。
- Zhu et al. (2018) はさらに、屈折散乱（refractive）により「像のゆらぎ」が生じ、230 GHz でも無視できない場合があることを数値で示します（Kolmogorov 想定では平均 distortion ≈0.72 μas（≈1.4%）、別モデルでは最大 ≈6.3 μas（≈12%））。
- 本リポジトリでは上記の“スケール感”を `output/eht/eht_refractive_scattering_limits.png` にまとめ、κ（リング/シャドウ変換）の誤差予算を詰める際の下敷きとして扱います。
- 重要な点：平均散乱カーネルの“係数の不確かさ”自体は 1%級でも、散乱サブ構造（refractive）や放射モデルの違いが κ（リング/影変換）を支配し得るため、κ を 1% で固定できるとは限りません（この整理が Step 8.2 の焦点）。

---

## P-model 側の仮定（最小モデル）

`doc/P_model_handoff.md` の定義に従い、

- 点質量解（弱場Poisson解の延長）：
  - `P/P0 = exp(GM/(c^2 r))`
- 光の屈折率（Cassini拘束で β≈1）：
  - `n(r) = (P/P0)^(2β) = exp(2β GM/(c^2 r)) = exp(β r_s / r)`（`r_s = 2GM/c^2`）

球対称屈折率 `n(r)` 中の光線では、角運動量保存に相当する関係として

- `b = n(r) r`

（インパクトパラメータ `b` が保存）を用います。  
このとき `b(r)=n(r)r` の極小が「捕獲境界」（影の縁）に対応し、最小モデルでは

- `b(r) = r exp(β r_s / r)` → `db/dr = 0` から `r = β r_s`
- `b_crit = b(r=β r_s) = e β r_s`

を得ます。従って、距離 `D` に対する角シャドウ直径は

- **P-model の影（シャドウ）直径（最小モデル）**  
  `θ_shadow = 2 b_crit / D = 4 e β (GM/(c^2 D))`

補足（“最小モデル”の意味）：
- ここでの P-model は「点質量・静的・球対称・屈折率 n(P) のみ」の最小仮定です。
- 自転（スピン）・プラズマ・放射モデル（どこが光っているか）は含みません。
- 従って EHT との比較は「まず係数スケールが合うか」の確認であり、精密比較は Phase 4/5 で扱います。

---

## 参考：標準理論（GR）の係数（Schwarzschild と Kerr）

Schwarzschild近似（スピンなし）では、シャドウ直径係数は

- GR（Schwarzschild）：`θ_shadow = 2√27 (GM/(c^2 D))`（係数 `2√27 ≈ 10.392`）

一方、Kerr（スピンあり）ではシャドウ形状は完全な円ではなく、観測で“直径”をどう定義するかで係数が揺れます。  
本プロジェクトでは参考として Kerr シャドウ境界を数値生成し、`(幅+高さ)/2` を“等価直径”とみなした係数レンジを `eht_shadow_systematics.png` に併記します（観測モデルではなく、系統誤差レンジの目安）。

### Kerr 参考レンジの「制約版」

Kerr の係数レンジは、スピン `a*` と視線角（inclination）`inc` の取り方で幅が変わります。  
本プロジェクトではデフォルトとして「広い参考レンジ」（`a*∈[0,0.999]`, `inc∈[5°,90°]`）を採用しつつ、一次ソースがある場合に限り **天体ごとの制約レンジ**で再計算します。

入力：`data/eht/eht_black_holes.json`
- `kerr_a_star_min/max`（任意）
- `kerr_inc_deg_min/max`（任意）

注記：
- M87* の `inc` は「ジェットの視線角」を **スピン軸の近似**として使います（厳密に同一とは限らないため、幅は保守的に取ります）。
- Sgr A* は放射モデル依存の不確かさが大きい一方、EHT Sgr A* Paper I（arXiv:2311.08680）のモデル比較では **高傾斜（i>50°）が不利（disfavor）** とされます。そこで本プロジェクトでは **系統誤差レンジ評価に限り**、Kerr参考レンジの上限として `inc≤50°`（`kerr_inc_deg_max=50`）を適用します（「確定的拘束」ではなく、Kerrレンジを現実的に詰めるための補助制約）。

---

## Phase 4（差分予測）：P-model と GR の「数%差」

最小モデルでは、βをCassini等で固定した後でも、シャドウ直径係数が

- P-model：`4eβ`（β=1で約10.873）
- GR：`2√27`（約10.392）

となり、**数%の差**が残ります。

この差を観測で判別するには、少なくとも次の不確かさを同時に詰める必要があります。

- 観測誤差（リング直径推定）
- 質量・距離の不確かさ（`GM/(c^2 D)`）
- `κ`（リング直径 / シャドウ直径）の系統誤差
- GR側（Kerr）のスピン/傾き依存

出力：
- `output/eht/eht_shadow_differential.png`（差と、3σで判別するための必要精度の目安）

### 3σ判別に必要な精度（目安；このリポジトリの定義）

このリポジトリでは、モデルA/Bの予測がそれぞれ

- A：N(μ_A, σ_pred,A² + σ_obs²)
- B：N(μ_B, σ_pred,B² + σ_obs²)

に従うとみなしたとき、分布の重なりが十分小さくなる目安として

- |μ_A − μ_B| > 3 × √(σ_pred,A² + σ_pred,B² + 2σ_obs²)

を採用し、必要な観測誤差（1σ）を逆算して `eht_shadow_differential.png` に表示します。  
（σ_pred は主に 質量・距離（GM/(c²D)）の不確かさ由来。）

この定義での所見（2026-01-21 時点；詳細は `output/eht/eht_shadow_compare.json`）：
- **M87***：質量・距離の相対誤差が大きく（θ_unit_rel_sigma≈11.8%）、この段階では **σ_obs→0 でも 3σ判別が不可能（n/a）**。  
  - 目安として、3σ判別が「可能」になるには θ_unit_rel_sigma≲1.1% 程度まで詰める必要があります（質量・距離の改善が必須）。
- **Sgr A***：質量・距離の相対誤差が小さい（θ_unit_rel_sigma≈0.46%）ため、差分予測を観測で判別できる余地があります。  
  - ただし、3σ判別のためには **影直径の観測誤差（1σ）を概ね 0.5 µas 程度以下**にする必要があり、現状のリング直径推定や散乱/放射モデル（κ）の系統を詰める必要があります。

---

## 重要：EHT の “リング直径” と “シャドウ直径” は同一ではない（κ）

EHT が推定する「リング直径」は、放射モデル・散乱（特に Sgr A*）・スピン・視線角などに依存します。  
そのため厳密には

- `リング直径 = κ × シャドウ直径`

という **変換係数 κ（系統誤差）** が必要になります（κ=1 が成り立つとは限らない）。

本リポジトリの扱い：
- **κを“確定した観測量（モデル非依存）”とは扱いません**（放射モデル/散乱/再構成の系統誤差を代表する未確定パラメータ）。
- ただし、値の意味を分けて次の2種類を出力します（どちらも「系統」を含む可能性がある点に注意）。
  - **κ_fit**：観測リング直径と、理論（P-model/GR）が予測するシャドウ直径から逆算した κ（`kappa_ring_over_shadow_fit_*`）
  - **κ_obs（参考）**：観測リング直径と、論文中で“シャドウ直径”が明示されている場合にのみ、その値から計算した κ（`kappa_ring_over_shadow_obs`）  
    - 例：Sgr A* Paper I ではシャドウ直径が「リング直径からの推定量」として示されますが、これは放射モデル等を含む導出量であり、**κを独立に測ったものではありません**。

出力JSON：`output/eht/eht_shadow_compare.json`

出力（κ・Kerrレンジの可視化）：
- `output/eht/eht_shadow_systematics.png`
- `output/eht/eht_kappa_fit.png`

---

## 入力データ（固定値）

オフライン再現のため、使用する観測値・質量・距離は `data/eht/eht_black_holes.json` に固定します。  
このJSONは「一次データの代替キャッシュ」であり、値の出典URLも同梱します。

---

## 一次ソースの保存（このリポジトリ内）

EHT の一次ソース（PDF/TeX）は、再現性のために `data/eht/sources/` に保存します。

- `data/eht/sources/arxiv_1906.11238.pdf`（EHT M87* Paper I, PDF）
- `data/eht/sources/arxiv_1906.11243.pdf`（EHT M87* Paper VI, PDF）
- `data/eht/sources/aanda_aa47932-23.pdf`（EHT M87* 2018 Paper I, A&A 2024, PDF）
- `data/eht/sources/aanda_aa51296-24.pdf`（EHT M87* 2018 Paper II, A&A 2025, PDF）
- `data/eht/sources/arxiv_2206.12066.pdf`（Vincent+2022, PDF；リング≒影の系統）
- `data/eht/sources/arxiv_2206.12066_src.tar.gz`（Vincent+2022, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2206.12066/`
- `data/eht/sources/arxiv_1907.04329.pdf`（Johnson+2020, PDF；photon ring干渉計シグナチャー）
- `data/eht/sources/arxiv_1907.04329_src.tar.gz`（Johnson+2020, arXiv source）
  - 展開先：`data/eht/sources/arxiv_1907.04329/`
- `data/eht/sources/arxiv_2008.03879.pdf`（Gralla+2020, PDF；photon ring形状テスト）
- `data/eht/sources/arxiv_2008.03879_src.tar.gz`（Gralla+2020, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2008.03879/`
- `data/eht/sources/arxiv_2311.08679_src.tar.gz`（EHT Sgr A* Paper II, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2311.08679/`
- `data/eht/sources/arxiv_2311.08679.pdf`（EHT Sgr A* Paper II, PDF）
- `data/eht/sources/arxiv_2311.08680_src.tar.gz`（EHT Sgr A* Paper I, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2311.08680/`
- `data/eht/sources/arxiv_2311.08680.pdf`（EHT Sgr A* Paper I, PDF）
- `data/eht/sources/arxiv_2311.09479_src.tar.gz`（EHT Sgr A* Paper III, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2311.09479/`
- `data/eht/sources/arxiv_2311.09479.pdf`（EHT Sgr A* Paper III, PDF）
- `data/eht/sources/arxiv_2311.08697_src.tar.gz`（EHT Sgr A* Paper IV, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2311.08697/`
- `data/eht/sources/arxiv_2311.08697.pdf`（EHT Sgr A* Paper IV, PDF）
- `data/eht/sources/arxiv_2311.09478_src.tar.gz`（EHT Sgr A* Paper V, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2311.09478/`
- `data/eht/sources/arxiv_2311.09478.pdf`（EHT Sgr A* Paper V, PDF）
- `data/eht/sources/arxiv_2311.09484_src.tar.gz`（EHT Sgr A* Paper VI, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2311.09484/`
- `data/eht/sources/arxiv_2311.09484.pdf`（EHT Sgr A* Paper VI, PDF）
- `data/eht/sources/arxiv_1807.09409_src.tar.gz`（GRAVITY Collaboration 2018: S2 重力赤方偏移, arXiv source）
  - 展開先：`data/eht/sources/arxiv_1807.09409/`
- `data/eht/sources/arxiv_1807.09409.pdf`（GRAVITY Collaboration 2018: S2 重力赤方偏移, PDF）
- `data/eht/sources/arxiv_2004.07187_src.tar.gz`（GRAVITY Collaboration 2020: S2 Schwarzschild 歳差, arXiv source）
  - 展開先：`data/eht/sources/arxiv_2004.07187/`
- `data/eht/sources/arxiv_2004.07187.pdf`（GRAVITY Collaboration 2020: S2 Schwarzschild 歳差, PDF）
- `data/eht/sources/arxiv_1904.05721.pdf`（GRAVITY Collaboration 2019, PDF）
- `data/eht/sources/arxiv_1608.05063.pdf`（Mertens et al. 2016, PDF）
- `data/eht/sources/arxiv_1805.01242.pdf`（Psaltis et al. 2018, 散乱モデル, PDF）
- `data/eht/sources/arxiv_1808.08966.pdf`（Johnson et al. 2018, 散乱と固有構造, PDF）
- `data/eht/sources/arxiv_1808.08966_src.tar.gz`（Johnson et al. 2018, arXiv source）
  - 展開先：`data/eht/sources/arxiv_1808.08966/`
- `data/eht/sources/arxiv_1811.02079.pdf`（Zhu et al. 2018, 散乱による制限, PDF）
- `data/eht/sources/arxiv_1811.02079_src.tar.gz`（Zhu et al. 2018, arXiv source）
  - 展開先：`data/eht/sources/arxiv_1811.02079/`

注記：
- M87*（2019）は arXiv 側で TeX source が提供されないため、**PDFのみ**が一次ソースになります（本リポジトリでは値の固定入力は `data/eht/eht_black_holes.json` に集約）。
- Sgr A*（2022）は arXiv source が入手でき、本文（TeX）内に「シャドウ直径」の記載があるため、`shadow_diameter_uas` として固定入力に含めています（ただし上記の通りモデル依存の導出量として扱う）。
  - 追加：散乱一次ソース（Johnson+2018 / Zhu+2018）も arXiv source を固定し、TeX内アンカーで値の転記ミス検出を強化。

### 出典（一次ソース）

- M87* リング直径 42±3 µas、距離 16.8±0.8 Mpc、質量 6.5±0.7×10^9 M⊙：EHT M87 Paper I（arXiv:1906.11238）
  - 幾何モデルの fractional width（W/d）: <0.5（上限）
- M87* の視線角（ジェット角）17.2±3.3°（参考）：Mertens et al.（A&A 595, A54, 2016）  
  - 本プロジェクトでは制約レンジ例として `inc∈[10°,25°]` を採用（中心値±3.3°を含み、さらに余裕を持たせた保守的範囲）
- Sgr A* リング直径 51.8±2.3 µas：EHT Sgr A* Paper I（arXiv:2311.08680）
  - 同論文の ring fractional width（W/d）：およそ 30–50%
  - 同論文の brightness asymmetry（A）：およそ 0.04–0.3
  - 同論文の比較図例（thin ring + 散乱カーネル）：Gaussian kernel FWHM 23 µas（可視化の比較として提示）
  - 同論文のモデル比較：高傾斜（i>50°）が不利（disfavor）→ Kerr参考レンジ制約（`kerr_inc_deg_max=50`）に利用（系統誤差レンジのみ）
- Sgr A* の質量・距離：GRAVITY Collaboration（A&A 625, L10, 2019）
- Sgr A* の散乱（一次ソース）：Psaltis et al. (2018, arXiv:1805.01242), Johnson et al. (2018, arXiv:1808.08966)
- 散乱による制限（参考）：Zhu et al. (2018, arXiv:1811.02079)

---

## 再現方法

- 一次ソース取得（online）：`python -B scripts/eht/fetch_eht_sgra_series.py`（取得済みなら不要）
- 一次ソース取得（online; 補強論文）：`python -B scripts/eht/fetch_eht_supporting_papers.py`（取得済みなら不要）
- 一次ソース取得（online; S2）：`python -B scripts/eht/fetch_gravity_s2_papers.py`（取得済みなら不要）
- 実行：`python -B scripts/eht/eht_shadow_compare.py`
- 一次ソース整合（sha256）：`python -B scripts/eht/eht_sources_integrity.py`
- 一次ソース値アンカー（TeX内ヒット）：`python -B scripts/eht/eht_primary_value_anchors.py`
- 出力：`output/eht/`（CSV/JSON/PNG）
- 公開レポート反映：`python -B scripts/summary/public_dashboard.py`（`output/summary/pmodel_public_report.html` に追加）
- 論文HTML更新：`python -B scripts/summary/paper_html.py`（`output/summary/pmodel_paper.html`）
