# Pモデル（時間波密度P）まとめ — 引き継ぎ用（1ファイル）

このファイルは、これまでの議論・検証結果を **次のチャットへ引き継ぐための要約**です。  
（目的：必要な前提・式・検証手順・成果物の場所・結論が一目で分かる）

---

## 0. ゴール（何を検証したか）
1) **地球周回（GNSS/GPS）衛星クロック**で、Pモデルが現行（放送暦ベース）と同等精度で成立するか。  
2) **惑星間（Cassini太陽会合）**で、光伝播（Shapiro遅延・Doppler波形）が論文レベルと矛盾しないか。  
3) 最後に **論文図（Fig.2）をデジタイズ**して、Pモデル波形と重ね合わせ。

---

## 1. 理論の骨格（確定版）

### 1.1 時間波密度 P（空間スカラー）
- 空間には「時間波」が満ちており、その波面の詰まり具合（時間の刻みの細かさ）を **時間密度 P(x)** とする。
- 質量が存在すると P が歪み、その歪みは中心から放射状に広がり、距離とともに滑らかに減衰する。

### 1.2 Pとポテンシャル（ニュートン形に統一）
- 解釈：**質量近傍でPが増える**（\(P>P_0\)）。
- ニュートン形ポテンシャル（無限遠で0、質量近傍で負）を
  \[
  \phi \equiv -c^2\ln\Big(\frac{P}{P_0}\Big)
  \]
  と定義する（\(P>P_0\Rightarrow \phi<0\)）。
- 弱場での源方程式（Poisson型）：
  \[
  \nabla^2\phi = 4\pi G\rho
  \]
- 点質量 M のとき：
  \[
  \phi(r)=-\frac{GM}{r},\quad \frac{P(r)}{P_0}=\exp\Big(\frac{GM}{c^2r}\Big)
  \]

### 1.3 重力（運動方程式）
- 加速度：
  \[
  \mathbf a = -\nabla\phi = c^2\nabla\ln(P/P_0)
  \]
- これにより円軌道など古典結果（v=√(GM/r), v_esc=√(2GM/r)）が導出可能。

### 1.4 時計（固有時）
- 衛星クロック比較用の確定モデル：
  \[
  \frac{d\tau}{dt}
  =
  \exp\Big(-\frac{\mu}{c^2r}\Big)\;
  \sqrt{1-\frac{v^2}{c^2}}
  \quad(\mu=GM)
  \]
- 速度飽和 `δ_0` は Part I（コア）には含めず、強場/高γ領域における拡張仮説として Part II で上限拘束と棄却条件の形を整理する（採用する場合は Part I 改訂に集約する）。
- 周波数比からオフセットへ：
  \[
  \Delta t_P(t)=\int_{t_0}^{t}\Big(\frac{d\tau}{dt'}-1\Big)dt'
  \]

### 1.5 光の伝播（Cassiniにより拘束）
- 光への効き方を \(\beta\) として
  \[
  c_{eff}(r)=c\Big(\frac{P_0}{P(r)}\Big)^{2\beta},\quad
  n(r)=\Big(\frac{P(r)}{P_0}\Big)^{2\beta}
  \]
- CassiniのPPN拘束（gamma ~ 1 +/- 2.3e-5）から **β ~ (1+gamma)/2 ~ 1** に強く固定される。
- 重要：Shapiro遅延や光の偏向で現れる **係数2（GRの(1+γ)=2）は、\(\phi\) の定義ではなく \(\beta\)（光伝播側）が担う**。  
  したがって「\(\phi=-c^2\ln(P/P_0)\) を固定し、\(\beta\) を微調整する」方針で運用する（\(\phi\) に `sqrt()`（1/2乗）を混ぜて係数を吸収しない）。

### 1.6 回転（フレームドラッグ：最小拡張）
- 相対論の置換候補としては、回転質量が作る効果（フレームドラッグ / Lense-Thirring）を説明できる必要がある。
- 現段階では「弱場で既知の結果（Lense-Thirring極限）と矛盾しない」ことを必須要件として固定し、観測値との整合性チェックを追加した（GR/SRとは概念は別だが、同じ現実を説明する限り弱場で一致する部分が現れる）。
- 比較は符号依存を避けるため、比 \(\mu \equiv |\Omega_{obs}|/|\Omega_{pred}|\)（GRの予測は \(\mu=1\)）で行う。
- 入力固定：`data/theory/frame_dragging_experiments.json`
- 再現：`python -B scripts/theory/frame_dragging_experiments.py`
- 出力：`output/theory/frame_dragging_experiments.png`（GP-B / LAGEOS の一次ソース値）

### 1.7 動的P（放射・波動：最小仮定）
- 連星系の軌道減衰や重力波chirpは「放射（エネルギーが系外へ出る）」現象を含むため、Pを時空関数 \(P(x,t)\) として扱う必要がある。
- 最小の作業仮定として、変数 \(u\equiv \ln(P/P_0)\) を基本変数に取り、静的極限でPoisson方程式を回収する形で
  \[
  \Box u \equiv \left(-\frac{1}{c^2}\frac{\partial^2}{\partial t^2}+\nabla^2\right)u
  = -\frac{4\pi G}{c^2}\rho
  \]
  を採用する（真空では \(\Box u=0\) の波動方程式）。
- 遠方場 \(r\gg R\)（系のサイズ \(R\) に比べて十分遠い領域）では、遅延解の多極展開により放射の主項が決まる。質量モーメントを
  \[
  M=\int d^3x\,\rho,\quad
  D_i=\int d^3x\,\rho x_i,\quad
  Q_{ij}=\int d^3x\,\rho\left(x_i x_j-\frac{1}{3}\delta_{ij}x_k x_k\right)
  \]
  とすると、質量保存より \(dM/dt=0\)（単極は放射しない）、運動量保存より \(d^2D_i/dt^2=0\)（重心系では \(D_i\) は定数で双極放射しない）となる。
  したがって弱場・遠方では **四重極放射が主項**になる。
- 重要：もし天体の種類によって \(P\) への結合（“荷”）が変わると、NS–WD 連星などで双極放射が顕在化し、観測で強く棄却されやすい。本リポジトリでは「普遍結合（双極なし）」を弱場極限の必要条件として固定する。
- 観測量としては
  - 二重パルサー：軌道周期の減衰 \(\dot{P}_b\)
  - 重力波：周波数上昇 \(df/dt\) と位相（chirp）
  を固定し、P-modelはまず弱場極限で標準の四重極則と矛盾しないことを必要条件として扱う。
- 参考（スケール感）：`python -B scripts/theory/dynamic_p_quadrupole_scalings.py`
  - 出力：`output/theory/dynamic_p_quadrupole_scalings.png`

---

## 2. 地球周回（衛星クロック）検証まとめ

### 2.1 現行（放送暦）式（BRDC）
- 衛星時計補正（比較用に最小構成）：
  \[
  \Delta t_{BRDC}(t)
  =a_{f0}+a_{f1}(t-t_{oc})+a_{f2}(t-t_{oc})^2 + \Delta t_{rel}(t)
  \]
- 相対論補正（楕円起因）：
  \[
  \Delta t_{rel}(t)=F\;e\;\sqrt{A}\;\sin E(t),\quad
  F=-\frac{2\sqrt{\mu}}{c^2}
  \]

### 2.2 比較の基準
- 「準実測」：IGS Final Clock（CLK）
- 軌道：IGS Final Orbit（SP3）

### 2.3 比較方法（重要）
- 各衛星で **IGS − model** に対して **バイアス＋ドリフト（一次）**を最小二乗で除去：
  \[
  r(t)=C_{IGS}(t)-\{C_{model}(t)+a+bt\}
  \]
- 残差の RMS / Max を比較。

### 2.4 結果（2025-10-01, GPS Week 2386）
- 単一衛星（G01）精密版：
  - BRDC RMS ≈ 1.342 ns
  - P-model RMS ≈ 1.367 ns
- 全衛星まとめ（31衛星）：
  - P-modelが良い：15衛星
  - BRDCが良い：16衛星
  - 平均差（P−BRDC）：+1.70 ps（ほぼ同等）
  - 中央値：+0.09 ps

→ **衛星クロック検証の範囲では、P-modelは現行式と同程度の精度レンジ（ns級）で成立**。

### 2.5 使用ファイル（ログイン不要）
BKG root_ftp（ログイン不要）から取得し、展開して使用：
- data/gps/BRDC00IGS_R_20252740000_01D_MN.rnx
- data/gps/IGS0OPSFIN_20252740000_01D_05M_CLK.CLK
- data/gps/IGS0OPSFIN_20252740000_01D_15M_ORB.SP3

結果CSV：
- output/gps/residual_G01.csv（15分）
- output/gps/residual_precise_G01.csv（5分）
- output/gps/summary_batch.csv（G01..G32の一括RMS）

---

## 3. 惑星間（Cassini）検証まとめ

### 3.1 Cassini論文式（往復）
- 往復Shapiro遅延（b近似形）：
  \[
  \Delta t = 2(1+\gamma)\frac{GM_\odot}{c^3}\ln\Big(\frac{4r_1r_2}{b^2}\Big)
  \]
- 往復Doppler：
  \[
  y=-\frac{d(\Delta t)}{dt}\approx 4(1+\gamma)\frac{GM_\odot}{c^3}\frac{1}{b}\frac{db}{dt}
  \]
- P-modelは Cassini拘束により β≃1（＝γ≃1）で固定して検証。

### 3.2 幾何データ取得
- JPL HORIZONS API を使用（社内プロキシ環境のため Python側でSSL検証OFF）
- Earth=399, Cassini=-82, center=500@10（太陽中心）
- 期間：2002-06-06〜2002-07-07
- 1分刻み

### 3.3 主要結果（解析版）
- b_min / R_sun = 1.6667（2002-06-21 13:07 UTC）
- |y|_peak ≈ 4.25×10^-10
- Δt（往復）= 126.1〜258.6 μs

→ 会合幾何・波形（Δtの山形、yのS字＋ゼロ交差）が理論式の期待と整合。

### 3.4 “実測”との重ね合わせ（論文Fig.2デジタイズ）
- 論文図（Fig.2）を画像化し、曲線を近似デジタイズして点列化。
- 点列と P-model y(t) を会合中心（b_min）で時間軸整列して重ね合わせグラフを生成。

生成画像：
- output/cassini/cassini_fig2_overlay_full.png
- output/cassini/cassini_fig2_overlay_zoom10d.png
- output/cassini/cassini_fig2_residuals.png
- output/cassini/cassini_fig2_overlay_bestbeta_zoom10d.png
- output/cassini/cassini_pds_vs_digitized.png（整合チェック: PDS処理後 vs 論文図デジタイズ）
- scripts/cassini/cassini_fig2_overlay.py (repro: overlay + beta sweep, offline from cached PDS)
- data/cassini/pds_sce1/（入力: Cassini SCE1 PDSキャッシュ）
- data/cassini/cassini_fig2_digitized_raw.csv（参考: 論文Fig.2デジタイズ点列）

---

## 4. 生成物（このプロジェクトで作ったファイル一覧）
### 4.1 プレゼン
- 元（概念デッキ）：doc/slides/時間波密度Pで捉える「重力・時間・光」.pptx
- Cassini（Fig.2重ね合わせ）：output/cassini/時間波密度Pで捉える「重力・時間・光」_Cassini_Fig2重ね合わせ.pptx
- Cassini（β感度）：output/cassini/時間波密度Pで捉える「重力・時間・光」_Cassini_Fig2重ね合わせ_beta感度.pptx

### 4.2 Cassini関連CSV
- output/cassini/cassini_shapiro_y_full.csv（Eq(2) + FULL）
- output/cassini/cassini_sce1_tdf_extracted.csv（PDS一次データ抽出: y/復元情報つき）
- output/cassini/cassini_sce1_tdf_paperlike.csv（平滑化＋デトレンド後の y(t)）
- output/cassini/cassini_pds_vs_digitized_metrics.csv（整合チェック指標）
- output/cassini/cassini_fig2_matched_points.csv（観測点列 + y_model + residual）
- output/cassini/cassini_fig2_metrics.csv（観測（PDS処理後） vs P-model 指標）
- output/cassini/cassini_beta_sweep_rmse.csv（β感度スイープRMSE）

### 4.3 Cassini関連画像
- output/cassini/cassini_beta_sweep_rmse.png（β感度の可視化）
- output/cassini/cassini_fig2_overlay_full.png（PDS一次データ（処理後） vs P-model重ね合わせ：全体）
- output/cassini/cassini_fig2_overlay_zoom10d.png（同：±10日ズーム）
- output/cassini/cassini_fig2_residuals.png（残差）

---

## 5. 現時点の結論（短く）
- **衛星クロック（GNSS）**：P-modelは現行式と同程度の精度レンジ（ns級）で成立。  
- **Cassini会合**：PDS一次データ（TDF）から y(t) を復元し、Shapiro由来の波形（S字・ゼロ交差・オーダー10^-10）をP-modelで再現できるかを検証中。補助として論文図デジタイズとも整合チェックを行う。  
- Cassini拘束により、光伝播側の自由度 β は **β≃1（10^-5レベルで固定）**。

---

## 6. 次のチャットでやると良い次ステップ（候補）
1) Fig.2デジタイズ精度を上げる（手動アンカー点追加、軸目盛りの厳密キャリブレーション）。  
2) βを 1±1e-5 範囲で振って、重ね合わせ誤差がCassini拘束内で不変であることを数値確認。  
3) さらに進めるなら、生データ（PDS等）での残差フィット（コロナ補正・DSN局・軌道決定モデルを含む）。

---
