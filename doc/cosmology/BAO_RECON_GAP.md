# BAO一次統計（catalog-based）: recon と published のギャップ整理（Phase 4 / Step 4.5B）

目的：
- BOSS DR12 の BAO一次統計（銀河+random → ξ(s,μ) → ξ0/ξ2）を、P-model側で再導出する入口を「再現可能」にする。
- その前提として、**pre-recon は Satpathy 2016 と概ね整合**する一方、**簡易recon（grid/FFT）では Ross 2016 post-recon の ξ2 が十分一致しない**というギャップを、議論がブレない形で固定する。
  - 追加：ε（理論差）と「推定量差」が混ざらないよう、推定量定義（正規化/μ積分/誤差モデル）を `doc/cosmology/BAO_RECON_SPEC.md` §7 に固定し、`coordinate_spec_hash` / `estimator_spec_hash` で混在検出できるようにした。

## 現状（要点）

- pre-recon（catalog-based ξℓ）→ Satpathy 2016 pre-recon multipoles：概ね一致（RMSEは小）
- post-recon（簡易recon grid）→ Ross 2016 post-recon multipoles：**特に ξ2 が不一致（RMSEが大）**

要約図：
- `output/cosmology/cosmology_bao_recon_gap_summary.png`
  - 既存の overlay metrics を集約し、recon 設定の違いで ξ2 ギャップがどの程度動くかを zbin1..3 で可視化。

## 追加検証（2026-01-15）：z-bin 重複 → recon入力差の切り分け

背景：
- 本リポジトリの z-bin は重複している（b1:[0.2,0.5), b2:[0.4,0.6), b3:[0.5,0.75)）。
- もし recon を z-bin ごとに別々に走らせると、**recon入力の銀河分布が z-bin ごとに変わる**ため、同じ「recon_grid_iso」でも結果が揺れる可能性がある。

対応：
- `scripts/cosmology/cosmology_bao_xi_from_catalogs.py` に `--recon-z-min/max` を追加し、**recon入力の z 範囲**を測定 z-bin（z_min/z_max）から分離できるようにした。

実施：
- CMASSLOWZTOT / combined / lcdm / random1 / recon_grid_iso について、recon入力 z を `[0.2,0.75)` に固定して b1/b2/b3 を再計算。
  - out_tag：`__recon_grid_iso_rz_zmin0p2_zmax0p75`
  - overlay：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__recon_grid_iso_rz_zmin0p2_zmax0p75.*`

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：
- baseline（`__recon_grid_iso`）：zbin1=74.09, zbin2=72.44, zbin3=77.75
- recon入力固定（`__recon_grid_iso_rz_zmin0p2_zmax0p75`）：zbin1=74.18, zbin2=73.85, zbin3=78.03

結論：
- **recon入力 z 範囲（z-bin 重複の影響）は、Ross post-recon の ξ2 ギャップの主因ではない**（改善せず）。

## 追加検証（2026-01-15）：recon推定量の正規化（RR0分母 vs SS分母）

背景：
- 本リポジトリの簡易reconは、recon後の密度場差分 δ_rec = δ_D − δ_S の自己相関として ξ を作る。
- 現状の実装では、保存している pair-count grid（DD/DS/RR0/SS）から、概ね次の形で ξ を構成している（正規化を含む）：
  - stored（現行）：ξ = (DDn − 2DSn + SSn) / RR0n（RR0：unshifted random）
- 一方で published pipeline は、窓関数や正規化の定義が微妙に異なる可能性があるため、**分母（正規化）の違いが ξ2 ギャップの主因か**を診断する。

対応：
- `scripts/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay.py` に `--catalog-recon-estimator` を追加し、recon NPZに保存された pair-count grid から ξ0/ξ2 を再計算できるようにした。
  - `stored`：NPZに保存済みの ξ0/ξ2（現行）
  - `delta_ss`：ξ = (DDn − 2DSn + SSn) / SSn（SS：shifted random の自己相関；診断用）

実施：
- CMASSLOWZTOT / combined / lcdm / zbin1..3 / recon_grid_iso
  - overlay：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__recon_grid_iso__delta_ss.*`

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：
- baseline（`__recon_grid_iso`）：zbin1=74.09, zbin2=72.44, zbin3=77.75
- `delta_ss`（`__recon_grid_iso__delta_ss`）：zbin1=78.13, zbin2=77.35, zbin3=82.29

結論：
- **分母（RR0 → SS）の差し替えでは ξ2 ギャップは改善しない**（むしろ悪化）。  
  → 主因は「正規化（分母）の差」ではなく、引き続き **公式recon仕様差/選択関数/窓関数** 側の切り分けが必要。

## 追加検証（2026-01-15）：random の prefix_rows → reservoir（scan=5,000,000）での影響

背景：
- prefix_rows（先頭N行）抽出は、ファイル行順に依存した偏り（セクタ偏り等）が入り得る。
- ξ2 の “recon gap” が **理論差ではなく、選択関数/窓関数（randomの偏り）** に起因する可能性を切り分けるため、
  random を reservoir 抽出へ切替して再計算する。

対応：
- `data/cosmology/boss_dr12v5_lss/manifest.json` の CMASSLOWZTOT（north/south）の random1 を、
  prefix_2,000,000 → reservoir_2,000,000 へ更新（ただし計算量の都合で scan_max_rows=5,000,000）。
  - 抽出ファイル例：
    - `data/cosmology/boss_dr12v5_lss/extracted/random1_DR12v5_CMASSLOWZTOT_North.fits.gz.reservoir_2000000_seed0_scan5000000.npz`
    - `data/cosmology/boss_dr12v5_lss/extracted/random1_DR12v5_CMASSLOWZTOT_South.fits.gz.reservoir_2000000_seed0_scan5000000.npz`
- WSL（Corrfunc）で ξℓ（pre/recon_grid_iso）を b1/b2/b3 で再計算し、overlay と recon gap 要約図を更新。

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：
- 旧baseline（prefix_2,000,000；`__recon_grid_iso`）：zbin1=74.09, zbin2=72.44, zbin3=77.75
- 更新後（reservoir_2,000,000; scan=5,000,000；`__recon_grid_iso`）：zbin1=80.42, zbin2=71.71, zbin3=74.10

結論：
- reservoir化（scan=5M）で **zbin3 は改善**したが、**zbin1 は悪化**し、全体として “recon gap” の主因とは言いにくい。
- よって次工程は引き続き、**公式recon仕様差 / window / selection（RR multipoles・窓関数）** の切り分けへ進む。

## 追加検証（2026-01-15）：RSD shift の符号/強さ（`--recon-rsd-shift`）の影響

背景：
- 簡易recon（grid/FFT）で Ross post-recon の ξ2 が合わない要因として、RSD補正項（`Ψ_s = Ψ + (factor·f)(Ψ·LOS)LOS`）の扱いが疑われる。
- そこで、まず **RSD shift を無効化**（`factor=0`）したときの一致度を確認し、さらに **符号反転**（`factor=-1`）を診断する。

実施：
- CMASSLOWZTOT / combined / lcdm / random1 / recon_grid_ani について `--recon-rsd-shift 0` を b1/b2/b3 で再計算
  - out_tag：`__recon_grid_ani_rsdshift0`
  - overlay：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__recon_grid_ani_rsdshift0.*`
- 追加診断（zbin1のみ）：`--recon-rsd-shift -1`（他は baseline の global_constant + global_mean_direction のまま）
  - out_tag：`__recon_grid_ani_rsdshiftm1`

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：
- `__recon_grid_iso`（reservoir_2,000,000; scan=5,000,000 の baseline）：zbin1=80.42, zbin2=71.71, zbin3=74.10
- `__recon_grid_ani_rsdshift0`：zbin1=74.64, zbin2=72.34, zbin3=74.93（median=74.64, mean=73.97）
- `__recon_grid_ani_rsdshiftm1`（zbin1のみ）：zbin1=88.01（悪化）

結論：
- 現行の簡易recon（grid/FFT）では、**RSD shift を入れるほど（factor=1）むしろ ξ2 の一致が悪化**し、`factor=0` が最良だった。
- よって「RSD補正項の符号/強さ」を小手先で調整しても Ross post-recon の ξ2 ギャップは解消しない。
  - 次工程は引き続き、**公式recon仕様差 / selection / window** の切り分け（あるいはより公式に近い recon 実装の導入）へ進む。

## 何が混ざると危険か（差の“混在”を避ける観点）

ξ2 のギャップを議論するとき、次が混在すると「理論差」か「座標化差」かが分離できなくなる：
- redshift 定義（観測z / z_cosmic / z_CMB）
- LOS 定義（pairwise midpoint など）
- comoving distance の積分（精度・補間）
- recon の仕様（grid/box/mask/smoothing/assignment/RSD など）
- random 抽出（prefix/reservoir/全量）と footprint（IPOLY/ISECT）
- 重み（pair counts と recon density field の違い）

このリポジトリでは、上の仕様は “A0” として固定・記録し、混在は metrics の `coordinate_spec` で検出する（`scripts/cosmology/cosmology_bao_xi_from_catalogs.py`）。

## 追加検証（2026-01-16）：Ψ(k) の Fourier 符号（--recon-psi-k-sign）

背景：
- recon の変換は FFT 上で Ψ(k) を構成するため、Fourier 側の符号規約（`--recon-psi-k-sign`）が取り違えられていると、shift の向きが反転し得る。
- “符号を変えるだけで ξ2 ギャップが改善する”可能性を切り分ける。

実施：
- CMASSLOWZTOT / combined / lcdm / random1 / zbin1..3
  - recon：`--recon grid --recon-mode ani --recon-psi-k-sign 1`
  - out_tag：`__recon_grid_ani_psi_pos`
  - overlay：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__recon_grid_ani_psi_pos.*`

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：  
- zbin1=93.98, zbin2=88.49, zbin3=88.55（悪化）

結論：
- `psi_k_sign=+1` は改善しないため不採用。既定（`psi_k_sign=-1`）を維持する。

## 追加検証（2026-01-16）：窓関数の異方性（RR2/RR0, SS2/SS0）

背景：
- published pipeline（Ross post-recon）の ξℓ は、survey window（sky mask / radial selection）と畳み込まれた推定量である。
- window が異方的だと、ξ0 と ξ2 が混合し、ξ2 の形状や符号に影響し得る。
- そこで catalog-based の pair-count grid（RR0/SS）そのものから、window anisotropy の大きさを簡易に定量化する。

方法：
- `scripts/cosmology/cosmology_bao_catalog_window_multipoles.py` を追加し、既存の catalog-based 出力 NPZ（Corrfunc で生成済み）から
  **RR2/RR0**（および recon 時は **SS2/SS0**）を計算する。
- 指標の定義（μ∈[0,1]；偶関数を仮定）：

$$
\frac{R_2(s)}{R_0(s)} \equiv
\frac{5\int_0^1 R(s,\mu)\,L_2(\mu)\,d\mu}{\int_0^1 R(s,\mu)\,d\mu}
$$

出力（固定）：
- `output/cosmology/cosmology_bao_catalog_window_multipoles.png`
- `output/cosmology/cosmology_bao_catalog_window_multipoles_metrics.json`

結果（CMASSLOWZTOT / combined / lcdm；s∈[30,150] Mpc/h の max |R2/R0|）：
- zbin1：RR0 ≈ 0.058、SS ≈ 0.055–0.056
- zbin2：RR0 ≈ 0.156、SS ≈ 0.146–0.148
- zbin3：RR0 ≈ 0.139、SS ≈ 0.128–0.131

解釈（暫定）：
- window anisotropy（RR2/RR0）が zbin2/3 で ≈0.14–0.16 と無視できない規模で存在する。
- ただし本リポジトリの ξ(s,μ) は μ-binned の RR(s,μ) で正規化した上で ξℓ を積分しているため、単純な「RRが異方的だから ξ2 がズレる」とは限らない。  
  → 次工程では、published multipoles 側の window 畳み込み（あるいは deconvolution）の扱いと、catalog-based での再現条件（mixing）を分離して評価する。

## 追加検証（2026-01-16）：window mixing（ξ0↔ξ2 混合）の規模推定

背景：
- multipoles（ξ0, ξ2）は survey window（sky mask / radial selection）と畳み込まれた推定量であり、window が異方的だと **ξ0 と ξ2 が混合**する。
- そこで、catalog-based の RR(s,μ)（必要なら SS(s,μ)）から window multipoles（w2,w4,w6,w8）を推定し、Legendre積の結合係数から mixing 行列
  - ξ0_obs ≈ m00·ξ0_true + m02·ξ2_true
  - ξ2_obs ≈ m20·ξ0_true + m22·ξ2_true
  を計算し、**ξ0→ξ2 の漏れ（m20）**がどの程度になり得るかを定量化する。

方法：
- `scripts/cosmology/cosmology_bao_catalog_window_mixing.py` を追加し、既存の catalog-based NPZ から
  - RR0 由来の m20(s), m22(s) と、漏れの目安 **s²(m20·ξ0)** を推定する。
  - recon の場合は SS 由来も併記（参考）。

出力（追加）：
- `output/cosmology/cosmology_bao_catalog_window_mixing.png`
- `output/cosmology/cosmology_bao_catalog_window_mixing_metrics.json`

結果（要約；CMASSLOWZTOT/combined/lcdm、eval s∈[30,150] Mpc/h）：
- RR0 由来の max|m20| は、max|RR2/RR0| と同程度（zbin1≈0.058, zbin2≈0.156, zbin3≈0.139）。
- 漏れの規模（max|s²(m20·ξ0)|）は
  - zbin1≈2.1
  - zbin2≈3.9
  - zbin3≈3.2
  程度で、Ross post-recon vs catalog recon の ξ2 ギャップ（RMSE(s²ξ2)≈70–80）より十分小さい。

追加：ξ4→ξ2 の mixing（m24）も同様に評価：
- `output/cosmology/cosmology_bao_xi_from_catalogs_*.npz` に保存している ξ(s,μ)（xi_mu）から ξ4 を再構成し、
  window mixing 係数 **m24(s)**（ξ4→ξ2 の漏れ）と **s²(m24·ξ4)** を計算した。
- 漏れの規模（max|s²(m24·ξ4)|）は概ね
  - zbin1≈0.6
  - zbin2≈1.9
  - zbin3≈1.7
  程度で、依然として ξ2 ギャップ（RMSE≈70–80）を説明するには小さい。

結論（現段階）：
- **window mixing（ξ0↔ξ2 混合）だけで ξ2 ギャップ全体を説明するのは難しい。**
- よって主因は引き続き、recon 仕様差（境界/マスク/RSD除去/重み/割当など）や published pipeline 側の window 畳み込みの扱い差として切り分けるのが合理的。

## 追加メモ（2026-01-16）：published multipoles と window の位置づけ（B.14）

- Ross 2016 の公開データ（`data/cosmology/ross_2016_combineddr12_corrfunc/README`）は、BAO測定の再現に `baofit_pub2D.py`（LSSanalysis）を参照している。
- `baofit_pub2D.py` は **（ξ0, ξ2, ξ4）テンプレートを AP 変換して当てはめるフィットコード**であり、少なくとも同ファイル内には「survey window を明示的に deconvolution する」処理は見当たらない（データベクトルとしての multipoles をそのまま使用し、broadband 多項式で吸収する形式）。
- 公開multipoles（Ross）の「ファイル定義（bincent/covの並び/モデル側のξ4/broadband）」は `doc/cosmology/BAO_RECON_SPEC.md` の **§6** に整理した。
- bin-centering（bin center→shell平均rへの微小補正；`rbc`）は overlay でも試し、RMSEの差は ≲0.1 程度で、ξ2 ギャップ（RMSE≈70–80）の主因になりにくいことを確認した。
  - 出力：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__ross_rbc.png`
  - 指標：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__ross_rbc_metrics.json`
- したがって、現状のギャップは
  - 「published だけが window correction 済みで、catalog-based は未補正」などの単純な不一致よりも、
  - **reconstruction（特に RSD 除去の仕様差）**で ξ2 の符号や形が変わっている可能性が高い、
  という整理が妥当。

## 追加検証（2026-01-16）：Ross 2016 の固定値へ寄せた再計算（B.16）

目的：
- Ross 2016 が明記する recon 固定値（grid=512³ / smoothing=15 / b=1.85 / f=0.757 / z=0.2〜0.75で一括）へ設定を寄せ、
  ξ2 ギャップが「単なるパラメータ不一致」起因か、それとも **アルゴリズム差（radial solver / 境界・マスク・重み）**起因かを切り分ける。

実施：
- WSL（Corrfunc; `--threads 24`）で `scripts/cosmology/cosmology_bao_xi_from_catalogs.py` を実行（CMASSLOWZTOT/combined/lcdm、zbin1..3）。
  - 出力例：`output/cosmology/cosmology_bao_xi_from_catalogs_cmasslowztot_combined_lcdm_b2__recon_grid_ani_ross2016_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757_solverpadmanabh.npz`
- Ross post-recon multipoles との overlay：
  - `output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__ross2016.png`
  - `output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__ross2016_metrics.json`

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：
- Ross 2016 固定値（`__recon_grid_ani_ross2016_*`）：zbin1=81.34, zbin2=75.95, zbin3=77.67
- 参考：baseline（`__recon_grid_iso`）：zbin1=80.42, zbin2=71.71, zbin3=74.10
- 参考：best candidate（`__recon_grid_ani_rsdshift0`）：zbin1=74.64, zbin2=72.34, zbin3=74.93

結論：
- **固定値（f/b/grid/smoothing/z範囲）を Ross 2016 に寄せても ξ2 ギャップは大きくは縮まらない**。  
  → 主因は、Padmanabhan 2012 の **radial LOS + finite-difference solver（PETSc/GMRES）**や、境界・マスク・重みの扱い等の **実装アルゴリズム差**側にある可能性が高い。
- 以降の主戦場は「公式recon仕様差（境界/マスク/重み/RSD除去の定義）」の切り分け（B.17）へ移る。

## 追加検証（2026-01-16）：外部 solver（MW multigrid / radial LOS）導入（B.17）

目的：
- Padmanabhan 2012 が想定する **radial LOS + finite-difference solver** に近い reconstruction を導入し、Ross post-recon の ξ2 ギャップが縮むかを確認する。

実施：
- 外部recon backend（Martin White recon_code; multigrid）を WSL でビルドし、`--recon mw_multigrid` として統合（詳細：`doc/ROADMAP.md` 4.5B.17.1）。
- 実データ（CMASSLOWZTOT/combined/lcdm; zbin1..3）で ξℓ を再計算（recon入力 z は Ross 2016 に合わせて `[0.2,0.75)` に固定）。
  - recon density field weight：`recon_weight_scheme=boss_recon`（FKP を外す）
  - shifted random：既定は **RSD項を入れない**（Padmanabhan 2012）

出力：
- overlay：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid.png`
- 指標：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_metrics.json`

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：
- `mw_multigrid`（外部solver; reservoir scan=5,000,000）：zbin1≈57.60, zbin2≈52.12, zbin3≈64.53（`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_scan5m_metrics.json`）
- `mw_multigrid`（外部solver; reservoir fullscan の現行baseline）：zbin1≈57.70, zbin2≈55.14, zbin3≈71.15（`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_metrics.json`）
- 参考（grid recon; Ross固定値へ寄せた設定）：zbin1≈81.3, zbin2≈76.0, zbin3≈77.7（B.16）

暫定結論：
- **radial LOS + finite-difference solver（外部solver）の導入で ξ2 ギャップは明確に縮む**。  
  → 既存の periodic FFT 近似との差（solver/境界/LOS）が、ξ2 の形状差に実際に寄与している可能性が高い。
- ただし χ²/dof（xi0+xi2）は依然 1 より十分大きく、完全一致ではない。  
  → 次は残差要因（random側のRSD項/重み/selection/NGC-SGC）を最小差分で切り分ける（Roadmap: 4.5B.17.3）。

## 追加検証（2026-01-16）：残差要因の切り分け（B.17.3）

### (1) shifted randoms の RSD項（`--mw-random-rsd`）

目的：
- Padmanabhan 2012 の規約（既定：random側に RSD項を入れない）を崩した場合に、
  Ross post-recon との ξ2 ギャップが縮むかを確認する。

出力：
- overlay：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_mwrndrsd.png`
- 指標：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_mwrndrsd_metrics.json`
- 要約図：`output/cosmology/cosmology_bao_recon_gap_summary.png`

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：
- `mw_multigrid`（既定：random側RSDなし；現行baseline）：zbin1≈57.70, zbin2≈55.14, zbin3≈71.15
- `mw_multigrid + mw-random-rsd`（random側RSDあり；現行baseline）：zbin1≈76.89, zbin2≈73.66, zbin3≈83.26
  - 参考（scan=5,000,000 の旧計算）：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_{scan5m,mwrndrsd_scan5m}_metrics.json`

結論：
- random側に RSD項を入れると **悪化**するため、Padmanabhan 2012 の既定（random側RSDなし）を維持する。
- 実装上の注意：out_tag が80文字制限のため、on/off が上書き衝突しないよう `mwrndrsd` を recon_tag 側へ組み込む。

### (2) NGC/SGC split（north/south）

目的：
- NGC/SGC を混ぜた combined と、分割した north/south の間で、
  どの程度 “観測系（窓/selection/領域差）” の差が出るかを把握する。

出力：
- `output/cosmology/cosmology_bao_xi_ngc_sgc_split_summary.png`
- `output/cosmology/cosmology_bao_xi_ngc_sgc_split_summary_metrics.json`

結果（内部差の目安）：
- Δs_peak（NGC−SGC; s²ξ0 残差ピーク）：b1≈+1.78, b2≈−2.40, b3≈+7.40 [Mpc/h]
- 曲線RMSE（NGC vs SGC; s²ξℓ）：
  - b1：RMSE(s²ξ0)≈9.29, RMSE(s²ξ2)≈93.44
  - b2：RMSE(s²ξ0)≈12.38, RMSE(s²ξ2)≈21.85
  - b3：RMSE(s²ξ0)≈13.29, RMSE(s²ξ2)≈43.91

注記：
- これは Ross post-recon との一致度ではなく、**NGC/SGC の内部差**を示す。
- 特に ξ2 は領域差/サンプル密度差でノイズが増えやすいので、単一の特徴点だけで断定せず、
  曲線RMSEと併せて “系統の切り分け” に用いる。

### (3) 重み/selection（B.17.3.3）

目的：
- 4.5B.17.2 で縮んだ ξ2 ギャップが、残差として「理論差」ではなく **座標化/選択関数（random/sector/重み）**起因で動いていないかを、最小差分で固定する。

実施（要点）：
- random1 を **fullscan reservoir** で抽出して、sector の欠落を抑える（CMASSLOWZTOT; rows_saved=2,000,000）。
  - 抽出：`data/cosmology/boss_dr12v5_lss/extracted/random1_DR12v5_CMASSLOWZTOT_{North,South}.fits.gz.reservoir_2000000_seed0.npz`
- recon density field weight を `boss_recon`（FKPなし）と `same`（galaxyと同じ重み）で比較する。
  - overlay：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_rw_same.*`
- match_sectors（ISECT整合）を off にして、影響を確認する。
  - overlay：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_matchsectors_off.*`

結果（要約）：
- sector整合（match_sectors=auto; recon入力 z∈[0.2,0.75) / b2例）：
  - 旧（sector_key=ipoly_isect）：north≈0.9995 / south≈0.9625（south の小ポリゴンが落ちる）
  - 新（sector_key=isect; 既定）：north≈0.999987 / south≈0.999997
  - → south の欠落は「random不足」ではなく、(IPOLY<<32)+ISECT の厳密一致が原因（小ポリゴン落ち）。`--sector-key` で切替可能。
- `recon_weight_scheme=same` は明確に悪化：
  - RMSE(s²ξ2; Ross post-recon vs catalog recon)：zbin1≈77.74, zbin2≈73.87, zbin3≈81.08
  - → `boss_recon` を baseline に固定する。
- `match_sectors=off` は改善しない：
  - RMSE(s²ξ2)：zbin1≈59.15, zbin2≈55.28, zbin3≈71.31（baseline: 59.15/55.59/71.75）
  - → `match_sectors=auto` を baseline に固定する。

次：
- Roadmap: 4.5B.17.4（残る ξ2 ギャップを「published pipeline 差」として説明可能にする：window畳み込み／推定量定義差の最小化）。

## 次工程（Stage B：公式recon仕様差/選択関数の切り分け）

優先度の高い候補（例）：
1) **recon の境界条件/マスクの扱い**（periodic FFT 近似 vs 公式実装）
2) **RSD 除去（Kaiser/LOS）と再構成の順序**（等方/異方の定義差）
3) **window / selection の補正**（RR multipoles・窓関数畳み込みの一致条件）
4) **アルゴリズム差**（既存recon実装の導入 or 公式に近い反復法）

当面の方針：
- “小パラメータ調整で改善しない”ことは要約図で固定できたため、次は **実装差（仕様差）の切り分け**へ移る。

## 追加検証（2026-01-17）：残る ξ2 ギャップの位置づけ（B.17.4）

### (1) MW recon-mode iso の確認（不採用）

目的：
- 外部solver（MW multigrid）で `recon-mode=ani` を `iso` に切替えると、Ross post-recon の ξ2 一致度が改善するかを確認する。

実施：
- WSL（Corrfunc; `--threads 24`）で `--recon mw_multigrid --recon-mode iso` を zbin1..3 で再計算し、Ross post-recon multipoles へ overlay。

結果（RMSE; Ross post-recon vs catalog recon; s²ξ₂）：
- `mw_multigrid`（`iso`）：zbin1=78.58, zbin2=74.31, zbin3=83.40
  - 指標：`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_iso_metrics.json`

結論：
- `iso` は ξ2 の一致度が悪化するため、本稿の比較では `ani` を維持する（Ross post-recon の枠組みに合わせる意味でも自然）。

### (2) broadband で吸収できるギャップ量の定量化（fit と解釈の切り分け）

目的：
- Ross の公開フィット（`baofit_pub2D.py`）は broadband を周辺化しているため、
  **生の ξ2 カーブ不一致**が残っていても、BAOピーク（AP/ε）が不一致とは限らない。
- そこで「残る ξ2 ギャップ」が broadband でどこまで吸収され得るかを定量化し、議論の混線を防ぐ。

実施：
- baseline（`mw_multigrid`/`ani`）の zbin1..3 に対して、差分 `diff(s)=ξ2_ross−ξ2_catalog` を
  `diff(s)=a0+a1/s+a2/s²`（Ross err で重み付け最小二乗）でフィットし、
  `ξ2_catalog+diff_fit` の RMSE(s²ξ₂) を再評価した（s∈[30,150] Mpc/h）。

出力：
- `output/cosmology/cosmology_bao_recon_gap_broadband_fit.png`
- `output/cosmology/cosmology_bao_recon_gap_broadband_fit_metrics.json`

結果（RMSE; s²ξ₂；Ross post-recon vs catalog recon）：
- zbin1：59.15 → 12.01（broadband）
- zbin2：55.59 → 12.11（broadband）
- zbin3：71.75 → 15.04（broadband）
- χ²/dof（ξ2; Ross σ(ξ2)）：
  - raw：31.38 / 35.95 / 55.83（zbin1..3）
  - broadband：0.63 / 0.66 / 1.04（zbin1..3）

解釈：
- この結果は「recon が一致した」ことを意味しない（broadband は物理モデルではなく、形状差を吸収する自由度）。
- しかし少なくとも本データでは、残差の大部分が broadband 様の形で表現できるため、
  **“生の ξ2 ギャップ” と “BAOピーク位置（AP/ε）の整合” を分けて扱う**のが妥当である。

### (3) broadband吸収の頑健性（bincent / s_min）

目的：
- (2) は bincent0 のデータを s∈[30,150] Mpc/h で評価したが、
  Ross の公開 fit コード（`baofit_pub2D.py`）の既定は **min=50, max=150** である。
- また Ross は bincent0..4 の複数データ（bin中心を 0..4 Mpc/h シフト）を公開しているため、
  「結論が特定の bincent/レンジに依存していない」ことを確認し、議論のブレを抑える。

実施：
- zbin1..3（baseline: `mw_multigrid`/`ani`）について、bincent0..4 を読み、
  catalog ξ2 は Ross の bincenter に線形補間して比較。
- diff(s)=a0+a1/s+a2/s² のフィットを
  - s≥30（(2) と同じ）
  - s≥50（Ross fit の既定）
  で再評価した。

出力：
- `output/cosmology/cosmology_bao_recon_gap_broadband_sensitivity.png`
- `output/cosmology/cosmology_bao_recon_gap_broadband_sensitivity_metrics.json`

結果（中央値；bincent0..4 のばらつきを含む）：
- s≥30：χ²/dof（broadband, ξ2）= 0.66 / 0.51 / 0.77（zbin1..3）
- s≥50：χ²/dof（broadband, ξ2）= 0.24 / 0.34 / 0.66（zbin1..3）

解釈：
- broadband による吸収可能性は、bincent とレンジ（s_min）に対して概ね頑健である。
- したがって、残る “生の ξ2 ギャップ” は **主として broadband が周辺化する「滑らかな形状差」**として扱い、
  BAOピーク（AP/ε）の議論とは分離して進める方針を維持する。

### (4) Ross基準（baofit_pub2D）での残差評価（mono+quad cov / broadband基底）

目的：
- (2)(3) は ξ2 と対角誤差（Ross の err_xi）で評価したが、Ross の公開 fit（`baofit_pub2D.py`）は
  **データベクトル dv=[ξ0, ξ2] と full covariance（mono→quad の順）**で χ² を評価している。
- そこで “残差の大きさ” を一次コードの評価軸に揃え、過大/過小評価を避ける。

実施（一次コード整合）：
- fitレンジ：`baofit_pub2D.py` 既定に合わせ、r∈(50,150)（strict）を採用。
- dv の並び：Ross の covariance ヘッダ（「quadrupole comes after monopole」）通りに `[mono bins..., quad bins...]`。
- broadband 基底：Ross と同じ `a0+a1/r+a2/r²` を **mono/quad それぞれ**に付与（合計6係数）。
  - r は `baofit_pub2D.py` の rbc（殻内平均距離）を使用。
- catalog 曲線（xi0/xi2）は、Ross の bin center（bincent0..4）へ線形補間して比較。

出力：
- `output/cosmology/cosmology_bao_recon_gap_ross_eval_alignment.png`
- `output/cosmology/cosmology_bao_recon_gap_ross_eval_alignment_metrics.json`

結果（中央値；bincent0..4 のばらつきを含む；χ²/dof；Ross cov）：
- raw：4.33 / 4.33 / 6.37（zbin1..3）
- broadband後：1.59 / 1.70 / 1.95（zbin1..3）

解釈：
- Ross の “fit と同じ評価軸（dv+cov）” で見ると、残差の有意度は (2)(3) の対角誤差評価より小さく見える。
- broadband を許す Ross fit の枠組みでは、残差は概ね O(1) の χ²/dof へ近づくため、
  **recon gap と AP/ε の整合性評価を混線させない**整理がより正当化される。

補助（ξ4 はモデル側に入る）：
- Ross の公開 fit は、AP変換後の xi(μ) に ξ4 テンプレートを含む（観測データベクトルは ξ0/ξ2 のみ）。
- AP歪み（ε）により ξ4→ξ2 の混合が起き得るが、Ross template（α=1）で ε∈[-0.05,0.05] を走査すると、
  max relative Δ(s²ξ2) は概ね数％（例：ε=0.02 で約3%）で、オーダーとしては小さい。
  - よって当面は、broadband/評価軸（dv+cov）の整合を優先し、ξ4 は必要に応じて精密化する。
