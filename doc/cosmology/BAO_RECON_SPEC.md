# BOSS DR12 BAO reconstruction 仕様メモ（一次ソース→実装I/F）

目的：
- Phase 4 / Step 4.5B（BAO一次統計）で **「recon後 ξ0/ξ2」**を再現する際に、議論がブレる主因（recon仕様差）を潰す。
- 本リポジトリの `scripts/cosmology/cosmology_bao_xi_from_catalogs.py` の `--recon` 系オプションを、一次ソースで定義された仕様に合わせて固定する。

参照（一次ソース）：
- Ross et al. 2016（BOSS DR12 BAO）  
  - ソース：`data/cosmology/sources/arxiv_1607.03155_src/BAO_method.tex`（特に Reconstruction 段落）
- Padmanabhan et al. 2012（reconstruction の境界＋RSD取り込み）  
  - ソース：`data/cosmology/sources/arxiv_1202.0090_src/recon.tex`（Sec. “Solving for the Displacement”）

---

## 1. Ross 2016 が明記する recon の固定値（BOSSデータ側）

Ross 2016 の該当段落（`arxiv_1607.03155_src/BAO_method.tex`）の要点：

- recon は Padmanabhan 2012 のアルゴリズムを使用。
- 入力パラメータは（f, b）の2つ：
  - 成長率：fiducial ΛCDM の **z=0.5** における値を採用：**f=0.757**
  - 銀河バイアス：データは **b=1.85**
    - （参考）QPM / MD-Patchy mocks は b=2.2
- recon は **z-bin毎に別々に走らせず**、z=0.2〜0.75 の範囲で **全サーベイ体積に対して1回**走らせる（bulk flow の大スケール寄与を取り込むため）。
- グリッド：**512³**（各セルは約 6 h⁻¹ Mpc）
- 平滑化：**ガウス 15 h⁻¹ Mpc**

実装へ落とすと、BOSS DR12 “Ross post-recon” を狙う場合の最低限の固定は：
- `recon_grid=512`
- `recon_smoothing=15`
- `recon_bias=1.85`
- `recon_f=0.757`（もしくは ΛCDM で z=0.5 を強制）
- `recon_z_min=0.2`, `recon_z_max=0.75`（recon入力の z 範囲）

---

## 2. Padmanabhan 2012 の displacement 方程式（RSDを“方程式側”へ入れる）

Padmanabhan 2012 は、線形オーダーで redshift-space の密度と変位 Ψ を次で結ぶ：

```
∇·F = ∇·Ψ + f ∇·(Ψ_s ŝ) = -δ_gal / b
Ψ_s ≡ Ψ·ŝ
```

- ŝ は LOS（観測方向）。一般には **位置依存（radial）**で、これが並進対称性を壊す。
- Ψ は渦なし（irrotational）と仮定して Ψ=∇φ と置き、φ を解く。
- ŝ が一定（plane-parallel）なら FFT で解けるが、radial の場合は解けない → **有限差分化して線形方程式を解く**（Padmanabhan 2012 は PETSc の GMRES を使用）。

重要な実装含意：
- “RSD除去” は、単に後段で LOS 方向へシフトを足すだけでなく、**Ψの推定方程式に f が入る**（radial では特に）。
- random（均一場）は **-Ψ のみ**で shift（RSD補正はかけない）ことが明記されている（Padmanabhan 2012）。

---

## 3. 本リポジトリの現在実装（近似）と、一次ソースとの差

本リポジトリ（`scripts/cosmology/cosmology_bao_xi_from_catalogs.py`）の `--recon grid` は現状：

- **周期境界のFFT**で Ψ を解く（survey境界・選択関数は近似）。
- RSD-aware solver は **plane-parallel 近似**（global LOS）で2系統を用意している：
  - `--recon-rsd-solver global_constant`：`1/(1+β μ²)`（β=f/b；Kaiser inversion）
  - `--recon-rsd-solver padmanabhan_pp`：`1/(1+f μ²)`（Padmanabhan 2012 の plane-parallel 形）
- RSD除去の shift は `--recon-rsd-shift`（既定1.0）として、銀河側のみ `Ψ + f(Ψ·LOS)LOS` を使う（random は Ψ のみ）。

一次ソースとの差（Ross post-recon と ξ2 がずれる主因候補）：
- Padmanabhan 2012 の radial LOS + finite-difference solver（PETSc/GMRES）に対し、こちらは plane-parallel FFT 近似。
- Ross 2016 の固定値（grid=512³、b=1.85、z=0.2〜0.75で一括recon）に対し、計算設定が一致していない場合がある（スキャンで切り分け中）。

---

## 4. 再現コマンド（Ross 2016 の“設定”に寄せた実行例）

注意：
- Corrfunc を使うので **WSL + `.venv_wsl` + `--threads 24` が必須**（AGENTS.md）。
- 実際に 512³ grid を使うとメモリが大きいので、WSL側の割当を確認する。

例（CMASSLOWZTOT / combined / zbin2；LCDM座標で recon を Ross 設定へ寄せる）：

```
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_from_catalogs.py --sample cmasslowztot --caps combined --dist lcdm --random-kind random1 --z-bin b2 --recon grid --recon-mode ani --recon-rsd-solver padmanabhan_pp --recon-los-shift global_mean_direction --recon-rsd-shift 1.0 --recon-grid 512 --recon-smoothing 15 --recon-bias 1.85 --recon-z-min 0.2 --recon-z-max 0.75 --threads 24"
```

overlay（Ross post-recon との比較）：

```
python -B scripts/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay.py --catalog-recon-suffix __recon_grid_ani --out-png output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__recon_grid_ani.png --out-json output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__recon_grid_ani_metrics.json
```

（注）`--out-tag` を省略しても、recon の非デフォルト設定は自動tag化される（上書き事故防止）。

### 4.1 外部solver（MW multigrid; radial LOS）での実行例（Step 4.5B.17）

Padmanabhan 2012 の「radial LOS + finite-difference solver」に寄せるため、外部実装（Martin White recon_code; multigrid）を
`--recon mw_multigrid` として統合している（WSLのみ）。

注意：
- `mw_multigrid` は入力モードが2通りある：
  - `--dist lcdm`：upstream が（ra,dec,z）を内部で LCDM 距離へ変換（`input_mode=z`）。
  - `--dist pbg` 等：P-model側で comoving distance `D_M(z)` を計算して upstream へ距離 `χ` を直接渡す（`input_mode=chi`）。
- shifted random は既定で **RSD項を入れない**（Padmanabhan 2012）。別流儀（randomにもRSD）を試す場合は `--mw-random-rsd`。
- recon density field の重みは、BOSS公式に寄せるなら `--recon-weight-scheme boss_recon`（FKPを外す）。

例（zbin2。zbin1/3 は `--z-bin b1/b3` に差し替え）：

```
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_from_catalogs.py --sample cmasslowztot --caps combined --dist lcdm --random-kind random1 --z-bin b2 --weight-scheme boss_default --recon mw_multigrid --recon-mode ani --recon-grid 512 --recon-smoothing 15 --recon-bias 1.85 --recon-f 0.757 --recon-z-min 0.2 --recon-z-max 0.75 --recon-weight-scheme boss_recon --threads 24 --out-tag mw"
```

overlay（Ross post-recon との比較）：

```
python -B scripts/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay.py --bincent 0 --catalog-recon-suffix __recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757 --out-png output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid.png --out-json output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_metrics.json
```

---

## 5. 次の課題（Roadmap: 4.5B.15.4 へ接続）

- published multipoles（Ross）の「推定量の定義」を明文化する：
  - ξ4 を含む window mixing の扱い（mixing/補正/畳み込み）
  - broadband（多項式）吸収の範囲（“生ξ”か“処理後ξ”か）
  - bin-centering（bincent0..4）の意味と、比較に使う正準の選択
- その上で、必要なら **Padmanabhan 2012 型（radial LOS / finite-difference）**に近い solver を実装するか、既存実装（外部ライブラリ）の導入で代替する。

---

## 6. Ross 2016 公開 multipoles の定義（baofit_pub2D.py に基づく）

目的：
- “published multipoles（Ross）” が **何を推定しているデータベクトルか**を一次コードで固定し、recon gap の議論を「定義差」から切り離す。

参照：
- 公開データ：`data/cosmology/ross_2016_combineddr12_corrfunc/`
- 公開fitコード：`.tmp_lssanalysis/baofit_pub2D.py`（Ross 2016 README の LSSanalysis）

### 6.1 公開ファイルの中身（観測側の定義）

- `Ross_2016_COMBINEDDR12_zbin{1,2,3}_correlation_function_monopole_post_recon_bincent{0..4}.dat`
- `Ross_2016_COMBINEDDR12_zbin{1,2,3}_correlation_function_quadrupole_post_recon_bincent{0..4}.dat`

ヘッダに明記されている通り、各行は：
- bin center `s`（Mpc/h）
- multipole `xi`（単位なし）
- `err_xi`

注意：
- これらは **`s^2 xi` ではなく `xi`**（プロットでは見やすさのため `s^2` を掛ける）。
- “bincent0..4” は **bin center のシフト**（bin幅 5 Mpc/h に対して 0..4 Mpc/h のオフセット）であり、対応する covariance も同じ bincent を使う。

covariance：
- `Ross_2016_COMBINEDDR12_zbin{1,2,3}_covariance_monoquad_post_recon_bincent{0..4}.dat`
- ヘッダに **「quadrupole comes after monopole」**とあり、並びは `[mono bins..., quad bins...]`。

### 6.2 baofit_pub2D.py が使う “データベクトル” と処理範囲

`baofit_pub2D.py` の `__main__` は、公開ファイルから
- `d0 = correlation_function_monopole_*.dat` の第2列（xi）
- `d2 = correlation_function_quadrupole_*.dat` の第2列（xi）
を読み、`dv = [xi0(min<r<max), xi2(min<r<max)]` としてそのままフィットに渡す。

よって、公開multipoles（Ross）は少なくとも **「broadbandを差し引いた後の残差」等ではなく、生の推定量**として扱われる。

### 6.3 bin-centering（bin center と “平均r” のズレ）

`baofit_pub2D.py` は、モデル評価の `r` として bin center ではなく、殻内の平均距離（pairの `r^2` 重み）を

```
rbc = .75*((r+bs/2)^4-(r-bs/2)^4)/((r+bs/2)^3-(r-bs/2)^3)
```

で計算して用いている（`bs=5` Mpc/h、`r = i*bs+bs/2+binc`）。

重要：
- 公開ファイルに格納されている観測値そのものは bin center `r` の系列。
- ただし `rbc-r` は小さく、recon gap の主因（ξ2 の大きな形状差/符号差）になりにくい。

### 6.4 ξ4 と broadband（多項式）の位置づけ

`baofit_pub2D.py` のテンプレートは `xi0/xi2/xi4` を持ち、AP変換後に

```
xi(mu) = xi0(rp) + P2(mup)*xi2(rp) + P4(mup)*xi4(rp)
```

を積分して model multipoles を作る（つまり **ξ4 は「モデル側」に入る**）。

一方で、観測データベクトルは公開ファイルの `xi0/xi2` のみであり、少なくとも公開データ自体は
- ξ4 を観測として追加しているわけではない
- window deconvolution を明示的に適用しているわけでもない

系統/モデル依存は、主に broadband（例：`A0 + A1/r + A2/r^2`）で吸収している。

---

## 7. 推定量定義の最終固定（正規化 / μ積分 / 誤差モデル）

目的：
- ε（距離写像差）と「座標化差/推定量差」が混ざって議論がブレるのを防ぐ。
- そのために、**座標化（coordinate_spec）**と、**推定量（estimator_spec）**を別々に固定・ハッシュ化して、混在を検出できるようにする。

### 7.1 固定対象（coordinate_spec と estimator_spec）

- `coordinate_spec`（座標化の仕様）
  - redshift の定義（obs / cosmic / cmb）と、使用列（例：Z）
  - LOS 定義（Corrfunc の pairwise midpoint を固定）
  - comoving distance D_M(z)（lcdm: 数値積分＋補間 / pbg: 解析式）
  - recon の仕様（grid/smoothing/bias/f/weights 等：§1〜§4）
  - ξ(s,μ) の推定器（LS / recon 推定器；`xi_estimator`）

- `estimator_spec`（推定量の仕様）
  - s-bin / μ-bin の定義（範囲と刻み、bin中心の扱い）
  - pair-count の正規化（Corrfunc の ordered-pairs 規約に合わせる）
  - ξℓ の μ積分（離散化方法）
  - caps=combined 時の集約方法（paircounts を足し上げる／randomの密度差を補正）
  - screening（簡易ピークfit）で使う誤差モデル（Phase A の近似）

### 7.2 正規化（pair-count → ξ(s,μ)）

本リポジトリの catalog-based（recon なし）の基本推定器は Landy–Szalay：
- ξ(s,μ) = (DDn − 2DRn + RRn) / RRn

ここで DDn/DRn/RRn は「総ペア重み」で正規化した量で、Corrfunc の autocorr が **ordered pairs（i≠j を両方向で数える）**であることに合わせて、
次の total を採用する：
- DD_tot = (Σw)^2 − Σ(w^2)
- RR_tot = (Σw)^2 − Σ(w^2)
- DR_tot = (Σw_gal)(Σw_rnd)

recon の場合は δ_rec = δ_D − δ_S の標準形に固定し、推定器は：
- ξ_rec(s,μ) = (DDn − 2DSn + SSn) / RR0n
  - RR0 は unshifted random（境界/選択関数を固定するため）

### 7.3 μ積分（ξ(s,μ) → ξ0/ξ2）

Corrfunc の μ-bin（0≤μ≤μ_max）に対して、以下を固定する：
- μ-bin：`np.linspace(0, μ_max, nμ+1)`（一様）＋ μ_mid（midpoint）
- P2(μ) = (3μ^2 − 1)/2

離散化は midpoint のリーマン和（uniform dμ）：
- ξ0(s) = Σ ξ(s,μ_i) dμ
- ξ2(s) = 5 Σ ξ(s,μ_i) P2(μ_i) dμ

注：
- 既定は μ_max=1.0（0..1）で、偶関数対称を前提に 0..1 の定義を採用する。
- μ_max を変えると ξℓ の値が変わるため、これは `estimator_spec` に含めて固定し、混在を検出する。

### 7.4 caps=combined（NGC+SGC）集約の固定

caps を結合する場合、NGC/SGC は空間的に非連結なので、
per-cap で paircounts を計算し、合算して ξℓ を作る。

ただし random が subsample（prefix/reservoir 等）されていると cap 間で random 密度がズレて LS が偏り得るため、
本リポジトリでは **random重みを cap ごとに rescale**してから DR/RR/SS を集約する（policy を固定）。

### 7.5 混在検出（hash）

`scripts/cosmology/cosmology_bao_xi_from_catalogs.py` は出力 metrics に次を含める：
- `coordinate_spec_signature` / `coordinate_spec_hash`
- `estimator_spec_signature` / `estimator_spec_hash`

`scripts/cosmology/cosmology_bao_catalog_peakfit.py` は、入力群で `estimator_spec_hash` が混ざっている場合に停止する。
これにより、ε の変動が「理論差」か「推定量差」か混ざる事故を防ぐ。

### 7.5.1 一次入力（LSS catalog）の固定（raw cache policy）

BAO の “公開距離制約”（D_M/r_d, D_H/r_d など）は、観測統計→距離変換→テンプレートfit を含む **派生値**である。
本プロジェクトの一次入力は **LSS catalog（銀河 + random）** とし、再現性のために次を固定する：

- 入口：各 survey の取得スクリプトが `data/cosmology/<survey>/manifest.json` と `extracted/*.npz` を生成する。
  - BOSS：`scripts/cosmology/fetch_boss_dr12v5_lss.py`
  - eBOSS：`scripts/cosmology/fetch_eboss_dr16_lss.py`
  - DESI：`scripts/cosmology/fetch_desi_dr1_lss.py`
- galaxy（一次入力の中心）：**raw FITS を保存し sha256 を固定**する（サイズが現実的な範囲で）。
  - eBOSS：`--download-galaxy-raw`（randomは影響しない）
  - DESI：`--download-missing`（DR1 LRG galaxy を raw として保存）
- random（GB級になりやすい）：raw FITS は既定で保存せず、**決定論的サンプリング**で NPZ を固定する。
  - reservoir を推奨（seed 固定、sample_rows 固定、scan_max_rows は原則 None＝全走査）。
  - 例：`--random-sampling reservoir --random-max-rows 2000000 --sampling-seed 0`
- manifest に記録する最小要件：
  - `raw_path/raw_sha256`（galaxy）
  - `extract.rows_total/rows_saved/row_bytes`（galaxy/random）
  - `extract.sampling`（random の method/seed/sample_rows/scan_max_rows/chunk_rows）

再現コマンド例（online；raw+npz の固定）：
- eBOSS DR16（LRGpCMASS recon）：  
  `python -B scripts/cosmology/fetch_eboss_dr16_lss.py --sample lrgpcmass_rec --caps combined --download-galaxy-raw --random-sampling reservoir --random-max-rows 2000000 --sampling-seed 0`
- eBOSS DR16（QSO）：  
  `python -B scripts/cosmology/fetch_eboss_dr16_lss.py --sample qso --caps combined --download-galaxy-raw --random-sampling reservoir --random-max-rows 2000000 --sampling-seed 0`
- DESI DR1（LRG）：  
  `python -B scripts/cosmology/fetch_desi_dr1_lss.py --sample lrg --caps combined --download-missing --stream-missing --random-sampling reservoir --random-max-rows 2000000 --sampling-seed 0`

### 7.6 誤差モデル（Phase A の screening）

screening（smooth+peak の簡易fit）では、厳密な共分散はまだ使わない。
代わりに、`cosmology_bao_catalog_peakfit.py` は ξ(s,μ) の paircount から作る対角近似（Poisson proxy）を採用している。

注意：
- これは “幾何が合う/合わない” の入口用であり、決定打（確証）は Phase B で full covariance / recon 自前化へ接続する。

### 7.7 共分散（Phase B：Ross 2016 full covariance）

Phase B では、「推定量差」ではなく「幾何差」として ε（AP/warping）を比較できるように、
Ross 2016 が公開する **mono+quad の full covariance** を用いて χ² と CI を評価する。

- 対象：COMBINEDDR12（zbin1..3, post-recon）の `covariance_monoquad_post_recon_bincent{0..4}.dat`
- データベクトル：dv=[ξ0, ξ2]（mono→quad の順。`baofit_pub2D.py` と同順）
- 本リポジトリの実装：`scripts/cosmology/cosmology_bao_catalog_peakfit.py`
  - `--cov-source ross`（明示）
  - `--cov-source auto`（既定；現状は `sample=cmasslowztot,caps=combined,zbinonly` の場合に ross を自動採用）
  - `--ross-dir` / `--ross-bincent` で入力を固定

注：
- この段階でも peakfit の形（smooth+peak）は最小モデルであり、Ross の full template fit（ξ4 を含む）そのものではない。
  ただし、**cov（評価軸）を Ross と揃える**ことで、「対角近似で過信した CI」になることを避け、Phase B の確証に繋げやすくする。
