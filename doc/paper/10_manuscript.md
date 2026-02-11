# P-model（時間波密度P）：重力・時計・光伝播をスカラーで統一する試み

<!-- INTERNAL_ONLY_START -->
この本文は `doc/paper/00_outline.md` の肉付け版であり、本論文の作業用本文です。  
図表の固定パス一覧は `doc/paper/01_figures_index.md` を参照してください。
<!-- INTERNAL_ONLY_END -->

---

## 要旨（Abstract）

本稿は、時間波密度 `P(x)` をスカラー場として導入し、重力・時計・光伝播を同一の概念で説明する **P-model** を提示する。  
本モデルでは **質量近傍で `P>P0`** を採用し、`P` から定義されるニュートン形ポテンシャル `φ` により、重力を **「P勾配へ滑り落ちる」**現象として統一する。  
時計の進みは「重力項（`P0/P` による遅れ）」と「速度項」に分解し、速度項には飽和パラメータ `δ>0` を導入して `v→c` で無限大が現れない形を採用する。  
光はフェルマーの原理（光学経路長の停留条件）に従い、屈折率 `n(P)` により **高P側へ屈折**するとする。Shapiro遅延等で現れる係数2（PPNの `(1+γ)`）は光伝播側の自由度 `β` が担う。[Will2014][BornWolf1999]  
一次ソース（可能な範囲）に基づく再現可能な検証として、LLR（EDC CRD）、Cassini（PDS TDF）、GPS（IGS Final）、EHT（公表値）を統一枠で処理し、差分予測（EHT係数差・速度飽和δ）を示す。[EDC][PDSCassiniSCE1][IGSBKGRootFTP][EHTM87I2019][EHTSgrAI2022]
また、強重力・放射の代表例として二重パルサー（軌道減衰：観測/GRの一致度）と、重力波 GW150914（GWOSC公開strainからchirp位相の整合性）を整理する。[WeisbergHuang2016][Kramer2021][GWOSC][Abbott2016GW150914]

---

## 1. 序論（Introduction）

### 1.1 背景

一般相対性理論（GR）は古典的検証において極めて成功している一方、テンソル形式・座標系・境界条件の選択が複雑であり、直感的理解と検証再現の障壁が高い。[Will2014]  

現代物理学は量子力学と GR という二つの柱を持つが、両者の統一は未解決である。弦理論やループ量子重力などの試みは、検証困難な追加自由度（多次元空間など）を導入することが多い。  

本稿は別のアプローチを提案する。すなわち、「全ての物質は波であり、その媒質は時間そのものである」という仮定から出発し、単一のスカラー場 P(x)（時間波密度）で重力・時計・光伝播を記述する。最終的には、粒子を波の反射として、量子もつれを局所P構造による相関として再解釈し、量子現象から宇宙物理までを 3次元+時間の枠組みで統一することを目指す。  

本稿では、弱場テスト（LLR、Cassini、GPS等）から強場領域（EHT）、動的現象（二重パルサー、重力波）、宇宙論スケールまでを同一の枠組みで検証し、再現可能性と反証可能性を重視する。量子現象への接続は本稿の結論として主張しないが、ベルテスト公開データを用いた観測手続き（time-tag選別／trial 定義）の前提検証（selection 条件の数式化）を整備し、将来の反証へ接続できる状態にした。  

本稿の立場では、GR/SR は観測を優れた精度で説明する幾何学的近似（有効理論）だが、時間の実体（メカニズム）そのものを与えるものではない。P-model は時間を波として仮定し、同じ観測結果を別の機構（時間波密度Pの空間変化）から再導出する試みである。一致する部分は「現実が同じだから結果が似る」ためであり、概念は別である。弱場では一致が近似として現れるが、強場・宇宙論領域では測定可能な差が予言として現れる。  

本プロジェクトの狙いは、弱場テスト領域において、**直感的に説明しやすいスカラー表現**で同等の観測量を再現し得るかを、再現可能な形で検証することにある。

### 1.2 本研究の貢献

- 記号・符号規約を固定し、「質量近傍でP増」「P勾配へ滑り落ちる＝重力」「光もPに寄る＝屈折」を一貫させた。
- データ取得→解析→生成物のパイプラインを整備し、第三者が同じ結果を再現できる形を目指した。
- 図表と簡易解説を単一の補足資料に統合し、結果の俯瞰を容易にした。

### 1.3 本稿の立場と主張の範囲（有効モデル・反証条件）

本稿の主眼は、時間波密度 P(x) を導入したときに「重力・時計・光伝播」の観測量がどこまで一貫して説明できるかを、一次ソースと再現可能な出力で検証することにある。  
そのため本稿は、P の支配方程式（場の方程式）や量子現象まで含む完全理論を提示するものではなく、観測量への写像を先に固定して検証する **有効モデル**として位置づける。

本稿の約束（誤解防止のための運用）を以下に明文化する。

- **fit と predict を分離**する。β と δ は一次ソース（Cassini の PPN γ、超高γ粒子の上限拘束など）から凍結し、以後は固定して外挿（predict）に用いる（凍結値：`output/theory/frozen_parameters.json`）。
- 弱場（太陽系）では GR/SR の予測と高精度に一致することが要請される一方、強場・宇宙論では差分予測が観測可能な形で現れる（例：EHT の影直径係数差、速度飽和 δ、距離指標前提の再接続条件）。
- 公開一次ソースの一部（測地系・時刻系・カタログ等）は標準理論に基づく前処理を含み得る。本稿は現段階で「生データ再解析」を主張せず、入力の一次ソース固定と、補正・系統の可視化（出力の固定）を優先する。
- 反証可能性を重視し、「どの精度で決着するか」を必要精度（例：EHT）と合わせて機械可読で固定する（`output/summary/decisive_falsification.json`、弱場：`output/summary/weak_field_falsification.json`）。
- 量子現象への接続は本稿の結論として主張しない。ただしベルテストの観測手続き（time-tag選別／trial 定義）に関する一次データ再解析の入口と、CHSH 前提（selection 条件）の数式化までは整備し、反証可能な形へ接続する。

一次ソース一覧（URLとローカル保存先）は `doc/PRIMARY_SOURCES.md`、公開マニフェスト（再現入口）は `output/summary/release_manifest.json` を正とする。

---

## 2. 理論（Theory）

主要な記号・符号規約は付録（記号・定義）にまとめ、本文では必要箇所のみを再掲する。

### 2.0 統一解釈（Pで書く）

P-model の主張は「重力・時計・光伝播を、時間波密度 P(x) だけで表現できる」という点にある。  
本稿では物質を束縛波、光を自由波とみなし、波が辿る「最短」は幾何学距離ではなく **到達時間（光学経路長）**の停留条件として理解する。

このとき、観測量への写像は次の形に統一される（詳細は 2.1〜2.4 節）：

- 重力ポテンシャル：φ = -c^2 ln(P/P0)
- 重力（運動方程式）：a = -∇φ = c^2 ∇ ln(P/P0)
- 時計：dτ/dt = (P0/P) · (dτ/dt)_v
- 光：n(P) = (P/P0)^(2β)（β>0）により高P側へ屈折

### 2.1 P とポテンシャル φ

`P(x)` は空間点 `x` における **時間波密度**、`P0` は遠方基準の密度とする。  
本プロジェクトでは **質量近傍で `P>P0`** を採用する。

ニュートン形ポテンシャル（無限遠で0、質量近傍で負）へ写像するため

$$ \\phi \\equiv -c^2 \\ln\\left(\\frac{P}{P_0}\\right) $$

を定義する。点質量 `M`（弱場・静的）では

$$ \\frac{P(r)}{P_0} = e^{\\frac{GM}{c^2 r}} ,\\quad \\phi(r)=-\\frac{GM}{r} $$

となり、ニュートン極限を回収する。

### 2.2 重力＝P勾配へ滑り落ちる

重力加速度はニュートン形に統一し、

$$ \\mathbf{a} = -\\nabla\\phi = c^2\\,\\nabla\\ln\\left(\\frac{P}{P_0}\\right) $$

とする。これにより「P勾配へ滑り落ちる＝重力」という言葉と式を同一化する。

### 2.3 時計（重力項 + 速度項）

観測時刻 `t`（外部基準時）と固有時 `τ`（時計自身の時刻）を分け、時計の進みを

$$ \\frac{d\\tau}{dt} = \\frac{P_0}{P}\\left(\\frac{d\\tau}{dt}\\right)_v $$

と分解する。静止時計（`v=0`）では、弱場で

$$ \\frac{P_0}{P} = e^{\\phi/c^2} \\approx 1 + \\frac{\\phi}{c^2} $$

となり、一次の重力赤方偏移と整合する。  
速度項は「100%（無限大）は現実に出ない」という仮定に基づき、飽和パラメータ `δ>0` を導入して

$$ \\left(\\frac{d\\tau}{dt}\\right)_v = \\sqrt{\\frac{1-\\frac{v^2}{c^2}+\\delta}{1+\\delta}} $$

と置く。

このとき、SRのローレンツ因子に対応する「有効ガンマ」を

$$ \\gamma_{\\rm eff}(v) \\equiv \\frac{1}{\\left(\\frac{d\\tau}{dt}\\right)_v}=\\sqrt{\\frac{1+\\delta}{1-\\frac{v^2}{c^2}+\\delta}} $$

と定義すると、`v→c` の極限で

$$ \\gamma_{\\max}=\\lim_{v\\to c}\\gamma_{\\rm eff}(v)=\\sqrt{\\frac{1+\\delta}{\\delta}} \\approx \\frac{1}{\\sqrt{\\delta}} \\quad (\\delta\\ll 1) $$

となり、**発散ではなく飽和**が生じる。

また、`γ_eff(v)` の定義から

$$ \\frac{v^2}{c^2} = 1+\\delta - \\frac{1+\\delta}{\\gamma_{\\rm eff}(v)^2} $$

となるため、`γ_eff(v)` が最大値 `γ_max` に近づく極限で `v→c` に収束する（速度そのものの上限は `c` のまま）。  
なお `δ` は「速度の上限を `c` 未満に下げる」ための係数ではなく、`v→c` の極限で `dτ/dt` が 0 に落ちない（したがって `γ` が無限大にならない）ようにするための係数である。

補足：`δ` を速度側だけの任意パラメータに見せないため、飽和パラメータを時間波密度で与える（例：`δ(P)=δ0(P/P0)^k`）と書くこともできる（付録A.3）。本稿で扱う弱場（太陽系）では `P/P0≈1` のため `δ(P)≈δ0` とみなせる。

### 2.4 光（屈折：高P側へ曲がる）

光（自由波）は P により屈折率が変わると置き、

$$ n(P)=\\left(\\frac{P}{P_0}\\right)^{2\\beta} \\quad (\\beta>0) $$

を採用する。光路はフェルマーの原理（光学経路長 `∫ n ds` の停留）で定まり、結果として高P側へ屈折する。  
ここで `β>0` を採るため、質量近傍（`P>P0`）では `n>1` となり、`c_eff(P)=c/n(P)` が低下する。  
したがって光は光学的に高P側（高n側）へ曲がる。

また、本稿でいう「波が最短経路を好む」とは、幾何学的距離 `∫ ds` の最小化ではなく、光学経路長 `∫ n ds`（同値に到達時間や位相蓄積）の停留条件を指す。
PPN対応は

$$ 1+\\gamma = 2\\beta $$

で表し、GR整合は `β=1` で担保する。[Will2014]

**重要（混乱防止）**：Shapiro遅延等で現れる係数2は `β` が担う。よって、`φ` の定義式（2.1節）は固定し、`φ` 側へ `sqrt()` を混ぜて係数を吸収しない。

### 2.5 宇宙論（宇宙膨張なしの赤方偏移：背景P）

宇宙論的スケールでは、局所重力による `P_local(x)` と、宇宙全体の時間基準を与える `P_bg(t)` を分けて

$$ P(x,t) = P_{\\rm bg}(t)\\,P_{\\rm local}(x) $$

と書く（背景Pの時間変化を導入）。ここで、観測される周波数比を

$$ \\frac{\\nu_{\\rm obs}}{\\nu_{\\rm em}} = \\frac{P_{\\rm obs}}{P_{\\rm em}} $$

と置くと、赤方偏移（定義 1+z = ν_em/ν_obs）は

$$ 1+z = \\frac{P_{\\rm em}}{P_{\\rm obs}} $$

となる。もし `P_bg(t)` が時間とともに減少するなら、過去ほど `P_em>P_obs` となり `z>0`（赤方偏移）が自然に出る。  
これは「空間が伸びる」ことを仮定せず、「時計の刻み（時間の基準）が宇宙規模で変化する」ことだけで赤方偏移が出る、という主張である。

低 `z`（近傍）では

$$ H_0^{(P)} \\equiv -\\left.\\frac{d}{dt}\\ln P_{\\rm bg}(t)\\right|_{t_0} $$

を導入すると

$$ z \\approx H_0^{(P)}\\,\\Delta t \\approx H_0^{(P)}\\,\\frac{D}{c} $$

となり、Hubble則の形が現れる（ここで `Δt` はルックバック時間、`D` は静的近似の距離 `D≈cΔt`）。  
この節の式と可視化は「赤方偏移だけから宇宙膨張や特異点を唯一の説明として結論づけられない」ことを示す **代替機構の提示**であり、宇宙論全体（CMB/元素合成など）の置換は今後の差分予測と一次ソース整備が必要である。

`output/cosmology/cosmology_redshift_pbg.png`

この節は内容が多いため、以下の3章で整理する。

- **2.5.1 BAO一次統計の詳細について**：銀河カタログ＋randomカタログから ξ(s,μ)→ξℓ を再計算し、P-model距離写像で座標化を入れ替えて幾何（特に異方）を比較する。
- **2.5.2 宇宙論 I：距離指標と独立な検証**：SN time dilation（スペクトル年齢）/ CMB温度 T(z) / AP（幾何比）など、距離指標と独立な一次ソース・無次元量で背景Pの最小予測を検証する。
- **2.5.3 宇宙論 II：距離指標依存と BAO の前提**：DDR/Tolman/誤差予算/再接続要件/BAO圧縮出力の前提を整理し、張力がどこから来るかを切り分ける。

#### 2.5.1 BAO一次統計の詳細について

BAO の `D_M/r_d`, `D_H/r_d` や、それを用いた DDR（`ε0`）などの“圧縮出力”は、観測統計（角度×赤方偏移）をある前提（fiducial cosmology / テンプレート / 再構成 / 系統の扱い）で処理して得られる推定量である。前提が異なれば同じ観測から異なる BAO 出力が得られ得るため、出力値だけを「観測事実」として解釈すると物理的実像を誤る可能性がある。

P-model 側で宇宙論を検証するなら、BAO の既存出力をそのまま採用するのではなく、角度赤方偏移統計（相関関数/パワースペクトル）という一次情報から、距離変換を含めて P-model 側で再導出するのが自然である（ただし現段階では“入口”）。

その入口として、BOSS DR12 の post-reconstruction 相関関数 multipoles ξ0, ξ2（Ross et al. の公開データパッケージ）を用い、最小モデル（smooth+peak）で BAOピーク位置を fit し、AP の warping（ε）を ξ2（異方）で評価した。[Ross2016CorrfuncData]
ここでの ε は Alcock–Paczynski（AP）warping の指標であり、後述の重力赤方偏移テスト（GP-A / Galileo）の偏差パラメータ ε や DDR の ε0 とは別定義である。

`output/cosmology/cosmology_bao_xi_multipole_peakfit.png`

ここでは r_d を自由（全体スケール α に吸収）とし、ε を (i) free、(ii) ε=0、(iii) 静的背景P（指数 P_bg）の F_AP 比から予測される ε に固定、の3条件で比較している。これは次段階（銀河カタログからの ξ(s,μ) 再計算や P(k) multipoles からの再導出）へ接続するための予備的な診断図である。

pre-recon（再構成前）ξℓ でも同様にクロスチェックし、再構成手法の影響で結論が過度に左右されないかを確認する。

`output/cosmology/cosmology_bao_xi_multipole_peakfit_pre_recon.png`

（pre-recon の一次統計は Satpathy et al. の公開データパッケージを用いる。）[Satpathy2016CorrfuncData]

さらに、BOSS DR12v5 LSS の銀河カタログと random カタログ（pre-recon）から ξ(s,μ) を再計算し、そこから ξ0, ξ2 を積分して同じ peakfit（smooth+peak）を適用した。公開 multipoles（post/pre-recon）との ε（AP warping）一致度をクロスチェックし、確証段階（Stage B）へ進む前に、(i) 再計算パイプラインの整合と、(ii) recon の効果（pre→post で ξ2 がどの程度変わるか）を分離して確認する。ここでの位置づけはスクリーニング段階（Stage A）であり、誤差は対角近似の粗い CI に留める。

`output/cosmology/cosmology_bao_catalog_vs_published_crosscheck.png`

加えて、ε という1つの指標だけでは見落としやすい「正規化・振幅・形状」の差を確認するため、公開 multipoles（Ross post-recon / Satpathy pre-recon）と catalog-based（galaxy+random）の s²ξ0/ξ2 曲線を直接重ねて比較した（Stage A の品質管理）。ここで大きな破綻（スケールの桁違い、ピーク消失など）があれば、Landy–Szalay 正規化や入力仕様（random/重み/セクタ一致）の問題が疑われるため、Stage B へ進む前に切り分ける必要がある。

この混線を防ぐため、本リポジトリでは座標化仕様（coordinate_spec）と推定量仕様（estimator_spec）を別々に固定し、出力 metrics に `coordinate_spec_hash` / `estimator_spec_hash` を同梱して混在を検出できるようにしている（詳細は `doc/cosmology/BAO_RECON_SPEC.md` §7）。

さらに、catalog-based（pre-recon）に加えて、銀河＋randomカタログから簡易 reconstruction（grid）を適用して ξℓ（post-recon相当）も生成し、公開 multipoles（Ross post-recon）への一致度がどの程度改善するかを同一図で確認した。

このとき recon 版の推定器は、reconstruction の標準形（再構成密度場 δ_rec=δ_D−δ_S の相関）として

$$ \\xi_{\\rm rec}(s,\\mu)=\\frac{DD - 2DS + SS}{RR_0} $$

（D=displaced galaxies, S=shifted randoms, RR0=unshifted randoms）を採用し、推定器の差が「理論差」と混ざらないよう固定した。

現段階では recon の前提（境界/選択関数/LOS/RSD/格子割当/平滑化）により特に ξ2 が大きく揺られる。periodic FFT の grid recon（recon-mode=iso を baseline）では Ross post-recon の ξ2 ギャップが残る一方、radial LOS + finite-difference solver に近い外部recon（MW multigrid）を導入すると RMSE(s²ξ2; Ross vs catalog recon) が zbin1=57.6, zbin2=52.1, zbin3=64.5 まで縮む（それでも chi2/dof は 1 を大きく超える）。指標は `output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_metrics.json` に保存している。なお、格子割当を CIC にすると expected(random) が極小のセルまで非ゼロ化して `delta=(n-expected)/expected` が爆発し得るため、mask（expected床; `mask_expected_frac`）を仕様として固定している（CIC baseline は `mask_expected_frac=1.5`）。

なお、MW multigrid の `recon-mode` を `iso` に切り替えると RMSE(s²ξ2) が **78.6 / 74.3 / 83.4** と悪化するため（zbin1..3）、MW 側でも `ani` を維持する。

さらに、Padmanabhan 2012 の別実装案として shifted randoms 側にも RSD項を入れる（`--mw-random-rsd`）差分を評価したが、RMSE(s²ξ2) は **80.9 / 74.1 / 79.6** と悪化した。従って本稿では Padmanabhan 2012 の既定（random側RSDなし）を採用する。指標は `output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay__mw_multigrid_mwrndrsd_metrics.json` に保存している。

また、NGC/SGC を混ぜた combined と north/south の内部差がどの程度出るか（観測系の系統の目安）を固定するため、Δs_peak と曲線RMSE（NGC vs SGC; s²ξℓ）の要約図を追加した（`output/cosmology/cosmology_bao_xi_ngc_sgc_split_summary.png`）。

この “ξ2 ギャップ”について、これまでに試した recon 設定（solver/shift/assignment/Ωm/重み等）で RMSE がどの程度動くかを、既存の overlay metrics から集約した要約図を固定した（Stage B の入口）。切り分けメモは `doc/cosmology/BAO_RECON_GAP.md` に置く。

`output/cosmology/cosmology_bao_recon_gap_summary.png`

さらに、残る ξ2 ギャップが Ross の公開フィットで周辺化される broadband（多項式）でどこまで吸収できるかを定量化した。差分 `diff(s)=ξ2_ross−ξ2_catalog` に `diff(s)=a0+a1/s+a2/s²` を当て、catalog側へ加えると、baseline（MW multigrid/ani）では RMSE(s²ξ2) が **59/56/72 → 12/12/15**（zbin1..3）まで縮む。さらに ξ2 の χ²/dof（Ross σ(ξ2)）も **31/36/56 → 0.63/0.66/1.04**（zbin1..3）まで縮むため、“生の ξ2 ギャップ” の大部分が broadband として周辺化され得ることが分かる。従って本稿では、（i）BAOピーク位置（AP/ε）の議論と、（ii）生の ξ2 形状差（broadband/定義差）の議論を分離して扱う。

`output/cosmology/cosmology_bao_recon_gap_broadband_fit.png`

この結論は、Ross fit の既定レンジ（s∈[50,150]）および bin-centering（bincent0..4）に対して概ね頑健である（`output/cosmology/cosmology_bao_recon_gap_broadband_sensitivity.png`）。

さらに、Ross の一次コード（`baofit_pub2D.py`）が実際に用いる評価軸（dv=[ξ0,ξ2] と full covariance）に揃えて残差を評価すると、bincent0..4 の中央値で χ²/dof（Ross cov）は **raw=4.33/4.33/6.37 → broadband後=1.59/1.70/1.95**（zbin1..3）となり、(2)(3) の対角誤差評価よりも「ギャップの有意度」が穏やかに見える。これは “recon が一致した”ことを意味しないが、少なくとも **Ross fit が周辺化する自由度（broadband）と評価軸（cov）を揃えると、recon gap は AP/ε の議論と分離しやすい**ことを支持する（`output/cosmology/cosmology_bao_recon_gap_ross_eval_alignment.png`）。

補足として、Ross fit はモデル側に ξ4 テンプレートを含むが、AP歪み（ε）により ξ4→ξ2 の混合が起きるとしても、Ross template（α=1）で ε∈[-0.05,0.05] を走査した範囲では max relative Δ(s²ξ2) は概ね数％であり、オーダーとしては小さい（同図下段）。

追加の診断として、catalog-based の pair-count grid（RR0/SS）から window anisotropy の大きさ（RR2/RR0, SS2/SS0）を計算した。BOSS DR12v5（CMASSLOWZTOT, combined）では s∈[30,150] Mpc/h において max|RR2/RR0| が zbin1≈0.058, zbin2≈0.156, zbin3≈0.139（SS は ≈0.055–0.056, 0.146–0.148, 0.128–0.131）となり、窓関数の異方性が無視できない規模で存在する。

さらに RR multipoles（w2,w4,w6,w8）から、Legendre積の結合係数を用いて window mixing 係数 m20（ξ0→ξ2 の漏れ）と m22 を計算し、漏れの規模を **s²(m20·ξ0)** として見積もった。結果として、max|s²(m20·ξ0)| は zbin1≈2.2, zbin2≈3.9, zbin3≈3.3 程度であり、Ross post-recon vs catalog recon の ξ2 ギャップ（RMSE(s²ξ2)≈50–65（MW multigrid）/ ≈70–80（grid recon））に比べて十分小さい。従って、**窓関数の ξ0↔ξ2 混合だけで ξ2 ギャップ全体を説明することは難しく**、主因は引き続き recon 仕様差（境界/マスク/RSD除去/重み）や published pipeline 側の window 畳み込みの扱い差として切り分ける必要がある。

`output/cosmology/cosmology_bao_catalog_window_multipoles.png`

`output/cosmology/cosmology_bao_catalog_window_mixing.png`

確証段階（Stage B）の狙いは、reconstruction の一次ソース仕様（公式recon手順・境界/選択関数処理）へ寄せて公開 multipoles に対する再現性を上げたうえで、距離写像を P-model 側へ差し替えたときの幾何（特に異方）を評価することにある。

`output/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay.png`

また、reconstruction のチューニング（例：平滑化スケール）が一致度にどの程度効くかを確認するため、grid recon（iso）で平滑化スケール（10/15/20 Mpc/h）などを振り、Ross post-recon との一致度をスキャンした（RMSEに加え、Ross 公開 covariance から chi2/dof（xi0+xi2）も併記）。結果として ξ0 の一致はわずかに改善し得るが、ξ2 の RMSE は大きく（~76）、単純な smoothing 調整だけでは Ross post-recon の ξ2 を再現できない。一方、solver/LOS を外部実装（MW multigrid）へ寄せると ξ2 の RMSE が明確に改善するため、差分の支配項は smoothing より **solver/LOS/境界・重みの仕様差**側である可能性が高い。したがって、Stage B では（A）公式reconに近い仕様（境界/selection/重み/RSD除去）へ寄せて再現性を上げる、または（B）recon 済みカタログ（一次入力）を用いて距離写像だけを差し替える、などの“recon前提そのもの”の検証が必要になる。

`output/cosmology/cosmology_bao_recon_param_scan.png`

同様に、RSD項を含む簡易 reconstruction（grid; ani）についても一致度をスキャンし、χ2/dof（xi0+xi2）を併記したが、chi2/dof は依然として 1 を大きく超える（公開パイプラインとの差が支配的）。

`output/cosmology/cosmology_bao_recon_param_scan_ani.png`

ここでの catalog-based ξℓ 再計算は、A0（座標化仕様）を固定している。A0 は以下である（議論のブレ防止）。

- redshift（z）：`z_source=obs`（既定：列 Z。Z_CMB 等は列がある場合のみ選択）
- LOS：pairwise-midpoint（Corrfunc 定義）
- comoving distance `D_M(z)`：
  - lcdm：台形則の累積積分（`Ωm=0.315`, `n_grid=6000`, `z_grid_max>=max(z)`（例：BOSSは2.0、eBOSS QSOは2.5））+ 線形補間
  - pbg（静的）：解析式 `D_M=(c/100) ln(1+z)`
- 距離写像の差し替え：同一インターフェース（z→D_M）で model を切替
- reconstruction（grid）の密度場重み：既定は paircount と同じ（`recon_weight_scheme=same`）。別案として `boss_recon`（密度場から FKP を外す）も試したが、Ross post-recon との ξ2 RMSE はほぼ改善せず、支配要因ではないことを確認した。

同一 A0 の下で、銀河+random から得た ξ0/ξ2 に最小モデル（smooth+peak）peakfit を適用し、AP の warping（ε）を推定した。
基本は Stage A（スクリーニング：対角近似）だが、**z-bin（CMASSLOWZTOT/combined）の pre-recon については Satpathy 2016 の full covariance（dv=[ξ0,ξ2]）で再評価**し、結論が「対角近似の過信」や「recon実装差」に混ざらないようにしている（`--cov-source satpathy`；詳細は `doc/cosmology/BAO_RECON_SPEC.md` / `doc/cosmology/BAO_RECON_GAP.md`）。主な結果（pre-recon; out_tag=prerecon）は以下。

- CMASS（combined; z≈0.56）：ε(lcdm)≈+0.032（要改善）、ε(pbg)≈+0.042（要改善）
- LOWZ（combined; z≈0.33）：ε(lcdm)≈−0.010（OK）、ε(pbg)≈+0.018（OK）
- z-bin（CMASSLOWZTOT; z≈0.38/0.51/0.61）：ε(z) の z依存を比較（特に ℓ=2 を重視）

`output/cosmology/cosmology_bao_catalog_peakfit_cmass_combined.png`

`output/cosmology/cosmology_bao_catalog_peakfit_lowz_combined.png`

eBOSS DR16（LRGpCMASS; recon; z≈0.72）についても、recon 済み銀河+random から同様に ξℓ→peakfit を行い、まずは座標化差（lcdm vs pbg）の符号・オーダーをスクリーニングした（random は reservoir 2M（seed=0）; diag）。現状は pbg: ε≈+0.008±0.010（|z|≈0.82σ）、lcdm: ε≈−0.020±0.010（|z|≈2.01σ）。

注：random を prefix 抽出した場合、同じ設定でも ε が大きく動く例があり、ε の差が「理論差」ではなく「抽出仕様差」に混ざり得る。したがって、random 抽出法も座標化仕様の一部として固定し、north/south（NGC/SGC）分割も含めて頑健性を確認する必要がある（詳細は metrics を参照）。

`output/cosmology/cosmology_bao_catalog_peakfit_lrgpcmass_rec_combined.png`

eBOSS DR16（QSO; z≈1.52）についても、銀河+random から同様に ξℓ→peakfit を行い、座標化差（lcdm vs pbg）を一次統計でスクリーニングした（random は reservoir 2M（seed=0）; diag）。現状は combined: pbg ε≈−0.004±0.021（|z|≈0.19σ）、lcdm ε≈−0.004±0.013（|z|≈0.31σ）、north/south 分割でも概ね |z|≲1.3σ の範囲で推移している（ただし pbg は α がスキャン上限（α=1.1）に張り付くため、距離スケール側は下限（α≥1.1）として解釈する）。

`output/cosmology/cosmology_bao_catalog_peakfit_qso_combined.png`

DESI DR1（LRG; 0.4<z<1.1; z_eff≈0.780）についても、銀河+random から同様に ξℓ→peakfit を行い、座標化差（lcdm vs pbg）を一次統計でスクリーニングした（random は reservoir 2M（seed=0）; weight は WEIGHT_FKP*WEIGHT; diag）。現状は combined: pbg ε≈+0.024±0.010（|z|≈2.47σ）、lcdm ε≈+0.016±0.008（|z|≈2.06σ）で、north/south 分割でも符号は同じだが、south は |z|≈1.63σ と比較的穏やかで、north は |z|≈2.38σ と大きい。なお、pbgでは α がスキャン上限（α=1.1）に張り付くため、距離スケール側は下限（α≥1.1）として解釈する（εは同一範囲で評価）。

`output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off.png`

DESI BAO（VI）論文は、距離指標（D_M/r_d, D_H/r_d）の形で LRG1（0.4–0.6）/LRG2（0.6–0.8）の制約を公開している。これは解析パイプラインと fiducial（ΛCDM）を通した **公開距離制約（派生値）**であり、本リポジトリでは一次ソースとは扱わない。ただし **F_AP=D_M/D_H は r_d が相殺される**ため、AP異方（ε）の cross-check として、公開距離制約（2×2の相関つき）から ε_expected を計算できる。なお、同じ mean/cov（Gaussian BAO likelihood）は DESI data portal が案内する `CobayaSampler/bao_data`（`desi_2024_gaussian_bao_*`）にも公開されており、本リポジトリでは `data/cosmology/desi_dr1_bao_bao_data.json` に固定キャッシュした。
一方で、post-recon の ξℓ（ℓ=0,2）や P(k) multipoles の **測定ベクトル＋共分散（fitに使う形）**を、DESI DR1 公開領域（`data.desi.lbl.gov`）から直接取得して overlay できる一次ソースは、現状の探索範囲では特定できていない（探索ログ：`doc/cosmology/DESI_PUBLISHED_2PT_PRODUCTS.md`）。したがって、現時点の DESI に関する確証（Stage B）は、(i) 本リポジトリで銀河+random から再構築した一次統計 ξℓ（+ jackknife dv+cov）を主軸とし、(ii) 公開距離制約（mean/cov; 派生値）から計算した ε_expected は cross-check として併用する。公開 multipoles/cov が入手でき次第、overlay による切り分けへ更新する。
そこで、LRG1/LRG2 の z-bin に揃えて catalog-based ξℓ→peakfit（diag）を再計算し、ε_expected と比較した。現状は pbg 座標では LRG2 が概ね一致する一方、lcdm 座標の LRG2 は peakfit がスキャン境界に張り付き、公開距離制約 ε_expected と大きくずれる。これは「diag（誤差）＋最小モデル（smooth+peak）＋reconstruction/窓関数の未整合」の影響が混ざり得るため、確証段階（Stage B）では dv=[ξ0,ξ2]+cov を用い、cov 推定法（jackknife/RascalC）や tracer を跨いだ安定性で切り分ける必要がある（公開 multipoles/cov（一次ソース）が入手できれば overlay による追加検証を行う）。

`output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins.png`

さらに、公開 multipoles/cov が未特定でも「diag 近似だけが原因か」を切り分けるため、**cov 代替策として sky jackknife（dv+cov）**で peakfit を再評価した。現状では LRG1（z≈0.51）は概ね |z|<1 で整合する一方、LRG2（z≈0.71）は pbg 座標でも |z|≈3 と大きめの歪みが残り、diag 近似だけでは説明できない可能性を示す（詳細は metrics を参照）。

`output/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins_full_r0__jk_cov_both_full_r0.png`

加えて、LRG2 の「不一致」が peakfit の任意設定（r_range/テンプレート/グリッド/ℓ=2重み等）に起因していないかを切り分けるため、jackknife cov を固定したまま peakfit の設定感度をスキャンした。ここでの設定感度は **最小モデル（smooth基底 pmax=2：1, 1/r, 1/r^2 + peak）を固定**した上での結果で、lcdm の LRG2 は設定を変えても最良でも |z|≈3.3 を下回らず（多くは |z|≈9 付近）となり、pmax=2 の範囲では不一致が比較的頑健に見える。一方、pbg の LRG2 は |z| が 1.5〜6 程度で動き、例えばテンプレート中心 r0=110 では |z|≈1.5（要改善）まで下がる。

ただし、broadband（smooth基底）を pmax=3（1/r^3 追加）へ拡張すると、ε は LRG1/LRG2 間で **入れ替わる**ほど大きく動き得る。したがって現時点の “LRG2 の不一致” は、幾何（AP）そのものだけでなく **ξ2 の broadband/feature 形状と、そのモデル化自由度**に強く依存し得る（過剰な broadband が ε を吸収する可能性）。確証（Stage B）は dv=[ξ0,ξ2]+cov を用いた安定性（cov 推定法や tracer を跨ぐ）で切り分けるか、公開 multipoles/cov（一次ソース）が入手できる場合は overlay で分離する必要がある（より踏み込むなら full-shape/RSD を含む枠組みが必要）。

`output/cosmology/cosmology_bao_catalog_peakfit_settings_sensitivity__lrg_combined__w_desi_default_ms_off_y1bins.png`

さらに、座標化仕様のブレが LRG2 の不一致に混ざっていないかを確認するため、z_source（obs→cmb（CMB dipole補正））と lcdm 距離積分設定（n_grid / z_grid_max）を変更して ξℓ→peakfit（diag）を再計算した。結果、LRG2（0.6–0.8）の lcdm ε≈−0.096 はほぼ不変（CMB補正でも ε≈−0.098、積分粗密でも差は ≲O(10^-3)）であり、少なくとも「今回の範囲の座標化差」では LRG2 の不一致は説明しにくい。

`output/cosmology/cosmology_bao_catalog_coordinate_spec_sensitivity__lrg_combined__w_desi_default_ms_off_y1bins.png`

`output/cosmology/cosmology_desi_dr1_bao_y1data_eps_crosscheck__w_desi_default_ms_off_y1bins.png`

ここで `z_score_vs_y1data` は Δε/σ_expected（公開 mean/cov 由来）で、`z_score_combined` は Δε/√(σ_expected²+σ_fit²)（公開誤差と peakfit の誤差を二乗和で合成）である。現時点では DESI の公開 multipoles/cov（fitに使う形）が未特定のため、ξℓ→peakfit は **Stage A（スクリーニング：対角近似）**に加え、dv=[ξ0,ξ2]+cov を自前生成して cov 推定法（jackknife/RascalC）で再評価する（Stage B）。公開 multipoles/cov が入手でき次第、overlay による切り分け（pipeline 系統の追加検証）へ更新する。

加えて、DESI DR1 の別トレーサ（LRG3+ELG1, Lyα QSO）についても、raw/VAC から dv=[ξ0,ξ2]+cov を自前生成し、同じ ε_expected cross-check（`z_score_combined`）を行った。結果として pbg 座標では、LRG3+ELG1 は z_score_combined≈−5.5..−6.4、Lyα QSO は ≈−3.3..−3.0 と、covariance shrinkage λ と cov 推定法（jackknife/RascalC/VAC）を跨いでも符号が反転せず |z|≥3 を満たす（stable）ことが分かる。これを「screening（Stage A）→確証（Stage B）（距離指標依存の張力）」の昇格条件（min_tracers=2）として判定した結果は `output/cosmology/cosmology_desi_dr1_bao_promotion_check.json` に固定した。ただし本稿では、この張力を直ちに P-model 自体の棄却と解釈せず、2.5.3 節の枠組みで「距離指標の前提に対して必要な補正（再接続条件）」として定量化する。

`output/cosmology/cosmology_desi_dr1_bao_promotion_check.png`

補助的に、DESI LRG1/LRG2（y1bins）について、(i) μ-wedge（横/縦）からのピーク差（s⊥, s∥）と (ii) ξ2 の「特徴点」（broadband 除去後の支配極値）を確認した。
結果として LRG2（z≈0.71）では、lcdm の peakfit が ε≈−0.096（|z|≈9.4σ）と大きい一方で、wedge 由来の ε_proxy は ≈−0.012 と小さく、wedge ピーク差だけを見る限り強い異方は示唆されない。したがって LRG2 の “不一致” は、幾何（AP）そのものより、ξ2 の broadband/feature の形（特徴点の位置や符号）が peakfit に強く効いている可能性がある（少なくとも追加の切り分けが必要）。

`output/cosmology/cosmology_bao_catalog_wedge_anisotropy__w_desi_default_ms_off_y1bins.png`

`output/cosmology/cosmology_bao_catalog_peak_summary__w_desi_default_ms_off_y1bins.png`

`output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__prerecon.png`

z-bin（pre-recon; Satpathy cov）の代表例として、b1（z_eff≈0.384）では ε(pbg)=0.036±0.015（|z|≈2.45σ）、ε(lcdm)=0.026±0.015、Δε≈+0.010 と推定される（詳細は metrics を参照）。これは、post-recon（MW multigrid + Ross cov）の Δε≈+0.02 と同符号・同オーダーであり、距離写像差の効果が「reconにのみ由来する」わけではないことを示す。

reconstruction（grid, iso; out_tag=recon_grid_iso）を入れた場合は以下。

- CMASS（combined; z≈0.56）：ε(lcdm)≈+0.022（要改善）、ε(pbg)≈+0.052（不一致）
- LOWZ（combined; z≈0.33）：ε(lcdm)≈−0.020（OK）、ε(pbg)≈+0.184（スキャン範囲の拡張が必要; 不一致）

`output/cosmology/cosmology_bao_catalog_peakfit_cmass_combined__recon_grid_iso.png`

`output/cosmology/cosmology_bao_catalog_peakfit_lowz_combined__recon_grid_iso.png`

`output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_grid_iso.png`

さらに、公開 multipoles（Ross post-recon）への整合が高い外部実装（MW multigrid; ani）の出力に対しても、同じ枠（dv=[ξ0,ξ2] + Ross cov）で peakfit を適用し、ε の推定が “recon実装差” に過度に依存していないかを確認した。結果、z-bin（CMASSLOWZTOT/combined; lcdm）の ε は baseline（out_tag=none）から大きくは動かず、概ね |Δε|≲0.01 の範囲で同オーダーのまま推移する。さらに、同じ MW multigrid recon のまま距離写像だけを pbg に差し替え（`input_mode=chi`）、ε の変化量を評価した。

- z-bin（MW multigrid; z≈0.38/0.51/0.61）：ε(lcdm)≈+0.016 / −0.014 / −0.018
- z-bin（MW multigrid; z≈0.38/0.51/0.61）：ε(pbg)≈+0.036 / +0.010 / +0.000
  - 参考：Δε≡ε(pbg)−ε(lcdm)≈+0.020 / +0.024 / +0.018
  - 付記：pbgでは α がスキャン上限（α=1.1）に張り付くため、距離スケール側は「下限（α≥1.1）」として解釈する（εは同一範囲で評価）。

`output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757.png`

さらに、系統切り分けのために NGC/SGC を分割して同じ recon（grid; iso）を適用し、ε の sky cap 依存を確認した。結果として特に SGC 側で ε が大きくずれ（CMASSでは P_bg が不一致、LOWZでは lcdm も不一致）、cap 分割は「理論差」と独立な系統検出器として機能し得ることが分かった。よって Stage B では、選択関数・窓関数・recon 仕様を一次ソースへ寄せたうえで、cap 依存が縮退するか（縮退しないなら観測/解析系統として扱うか）を明示する必要がある。

- CMASS（NGC）：ε(lcdm)≈+0.006（OK）、ε(pbg)≈+0.016（OK）
- CMASS（SGC）：ε(lcdm)≈+0.028（要改善）、ε(pbg)≈+0.088（不一致）
- LOWZ（NGC）：ε(lcdm)≈−0.020（OK）、ε(pbg)≈+0.158（要改善）
- LOWZ（SGC）：ε(lcdm)≈−0.120（不一致）、ε(pbg)≈+0.170（不一致）

`output/cosmology/cosmology_bao_catalog_peakfit_caps_summary__recon_grid_iso.png`

Stage A では ε の誤差は粗い対角近似（paircount proxy）で保守化し、判定は |ε|<0.015（OK）/ 0.015–0.03（要改善）/ >0.03（不一致）とした（閾値はスコアボードと整合）。
一方、z-bin（CMASSLOWZTOT/combined）は pre-recon を Satpathy cov、post-recon（MW multigrid）を Ross cov で評価し、CI が「対角近似で過信」にならないようにしている。
個々の点の χ²/CI/z_eff/座標化仕様（`coordinate_spec_signature/hash`）などは対応する `*_metrics.json` に保存している。

また、ε という1指標だけでは ξ2 の符号反転や broadband のずれ等を見落とし得るため、補助的に
(i) μ-wedge（横/縦）からのピーク差（s⊥, s∥）による異方スコア、(ii) ξ2 の「特徴点」（broadband 除去後の支配極値）を併記して z依存でまとめた。
ここでの ε_proxy は ratio=s∥/s⊥（wedge peak; fid座標）から **ε_proxy=(1/ratio)^(1/3)−1** と定義し、peakfit の ε と符号を揃える（wedgeは Stage A のスクリーニング用）。

`output/cosmology/cosmology_bao_catalog_wedge_anisotropy.png`

`output/cosmology/cosmology_bao_catalog_wedge_anisotropy__recon_grid_iso.png`

`output/cosmology/cosmology_bao_catalog_peak_summary.png`

`output/cosmology/cosmology_bao_catalog_peak_summary__recon_grid_iso.png`

加えて、**重み付け仕様も固定**している。BOSS の標準重み（例：`w=WEIGHT_FKP×WEIGHT_SYSTOT×(WEIGHT_CP+WEIGHT_NOZ−1)`）を外すと、同じ銀河+random から得られる ξℓ と ε が系統的に動き得て、ε の差が「理論差」なのか「座標化（実装）差」なのかが混ざるためである。  
実例として、公開 multipoles に対する ε の比較を、BOSS標準 / FKPのみ / 重みなし（w=1）で並べた感度図を示す（Stage A の品質管理）。重みなしでは z-bin によって ε が大きく崩れる例があり、少なくとも Stage A では仕様を固定して議論する必要がある。

`output/cosmology/cosmology_bao_catalog_weight_scheme_sensitivity.png`

また、random カタログの行数（max_rows）も ε を揺らし得る。random を小さくすると RR/DR の統計誤差が増え、特に z-bin ごとのサブサンプルでは ε が不安定化（符号反転や過大推定）する例がある。  
Stage A（スクリーニング）の baseline では random 抽出仕様（kind/sampling/max_rows）を固定し、以後の ε 差が「理論差」ではなく「入力仕様差」に起因しないようにする（Stage B では全量/均一サンプルへ移行する）。

`output/cosmology/cosmology_bao_catalog_random_max_rows_sensitivity.png`

さらに、同じ BOSS DR12 に対して power spectrum multipoles P0, P2（Beutler et al. の公開データパッケージ）でも同様の最小モデル（smooth+peak）でクロスチェックし、AP の warping（ε）が大きく矛盾しないかを確認する。P(k) は survey 窓関数の影響が大きいため、RR multipoles（Beutler window）を用いて窓関数畳み込み込みで比較する（離散Hankelの有限範囲誤差を抑える補正も含む）。[Beutler2016PowspecData]

`output/cosmology/cosmology_bao_pk_multipole_peakfit_window.png`

`output/cosmology/cosmology_bao_pk_multipole_peakfit_pre_recon_window.png`

post-recon（窓関数込み）では、ε は z-bin1..3 で概ね **+0.008 / +0.020 / +0.040**（最大|z|≈1.71σ; z-bin3）となり、ξℓ から得た ε と同オーダーで大きな矛盾は見られない。  
一方、pre-recon の P(k) は quadrupole の broadband（RSD 等）が支配的になり得るため、ここでの最小モデル（smooth+peak）では ε が過大に推定される例がある（例：z-bin1 で ε≈+0.132±0.019）。したがって pre-recon P(k) は「RSDを含む full-shape の次工程」で扱う。

#### 2.5.2 宇宙論 I：距離指標と独立な検証

時間伸長（time dilation）そのものは距離指標と独立に観測できる。Type Ia 超新星のスペクトル年齢（aging rate）を用いた一次ソースでは、

$$ \\Delta t_{\\rm obs} = (1+z)^{p_t} \\Delta t_{\\rm em} $$

の指数として **`p_t=0.97±0.10 (1σ)`** が報告されており、`p_t=0`（時間伸長なし、tired light）を強く排除する。[Blondin2008TimeDilation]  
したがって、後述の DDR（距離二重性）の不整合を `p_t` の変更だけで埋める（例：`p_t≈3`）という説明は成り立たず、再接続は不透明度/標準光源/標準定規など距離指標側の前提を明示して検証する必要がある。

`output/cosmology/cosmology_sn_time_dilation_constraints.png`

同様に、CMB温度の赤方偏移依存は

$$ T(z)=T_0(1+z)^{1-\\beta_T} $$

としてパラメータ化でき、SZ効果などから距離指標と独立に制約できる。一次ソースでは **`β_T=0.004±0.016 (1σ)`**（`z≈3` まで）が報告されており、標準スケーリング（`T∝(1+z)`）と整合する。[Avgoustidis2011CMBTz]  
したがって `β_T`（同値に `p_T`）の変更だけで後述の DDR の不整合を埋める説明も成り立たない。

`output/cosmology/cosmology_cmb_temperature_scaling_constraints.png`

Alcock–Paczynski（AP）テストは、銀河分布の等方性から「横/縦の比」を測る幾何プローブであり、距離二重性とは独立な情報を与える。
一次ソースで一般的に用いられる無次元量として

$$ F_{\\rm AP}(z) \\equiv \\frac{D_M(z)\\,H(z)}{c} $$

（`D_M=(1+z)D_A`）を用いる。BOSS DR12 は `D_M×(r_d,fid/r_d)` と `H×(r_d/r_d,fid)` を与えるため、積で `r_d` が相殺され、`F_AP` は定規校正に依存せず計算できる。[Alam2017BOSS]  
ただし `F_AP` の算出は（`D_M` と `H` を経由する限り）BAO解析の前提と切り離せないため、ここでは「距離スケール（r_d）の校正に依存しない」という意味での距離指標非依存として扱う（BAO一次統計と前提は 2.5.1/2.5.3 で整理）。
静的背景P（指数 `P_bg`, 最小仮定）では静的近似 `D(z)=(c/H0)ln(1+z)` から

$$ F_{\\rm AP}^{(P)}(z) = (1+z)\\ln(1+z) $$

となり、BOSS DR12（`z=0.38,0.51,0.61`）の点とは概ね数σ以内で整合する（AP単独では決定的ではない）。

`output/cosmology/cosmology_alcock_paczynski_constraints.png`

加えて、赤方偏移 z 自体は距離指標と独立な一次観測量であり、スペクトル線（輝線/吸収線）の波長ズレから直接決まる。  
距離指標（ΛCDM距離など）の二次産物ではなく、**スペクトル一次データ**をローカルに保存して再現できる入口を作るため、JWST/MAST の x1d（1D spectrum）取得→キャッシュ→QC を導入した。

`output/cosmology/jwst_spectra__jades_gs_z14_0__x1d_qc.png`

さらに、半自動で「線候補→z候補」を抽出する最小パイプライン（線同定は手動検証が必要）を追加した。  
例として JADES-GS-z14-0 の x1d から得た z候補の診断図と候補表を示す。

`output/cosmology/jwst_spectra__jades_gs_z14_0__z_diagnostic.png`

`output/cosmology/jwst_spectra__jades_gs_z14_0__z_estimate.csv`

さらに次段階として、手動線同定（採用する線と窓幅を固定）に基づき、z と不確かさ（統計＋系統）を分離して出力する I/F（z確定）も用意した。

`data/cosmology/mast/jwst_spectra/jades_gs_z14_0/line_id.json`

`output/cosmology/jwst_spectra__jades_gs_z14_0__z_confirmed.json`

`output/cosmology/jwst_spectra__jades_gs_z14_0__z_confirmed.png`

`output/cosmology/jwst_spectra__jades_gs_z14_0__z_confirmed_summary.png`

JADES-GS-z14-0 の z確定は、`[O III]5007` と Hα を採用線（`use=true`）として固定し、`min_snr_integrated_proxy` により弱いスペクトル（例：o001）を除外してクロスチェックしている（例：z≈14.197）。

注意：対象によっては proprietary 期間により x1d が 401（Unauthorized）で取得できない場合がある（例：GN-z11）。この場合も manifest に「見つかったが取得不能」を記録し、議論がブレないようにする。

現状では、対象候補（GN-z11, JADES-GS-z14-0, JADES_10058975, RX_11027, LID-568, LID2619, LID4959, GLASS-z12, CEERS2-5429）に対して、取得状況（x1dの有無、QC、z推定/確定の可否）を
`data/cosmology/mast/jwst_spectra/manifest_all.json` に集約し、offline で再現できる形に固定している。
また、公開待ち（proprietary；release日が未来）の状態は、waitlist として `output/cosmology/jwst_spectra_release_waitlist.json` / `.csv` に固定している（再現：`python -B scripts/cosmology/jwst_spectra_release_waitlist.py`）。
2026-01-24（UTC）時点では JADES-GS-z14-0 / JADES_10058975 / RX_11027 / LID-568 / LID2619 / LID4959 / GLASS-z12 / CEERS2-5429 / GNZ7Q の local x1d が存在し、GN-z11 は `t_obs_release_utc=2026-04-09...`（未来）で未公開のため local x1d が無く z推定は `ok=false` となる（取得できない場合も含めて manifest に残す）。
また、ここでの z推定は「線候補→z候補」を列挙する入口であり、線同定と系統評価は手動検証が必要である。
したがって本稿では JWST x1d は「距離指標と独立な一次データ入口」として位置づけ、z確定（`z_confirmed`）が得られても **Table 1 の σ 判定には入れない**（Table 1 は P-model の検証指標を要約するため）。総合スコアボードでは参考（取得状況＋z確定の進捗）として扱う。

参考として、LID-568 は MAST 側の target_name が `LID568` / `LID568MIRI`（program 1760）であり、NIRSpec の単線 Hα として線同定を固定し、z確定（例：z≈4.4349）まで到達している（近接候補線 [N II]/[S II]6731 は raw では `no_signal_over_baseline`）。

`output/cosmology/jwst_spectra__lid_568__z_confirmed_summary.png`

また同じ program 1760 の公開ターゲットとして LID2619 / LID4959（MIRI/LRS P750L）も追加し、[S II] 二重線（6716/6731）を採用線として z確定（例：z≈19.58 / 19.66）まで到達している。ただし輝線が 13–14 µm 付近（LRS端・ノイズ増大域）に集中するため、z の扱いは暫定である（MIRI/LRS の centroid 系統は `sigma_sys_um=0.035 µm` として反映し、過小評価を避ける）。

`output/cosmology/jwst_spectra__lid2619__z_confirmed.png`

`output/cosmology/jwst_spectra__lid4959__z_confirmed.png`

また、program 6073（NIRSpec/SLIT）の公開ターゲットとして RX_11027 も追加し、`[O III]5007` と Hα の二線クロスチェックで z確定（例：z≈2.4216）まで到達している（z候補には高z代替解も出るため、採用根拠は `line_id.json` に固定して扱う）。

`output/cosmology/jwst_spectra__rx_11027__z_confirmed_summary.png`

また、GN-z11 が未公開（release日が未来）で保留のため、公開済みターゲットとして GLASS-z12（MIRI/LRS P750L; program 3703）を追加し、`[O III]5007`（6.66 µm）で z確定（例：z≈12.307）まで到達している。8.7–8.8 µm 付近は Hα+[N II] のブレンドでピークが複数立つため、cross-check として測定は残すが z結合には入れない（MIRI/LRS の分解能・ブレンド由来の centroid 系統は `sigma_sys_um` として反映し、過小評価を避けている）。

`output/cosmology/jwst_spectra__glass_z12__z_confirmed_summary.png`

加えて、同じ program 3703 の公開ターゲットとして CEERS2-5429（MIRI/LRS P750L）も追加し、z_estimate は z≈19.4（Hα + [S II]）を示唆する。輝線が 13–14 µm 付近（LRS端・ノイズ増大域）に集中し、Hα+[N II] / [S II] 二重線のブレンド比にも依存するため、z の扱いは暫定である。一方で `line_id.json` により採用線を固定し、`min_ok_spectra=2` を満たす z_confirmed（ok=true; z≈19.407; σ_total≈0.052）まで到達している。

`output/cosmology/jwst_spectra__ceers2_5429__z_confirmed_summary.png`

#### 2.5.3 宇宙論 II：距離指標依存と BAO の前提

次に、この仮定（宇宙膨張なし＋背景Pで赤方偏移）を「観測で使う量」に結びつける。  
観測では距離指標（光度距離 `D_L`、角径距離 `D_A`）や表面輝度が直接扱われるため、赤方偏移の起源が違うならスケーリングに差が出る。

代表例として、距離二重性（Etheringtonの関係）を

$$ \\eta(z) \\equiv \\frac{D_L}{(1+z)^2 D_A} $$

と定義すると、FRW（光子保存＋幾何学的膨張）では `η(z)=1` となる。  
一方で、膨張なしの静的幾何で「赤方偏移（光子エネルギー）＋時間伸長（到達率）」のみを入れる最小モデルでは
`D_L=(1+z)D_A` となり `η(z)=1/(1+z)` へずれる。  
同様に Tolman 表面輝度は FRW が `(1+z)^-4`、静的背景P（最小仮定）が `(1+z)^-2` となり、差分予測候補になる。

上の違いは、FRW が「フラックス減衰としての (1+z)^2（光子エネルギー×到着率）」に加えて、
空間膨張により `D_A=D_M/(1+z)`（放射時刻の物理スケールが縮む）を仮定している点にある。
P-model（背景P）では前者は条件A（p_t=1）と周波数比（1+z=P_em/P_obs）で維持する一方、
後者（空間膨張）を採用しないため `D_A^(P)=D`（静的幾何）となり、
結果として `D_L^(P)=(1+z)D_A^(P)` が出る（導出前提の台帳と監査は doc/cosmology に固定）。
この最小DDR関係の固定（式と変数定義）と、代表制約（例：ε0）への再解釈は
`output/cosmology/cosmology_ddr_pmodel_relation.json` と
`output/cosmology/cosmology_ddr_epsilon_reinterpretation.json` に固定した。

`output/cosmology/cosmology_observable_scalings.png`

しかし距離二重性は、すでに観測で強く制約されている。一次ソースで一般的に用いられる破れのパラメータ化として

$$ D_L(z) = (1+z)^{2+\\epsilon_0}\\,D_A(z) $$

（したがって）

$$ \\eta(z) = (1+z)^{\\epsilon_0} $$

を採ると、標準（FRW + 光子数保存）は `ε0=0` である。  
SNIa+BAO（現在データ）による制約として `ε0 = 0.013 ± 0.029 (1σ)` が報告されており、`ε0=0`（FRW）は 0.45σ 以内で整合する。[Martinelli2021DDR]  
一方で、膨張なしの静的幾何で「赤方偏移（光子エネルギー）＋時間伸長（到達率）」のみを入れる最小モデルは `D_L=(1+z)D_A`（`ε0=-1`）に対応し、Martinelli2021（SNIa+BAO）の `ε0` 制約と比較すると差は **約35σ** となる。  
これは「距離指標の標準的な前提（SNe Ia/BAO の定義・校正・系統の扱い）をそのまま採る限り」、静的最小が強い張力を持つことを意味する。  
ただし、この張力は (A) 静的背景Pの最小仮定の棄却（P-model の一部仮定の否定）、または (B) 距離指標の前提（不透明度/標準光源/標準定規）側の再検証、の両解釈が可能である。本稿では後者を定量化し、必要補正（α, s_L, s_R）として固定出力する。

一方で、BAO を含まないモデル非依存テストとして、クラスター（SZE+X-ray）の角径距離 `D_A` と SNe Ia の光度距離 `D_L` を直接比較し、`η(z)=1+η0 z` などで制約する研究もある。  
Holanda+2012 は（Constitution SNe Ia を用いた）直接比較で **`η0=-0.28±0.21 (1σ)`**（等温楕円βモデル×SNe Ia）を報告している。[Holanda2012ClustersDDR]  同一論文内でもクラスター側モデルを変えると、例えば non-isothermal spherical βmodel では **`η0=-0.39±0.11 (1σ)`** が得られる。[Holanda2012ClustersDDR]  また Li+2011 は Union2 を用いた同種の解析で、例えば De Filippis サンプル（`η=1+η0 z`）として **`η0=-0.12±0.17 (1σ)`** を報告している。[LiWuYu2011DDR]  
これらは本節の `ε0` パラメータ化（`η=(1+z)^{ε0}`）と同一ではないため、直感量として `z=1` での等価指数 `ε0≡ln(η(z=1))/ln2` に換算すると、Holanda+2012 は **`ε0≈-0.47±0.43`**（等温楕円β）または **`ε0≈-0.71±0.26`**（non-iso spherical β）などとなり、Li+2011（Union2）でも **`ε0≈-0.18±0.28`**（De Filippis, linear）等が得られる（換算は近似）。

さらに、強重力レンズ（SGL）を SNe Ia と（z>1.5 まで外挿するために）GRB 距離モジュラスと組み合わせて高赤方偏移まで拡張した解析もあり、Holanda+2016 は power-law `η(z)=(1+z)^{η0}` のパラメータ化で **`η0=-0.16^{+0.24}_{-0.51}`**（PLAW; 2σ）や **`η0=0.27^{+0.22}_{-0.38}`**（SIS; 2σ）を報告している（このパラメータ化では `ε0=η0` と読める）。[Holanda2016SGLDDR]  
また、SNe Ia と超コンパクト電波ソース（標準定規）を直接比較するモデル非依存テストとして、Li & Lin 2017 は `η_src(z)≡D_A(1+z)^2/D_L` を `η_src(z)=1+η0 z` 等でパラメータ化し **`η0=-0.06±0.05`**（1σ）などを報告している（本稿の `η≡D_L/((1+z)^2 D_A)` とは逆数なので換算に注意）。z=1 の等価指数としては **`ε0≈0.09±0.08`**（linear）や **`ε0≈0.14±0.13`**（non-linear）程度となる（換算は近似）。[LiLin2017RadioDDR]  
以上を含め、BAOなしの直接比較を本節の `ε0` へ換算して見た場合でも、静的最小（`ε0=-1`）の張力度（σ換算）は **おおむね 1〜14σ 程度**まで幅があり、DDRだけでの張力評価は「距離指標の構成」と系統（幾何モデル・パラメータ化）に強く依存する。

ただし、この張力は「観測で用いる距離指標 `D_L` と `D_A` が、標準光源（SNIa）と標準定規（BAO）から構成される」ことを前提にしている。  
P-model の立場で「宇宙は無限に広がる静的空間で、赤方偏移は背景Pに由来する」と採るなら、過去の物理スケール（光度・定規・有効不透明度など）が背景Pとともに変わる可能性を明示し、その分が `η` をどれだけ押し上げる必要があるかを一次ソースで詰める必要がある。  
SNIa+BAO の `ε0`（Martinelli2021）に合わせるには、有効に `η` を `(1+z)^{Δε}` だけ補正する機構が必要で、**Δε≈1.013±0.029（z=1で約2倍）**に相当する（“観測方法の前提がどれだけ効くか”の定量的目安）。
この補正は一意ではないが、例えば次のように解釈できる。

- 有効不透明度（光子数非保存）として吸収：`exp(+τ/2)=(1+z)^α`（したがって `τ(z)=2α ln(1+z)`）と置けば **α=Δε** が必要。
- 標準光源（SNIa）の光度進化として吸収：`L_em = L0 (1+z)^{s_L}` と置けば **s_L = -2Δε** が必要（z=1で距離モジュラスが約1.5等級ずれるのに相当）。
- 標準定規（BAO）の物理スケール進化として吸収：`l_em = l0 (1+z)^{s_R}` と置けば **s_R=Δε** が必要。

（いずれも「静的背景P最小」と現行の距離指標の間に必要となる補正量の目安であり、静的無限空間の宇宙論として成立させるには、距離指標の一次ソース（定義・校正）と整合する形で導出し直す必要がある。）

再接続条件を機械的に扱うため、本リポジトリでは、時間伸長の指数 p_t（SNe の light curve の幅）と、光子エネルギーの指数 p_e（CMB 温度スケーリング T(z) を proxy）を導入し、距離二重性の指数を

$$ \\epsilon_0^{\\rm model} = \\frac{p_e+p_t-s_L}{2} - 2 + s_R + \\alpha $$

と書く。観測 `ε0_obs±σ` に対して必要となる組合せは

$$ s_R + \\alpha - \\frac{s_L}{2} = \\epsilon_{0,\\rm obs} + 2 - \\frac{p_e+p_t}{2} $$

で与えられ、右辺は（`p_t=p_e=1` なら）`ε0_obs+1` に一致する。各機構（α, s_L, s_R）について、独立制約に対する z スコア `z=(x_required-\\mu)/\\sigma` を算出し、目安として **|z|<3 を ok、3≤|z|<5 を mixed、|z|≥5 を ng** と分類する（判定基準と全行の結果は `output/cosmology/cosmology_ddr_reconnection_conditions_metrics.json` に固定）。

また Martinelli2021（SNe Ia + BAO）の ε0 制約は比 `D_L/D_A`（または `η`）を拘束するため、張力は “d_L 側（光度距離）” と “d_A 側（角径距離）” のどちらに寄せても説明できる（非一意）。直感のため `z=1` を参照点にとると、静的最小（ε0=-1）から ε0≈0 へ寄せるには `η(z=1)` を **約2.02倍**する必要がある。これは例えば、全てを d_L 側で吸収するなら `D_L(z=1)` を **約2.02倍**（距離モジュラスで **約1.52等級**、フラックスで **約4.07倍の減光**）に相当し、全てを d_A 側で吸収するなら `D_A(z=1)` を **約0.50倍**に相当する（対称分割なら `D_L≈1.42倍` かつ `D_A≈0.70倍`）。この寄与分解も上の metrics に固定した。

`output/cosmology/cosmology_distance_duality_constraints.png`

`output/cosmology/cosmology_ddr_reconnection_conditions.png`

上の Δε を、距離モジュラス補正 |Δμ(z)| と、灰色不透明度（光子数非保存）の等価減光 |τ(z)| に変換し、赤方偏移とともにどの程度増大するか（どのzまで距離指標の“誤差予算”に隠れ得るか）を整理した。  
代表例として、(i) BAOを含む制約の中で最小σ（tightest）と、(ii) BAOなし制約の中で |z_pbg_static| が最小（least rejecting）を併記する。

`output/cosmology/cosmology_distance_indicator_reach_limit.png`

さらに、距離指標（SNe Ia / BAO）の観測不確かさ（誤差予算）を一次データから z 依存で見積もり、上の必要補正 |Δμ(z)| が「観測の不確かさに隠れ得る範囲」を可視化した。  
Pantheon は lcparam の `dmb`（統計）を距離モジュラス誤差の代理として、加えて公開の系統共分散（sys covariance）を上限側の目安として併用する。BAO は BOSS DR12 の `D_M` の相対誤差を等価Δμへ変換して用い、さらに Table 8（BAO+FS consensus）の reduced covariance（c_ij）から `D_M` 間の相関を取り出して、3点（z=0.38/0.51/0.61）の2点線形補間に対する誤差伝播（目安）も併記した。[Scolnic2018Pantheon] [Alam2017BOSS] [Planck2018Params]

`output/cosmology/cosmology_distance_indicator_error_budget.png`

誤差予算の推定は、SNeサンプルや共分散の扱い（diag-only か full covariance か）でも数割程度動き得る。  
Pantheon（lcparam）と Pantheon+（距離＋共分散）を用いて、`z_limit`（必要補正が誤差予算に隠れ得る上限z）がどれだけ変動するかの感度を併記した。さらに Pantheon+ では、CID重複の扱い（同一SNの複数測光）と、SH0ES HF（Hubble Flow; `USED_IN_SH0ES_HF==1`）限定の感度も併記した。[Brout2022PantheonPlus]
この推定（opt_total の envelope）では、`z_limit`（3σ）は BAO含む代表で z≈0.048、BAOなし代表でも最楽観（bin幅0.05, n≥8）で z≈0.201 程度に留まる。

`output/cosmology/cosmology_distance_indicator_error_budget_sensitivity.png`

さらに、どの組合せ（不透明度 α / 光源進化 s_L / 定規進化 s_R）で ε0 を回復できるかを、パラメータ空間として可視化した。

`output/cosmology/cosmology_reconnection_parameter_space.png`

さらに、Δε を (i) 不透明度 α、(ii) 標準光源進化 s_L、(iii) 標準定規（BAO）スケール の **どれか1つだけ**で説明すると仮定した場合に必要となる量を、一次ソース拘束と比較した。
結果として、静的背景P最小（ε0=-1）から現行制約（ε0≈0）へ寄せるために必要な **α≈1.013** は、SNIa+H(z) による不透明度解釈の拘束（α=-0.04±0.0375）から **約28σ** 外れる。[Avgoustidis2010Opacity]  また、モデル依存の統合解析（SNIa+BAO+CMB+H(z)）でも **ε=0.023±0.018 (68%CL)** が報告されており（α=ε）、そこからは **約55σ** 外れる。[LvXia2016CDDR]  
同様に、標準光源進化だけで説明するには **s_L≈-2.026**（z=1で約1.5等級の系統）が必要で、SNe Ia の進化拘束として Ferramacho+2009 の time-law（z=1での等級変化 K）を s_L に換算した値 **s_L=0.056±0.056** から **約37σ** 外れる。[Ferramacho2009SNEvo]  Linden+2009 の log(1+z) モデルでも **s_L=0.22±0.35** とされ、そこからも **約6σ** 外れる。[Linden2009SNEvo]  
ただし、標準化後の明るさ（HR）がホスト銀河の population age と相関するという報告もあり、z=0→1 の典型的な progenitor age 差（約5.3 Gyr）を当てはめると、z=1 の ΔM は 0.1〜0.27 mag 程度（s_L に換算すると -0.15〜-0.36）となる。[Kang2020SNEvoAgeHR]  
標準定規だけで説明するには、z=1で **r_drag が約2倍**（147→約297 Mpc）が必要になり、標準宇宙論の校正スケール（r_drag=147.09±0.26）とは整合しない（この比較は “BAOを標準定規として採る” 場合の目安）。[Planck2018Params]  
なお、同じ Δε_needed は d_L/d_A の比（η≡d_L/((1+z)^2 d_A)）の補正であり、例えば代表（Martinelli2021）の Δε≈1.013 は z=1 で η を **約2.02倍**する必要がある。これは極端には d_L 側の系統として d_L×2.02（Δμ≈1.53 mag）と解釈でき、逆に d_A 側の系統として d_A×0.496 と解釈することもできる（分配は一意でない）。
したがって「単独機構」での再接続は現実的でなく、もし静的無限空間として成立させるなら、距離指標の定義（標準化/校正）を P-model 側で再導出した上で、複数要因の組合せが独立プローブ（SN time dilation, T(z), Alcock-Paczynski など）と矛盾しないことが条件になる。

なお、SN time dilation と CMB T(z) は距離指標と独立な一次ソースであり、背景Pの最小予測（`p_t=1`, `p_T=1`）とは整合する。一方で、tired-light（`p_t=0`）や no-scaling（`p_T=0`）は強く棄却されるため、DDR の不整合を「時間伸長なし」などで埋める説明は成り立たない（距離指標側の再導出が必須）。

`output/cosmology/cosmology_distance_indicator_rederivation_requirements.png`

`output/cosmology/cosmology_distance_indicator_rederivation_candidate_search.png`

この図では、(i) 独立プローブ固定なし（best_any）、(ii) 独立プローブ（p_t, p_e）固定（best_independent）に加え、(iii) GW標準サイレン由来の不透明度 α 拘束を用いた参考系列を併記する（観測: GW170817×NGC4993 の近傍整合 → αは実質未拘束、将来予測: Pantheon+ET）。[Abbott2017GW170817] [Hjorth2017NGC4993Distance] [Cantiello2018NGC4993SBF] [QiCaoPanLi2019GWOpacity]
代表（BAO含むDDR: Martinelli2021）の best_independent では、θ≈{s_R=0.21, α=0.39, s_L=-0.69, p_t=1.11, p_e=1.00} で **max|z|≈2.57σ（limiting=BAO s_R）**となり、BAO自己整合が支配的になる。  
一方、代表（BAOなしDDR: Holanda2012 non-iso spherical）は θ≈{s_R=-0.032, α=0.233, s_L=-0.190, p_t=0.971, p_e=0.996} で **max|z|≈0.03σ** と非常に小さく、“どのDDR一次ソースを採るか” が結論を強く左右する。

この “limiting（支配拘束）” がどれに偏るかを、DDR一次ソースごとに集計した（色=支配拘束、斜線=best_independent）。

`output/cosmology/cosmology_distance_indicator_rederivation_limiting_summary.png`

さらに、candidate_search は DDR一次ソースごとに α と s_L の一次ソース候補を入れ替えて「最小の max|z|」を選ぶため、選択される一次ソースには偏りが生じ得る。探索結果では、best_independent の多くで特定の（相対的に緩い）一次ソース拘束が繰り返し選ばれる傾向がある。これは「合致の可否」が、距離指標の定義/校正や系統（一次ソースの採り方）に強く依存し得ることを意味する。

`output/cosmology/cosmology_distance_indicator_rederivation_selected_sources.png`

最後に、candidate_search（scan）は「候補の中から最小の max|z| を選ぶ」ため、結果が “緩い一次ソース” に寄りやすい。そこで、independent-only の中で最小σ（tightest）の α と s_L を固定した場合（tightest_fixed）と、scan（best_independent）を並べて感度を示した。

`output/cosmology/cosmology_distance_indicator_rederivation_policy_sensitivity.png`

さらに、candidate_search は DDR一次ソースごとに最適な一次ソース拘束（α, s_L）を選び得るが、物理的には α と s_L は単一の prior（少なくとも整合する範囲）であるべきである。そこで、α と s_L の一次ソースを **1つずつ固定**し、全DDR行について「最悪の max|z|」が最小になる組合せを探索した。  
結果として、全候補（BAO/CMB/CDDR を含む）から選んでも全DDRの worst max|z| は **約2.4σ**、独立一次ソースに限定しても **約2.5σ** 程度に留まり、f=1（BAO σそのまま）では「全DDR行を 1σ 同時整合」に揃えるのは難しい（ただし全DDR行を 3σ 以内に収める組合せは存在する）。

`output/cosmology/cosmology_distance_indicator_rederivation_global_prior_search.png`

さらに、単一prior（α,s_L）を保ったまま BAO(s_R) の prior を soft constraint として緩めた場合（σ→fσ）、
全DDR行の worst max|z| は f の増加とともに低下し、**1σ同時整合**に入るには **f≈7.8**（スキャンでは **f=8** で worst≈0.98）が必要になる。
したがって、静的背景Pで距離指標の整合を得るには「BAO(s_R) の自己整合」を大きく緩める（あるいは BAO の一次ソースそのものを再評価する）必要がある。

`output/cosmology/cosmology_distance_indicator_rederivation_global_prior_search_bao_sigma_scan.png`

`output/cosmology/cosmology_distance_indicator_rederivation_candidate_matrix.png`

上の“再導出要件”が、一次ソース拘束と同時に成立し得るかを定量的に確認するため、
DDR（ε0）ごとに不透明度 α と標準光源進化 s_L の一次ソース候補を入れ替えつつ、
独立プローブとして p_t（SN time dilation）と p_e（CMB T(z)）を固定し、
さらに BAO（BOSS DR12 の (D_M,H)）から fit した s_R と同時に整合するかを WLS で探索した（目的：最大|z|を最小化）。
その結果、距離指標の前提（α, s_L）の“採り方”で整合性は大きく動き得るが、SNIa+BAO の tight な DDR 制約を満たしつつ BAO(s_R) も保つには、
一般に（z≳1で）大きめの s_L と/または α を要求し、一次ソースの選択依存が強いことが分かる。

さらに、一次ソース拘束（不透明度 α、標準光源進化 s_L）を受け入れたまま DDR を満たすように解くと、必要な標準定規進化 `s_R` は **`ε0_obs` の取り方**で大きく変わる。SNIa+BAO の `ε0` を採ると **`s_R≈1.016±0.061 (1σ)`** となる（LvXia2016CDDR の α、Dainotti+2021 の `H0(z)=H0/(1+z)^α` を `s_L=2α` として「標準光源の見かけの進化」に換算した s_L、Blondin+2008 の `p_t`、Avgoustidis+2011 の `p_T` を使用）。[Blondin2008TimeDilation] [Avgoustidis2011CMBTz] [Dainotti2021PantheonH0Evo]  これは z=1 で標準定規が **約2.02倍**（中央値、±1σで約1.94〜2.11倍）に相当する。  
一方で、BAO を含まないクラスター+SNe Ia の直接比較（z=1換算）を採ると、Holanda2012 は **`s_R≈0.309±0.270 (1σ)`**（non-iso spherical β）または **`s_R≈0.548±0.438 (1σ)`**（等温楕円β）などが得られ、Li2011（Union2, linear）では **`s_R≈0.607±0.203 (1σ)`**（Bonamente）や **`s_R≈0.837±0.289 (1σ)`**（De Filippis）となる。必要な定規進化の“中心値”は小さくなり不確かさも大きい（したがって BAO との張力は弱まり得る）。

さらに、α と s_L の一次ソースも “BAO/CMB を含むか”で独立性が変わる。ここでは DDR検証との循環性を減らすため、s_L について CDDR仮定の一次ソース（Hu+2023 / Kumar+2021）を primary から除外し、残りの中で（現状で最小σの）α: LvXia2016（SNIa+BAO+CMB+H(z)）と s_L: Dainotti+2021（Pantheon の H0(z) fit を「標準光源の見かけの進化」へ換算）を採った場合である。[Dainotti2021PantheonH0Evo]  
BAO/CMB を用いない一次ソースに限定して α と s_L を選び直すと（α: Avgoustidis+2010 / Holanda+2014 / Wang+2017 / Liao+2015、s_L: Dainotti+2021（Pantheon の H0(z) fit を換算））、例えば Martinelli2021（SNIa+BAO）の `ε0` を採る場合でも **`s_R≈1.079±0.069 (1σ)`** となる（z=1で標準定規が約2.01〜2.22倍）。ただし Dainotti+2021 の換算は「標準光源そのものの進化」と「距離スケール（H0）側の系統」を区別できない（退化）ため、拘束としては“見かけの上限”に近い点に注意が必要である。[Avgoustidis2010Opacity] [HolandaCarvalhoAlcaniz2014Opacity] [WangWeiLiXiaZhu2017CurvatureOpacity] [LiaoAvgoustidisLi2015Transparency] [Dainotti2021PantheonH0Evo]

`output/cosmology/cosmology_reconnection_required_ruler_evolution.png`

`output/cosmology/cosmology_reconnection_required_ruler_evolution_independent.png`

`output/cosmology/cosmology_reconnection_plausibility.png`

さらに、BOSS DR12 の一次ソース出力（`D_M×(r_d,fid/r_d)`, `H×(r_d/r_d,fid)`）を用い、静的背景P（指数）+ 有効標準定規進化 `r_d(z)∝(1+z)^{s_R}` の形で (D_M,H) を同時fitすると、`s_R` は **`s_R≈-0.033±0.094 (1σ)`** を好む。[Alam2017BOSS]  
DDR側から必要となる `s_R` との張力は、採る `ε0_obs` に強く依存する。例えば SNIa+BAO の `ε0` を採ると **約9.4σ**（ただし `ε0` 自体が BAO 距離に依存し得るため厳密に独立ではない）。一方、BAO を含まないクラスター+SNe Ia の直接比較（Holanda2012 の non-iso spherical を z=1換算）を採ると **約1.3σ**、Li2011（Union2, linear）では **約3.0σ** 程度となり、BAO との自己整合は“成立し得る”水準まで緩み得る（パラメータ化によっては 4σ級まで変動）。

独立一次ソース版（α と s_L を BAO/CMB 非依存の一次ソースに限定）の場合でも、SNIa+BAO の `ε0` を採ると BAO 自己整合（BOSS DR12）との張力は **約9.5σ** となる。一方、クラスター+SNe Ia（Holanda2012 の non-iso spherical, z=1換算）では **約1.3σ** 程度、Li2011（Union2, linear Bonamente）では **約3.1σ** 程度であり、傾向は大きく変わらない（ただし距離指標依存の系統で 4σ級まで変動し得る）。[Avgoustidis2010Opacity] [LiaoAvgoustidisLi2015Transparency]

`output/cosmology/cosmology_bao_scaled_distance_fit.png`

`output/cosmology/cosmology_bao_scaled_distance_fit_independent.png`

さらに、BAO(s_R) の推定について Table 8 の reduced covariance（cross-z相関）を用いた full 共分散や、diag-only / block（ρ(D_M,H)のみ）の扱いを切り替えて感度を確認した。結果は `s_R` の中心値が -0.04 付近、1σが 0.085〜0.106 程度の範囲に留まり、距離指標の再導出候補探索（best_independent）では BAO含むDDR（Martinelli2021）の limiting は BAO s_R のままである。BAOを 1σ に収めるには、`s_R` の不確かさを概ね **約2.5倍** に膨張させる必要がある（目安）。

`output/cosmology/cosmology_bao_scaled_distance_fit_sensitivity.png`

さらに、距離指標の再導出候補探索（WLS; DDR + BAO(s_R) + α + s_L + p_t + p_e の同時整合）側で、BAO(s_R) を soft constraint として扱う感度を確認するため、BAOの `s_R` の不確かさ（σ）を f 倍（`σ→fσ`）するスキャンを行った。代表（BAO含むDDR: Martinelli2021）の best_independent の max|z| は、f=1 で **約2.57σ**、f=6 で **約1.10σ**、f=10 で **約0.68σ** まで低下するが、limiting は BAO s_R のままであり、この枠組みでは BAO が支配的な拘束であることが分かる。一方、BAOなしDDR（例：Holanda2012 の non-iso spherical）は元々 max|z| が十分小さく（≪1σ）、“どのDDR一次ソースを採るか”が結論を強く左右する。

`output/cosmology/cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan.png`

したがって「代表（BAO含むDDR）が 1σ を満たす」ためには、単純な線形補間でも **f≈6.80**（σを約6.8倍）程度まで BAO(s_R) を緩める必要がある（現状の系列では支配拘束は最後まで BAO s_R）。

上の代表例を一般化するため、全DDR一次ソース（16行）について同じ BAO σスケール f の系列で best_independent の max|z| を再計算し、行列として整理した。  
このスキャン範囲（f≤10）では全行で **max|z|≤1** に到達するが、そのために必要な f の中央値は **約5**、最大は **約8** 程度である（DDR σは `category_sys` を使用）。

`output/cosmology/cosmology_distance_indicator_rederivation_bao_relax_thresholds.png`

さらに、BAO(s_R) の prior そのものを変えた場合の影響を見るため、BOSS DR12 の共分散モード（block/diag/full など）で得られる (s_R, σ) を BAO prior として candidate_search（best_independent）を全DDR行で再計算した。  
共分散モード差だけでは改善は限定的で、block/diag/full では **16行中4行**が max|z|≤1 に入るに留まる。一方、H のみを用いた場合（最も緩い側の感度）では **16行中8行**が max|z|≤1 に入り、中央値も **約1.0** まで下がるが、それでも残りは 1σ を超える。したがって「全DDR行を 1σ 同時整合」に収めるには、共分散モード差の範囲を超えた追加の緩和（f の増大）が必要になる。

`output/cosmology/cosmology_distance_indicator_rederivation_bao_mode_sensitivity.png`

さらに、単一prior（α,s_L）を保ったまま BAO(s_R) を緩める場合（σ→fσ）でも、BAO fit の前提（共分散モード）によって必要な f は動く。
本スキャンでは、block/diag/full では **f≈7.1〜8.8** 程度である一方、Hのみ（最も緩い側の感度）では **f≈2.9** 程度まで下がる。
したがって、距離指標の再導出が成立するかの判断は、BAOの一次ソース・共分散・解析前提の保守的な固定と切り分けて評価する必要がある。

`output/cosmology/cosmology_distance_indicator_rederivation_bao_mode_global_prior_sigma_scan.png`

さらに、BAO(s_R) の推定が BOSS DR12 のみの解析に依存しないかを確認するため、異方的BAOの別系統（eBOSS DR16：LRG z=0.698、QSO z=1.48、Lyα z=2.33）に加え、DESI DR1（LRG z=0.510/0.706、LRG+ELG z=0.930、ELG z=1.317、Lyα z=2.33）の距離比（D_M/r_d, D_H/r_d）も同一形式で取り込み、同じ静的背景P + 定規進化（r_d(z)∝(1+z)^{s_R}）モデルを fit した。各点の D_M と D_H の相関（一次ソースの共分散/ρ）も反映した上で、eBOSSのみでは **s_R≈0.138±0.026**、BOSSのみ（比へ変換）は **s_R≈-0.041±0.097**、DESIのみでは **s_R≈0.106±0.018**、併合（BOSS+eBOSS+DESI）では **s_R≈0.089±0.012** となり、単一の s_R で全系統を同時に説明するには張力（併合で χ²/dof≈3.66）が残る。[Alam2017BOSS] [Bautista2020eBOSSLRG] [Neveux2020eBOSSQSO] [duMasdesBourboux2020eBOSSLya]

`output/cosmology/cosmology_bao_distance_ratio_fit.png`

併合fit の点ごとの残差（共分散を含む χ²寄与）を √χ² の形で可視化すると、DESI Lyα（z=2.33）や DESI LRG+ELG（z=0.93）などが張力を支配し、BOSS（z=0.38）や eBOSS QSO（z=1.48）が続くことが分かる。
したがって、BAO多系統の整合性を厳密に議論するには、特定の点（系統）の扱いと共分散前提を含めた一次ソース側の検証が不可欠である。

`output/cosmology/cosmology_bao_distance_ratio_fit_residuals.png`

さらに、併合fit から「どの点が結論（s_R）や張力（χ²/dof）を支配するか」を確認するため、leave-one-out（点を1つ落として再fit）を行った。
その結果、DESI Lyα（z=2.33）を落とすと χ²/dof は **≈2.78** まで下がる一方、s_R は **0.063±0.015** 程度へシフトし、
高z点が s_R と張力の両方に強く影響することが分かる（eBOSS Lyα（z=2.33）を落とすと χ²/dof は **≈3.90**、s_R は **0.078±0.014** 程度）。
したがって、BAOの一次ソース差（BOSS/eBOSS）の議論では「特定点の系統」だけでなく「高z点による拘束の強さ」を分けて扱う必要がある。

`output/cosmology/cosmology_bao_distance_ratio_fit_leave_one_out.png`

さらに、この BAO の一次ソース差（BOSSのみ / eBOSSのみ / 併合）が、距離指標の再導出（単一prior）に与える影響を確認するため、
BAO(s_R) の prior を切り替えた上で、全DDR行を 1σ同時整合に入れるために必要な緩和量 f（σ→fσ）を推定した。
結果として、BOSSのみでは **f≈7.7** と baseline（BOSS DR12 の D_M,H fit）と同程度である一方、eBOSSのみでは **f≈21**、DESIのみでは **f≈33**、併合（BOSS+eBOSS+DESI）では **f≈52** 程度まで増加する。
したがって、高zを含む BAO の tight な s_R をそのまま採る限り、「単一priorで距離指標の整合を得る」ための緩和量は大きくなり、再接続はより困難になる。
ただし、BOSS と eBOSS の s_R の食い違い（多系統差）そのものを「系統幅 σ_sys」とみなし、
`σ_total^2 = σ_stat^2 + σ_sys^2`（`σ_sys = 0.5*max|s_R(subset_i)-s_R(subset_j)|`；BOSS/eBOSS/DESI の多系統差）で BAO prior を緩めると、併合でも必要 f は **f≈6.7** 程度まで下がり、BOSS基準と同程度のスケールになる。
また別の診断として、併合fit の leave-one-out による最大シフト Δs_R の半分を `σ_sys` とみなす（`σ_sys = 0.5*max|s_R(leave-one-out)-s_R(full)|`）と、必要 f は **f≈34** 程度まで下がる。
このとき「f≈52」という大きな値は、tight な併合prior（σ_stat）をそのまま採った場合に特有の結果であり、BAO の一次ソース差（多系統）をどう保守的に扱うかが結論を強く左右する。

`output/cosmology/cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan.png`

さらに、併合fit の leave-one-out で得られた (s_R,σ) を BAO prior として同様に f（σ→fσ）を推定すると、BOSS z=0.38 や QSO z=1.48 などを除外しても必要 f は **約47〜51** と大きいままに留まる一方、Lyα（z=2.33）を除外すると（σが大きくなるため）必要 f は **f≈42〜45** 程度まで下がる。
したがって、単一priorで全DDR行を 1σ同時整合に入れるという要件は、併合BAO prior の tight さ（とくに高z点の拘束）に強く支配され、単発の「支配点」除外だけでは根本的には解消しない。

`output/cosmology/cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan.png`

Tolman 表面輝度（SB dimming）の一次ソースとして、Lubin & Sandage は早期型銀河について有効指数 `n`（`SB(z)/SB(0) ∝ (1+z)^-n`）を
**R band: n=2.59±0.17**、**I band: n=3.37±0.13** と報告している。[LubinSandage2001Tolman]  
純粋な Tolman 効果（FRW: `n=4`）との差は「過去ほど明るい」光度進化で説明できる一方、静的背景P（最小仮定: `n=2`）に合わせるには
しばしば進化補正の符号が逆（過去ほど暗い側）になり、整合しにくい。  
ただし、この種の議論は銀河進化（系統）に依存するため、ここでは DDR ほど強い棄却根拠とはせず、**差の符号/スケール**の補助情報として扱う。

`output/cosmology/cosmology_tolman_surface_brightness_constraints.png`

上で扱った宇宙論の主要プローブ（DDR / Tolman / AP / BAO / T(z) / SN time dilation）について、静的無限空間の“最小仮定”に対する棄却度と、距離指標依存/系統の注意点を 1枚にまとめた（例外規定の整理）。

`output/cosmology/cosmology_static_infinite_hypothesis_pack.png`

上の整理を踏まえると、現状の張力は主に **距離指標（標準光源/標準定規）に強く依存するプローブ（DDR/BAO/Tolman）** に集中し、
距離指標と独立な一次ソース（SN time dilation / CMB T(z)）は背景Pの最小予測と整合している。
したがって本稿では、(i) DDR の一次ソース依存（BAOの有無、クラスター系統、校正）と、(ii) BAO の多系統差（BOSS/eBOSS、共分散/解析）と、(iii) Tolman の進化補正の符号/スケールを、同一枠組みで切り分けることが決定的である。

`output/cosmology/cosmology_tension_attribution.png`
<!-- INTERNAL_ONLY_START -->
（内部）入力固定：`data/cosmology/distance_duality_constraints.json`
（内部）集計値（固定）：`output/cosmology/cosmology_distance_duality_constraints_metrics.json`
<!-- INTERNAL_ONLY_END -->

### 2.6 動的P（放射・波動）

2.1〜2.4節は主に **静的な時間波密度** `P(x)` を前提に、重力・時計・光伝播の弱場テストを扱った。  
一方で、連星系の軌道減衰や重力波の chirp は「放射（エネルギーが系外へ出る）」現象を含むため、時間波密度を時空関数 `P(x,t)` として扱う必要がある。

ここでは最小の拡張として、以下の2点だけを **作業仮定**として導入する。

1) 変数の選び方：  
`u(x,t) ≡ ln(P/P0)`（すなわち `u = -φ/c^2`）を基本変数とする。

2) 動力学（静的極限との接続）：  
静的極限で `∇^2φ = 4πGρ`（ニュートン極限）を回収するように、`u` の方程式を

$$
\\left(-\\frac{1}{c^2}\\frac{\\partial^2}{\\partial t^2}+\\nabla^2\\right)u
= -\\frac{4\\pi G}{c^2}\\,\\rho
$$

（同値に `φ=-c^2u` から
$$
\\left(-\\frac{1}{c^2}\\frac{\\partial^2}{\\partial t^2}+\\nabla^2\\right)\\phi
= 4\\pi G\\,\\rho
$$
）
と書く。ここで `□` は d’Alembert 演算子であり `□≡-(1/c^2)∂^2/∂t^2+∇^2`、`ρ` は弱場での有効な質量密度（ソース）である。

このとき、遠方場（ソースが無い領域）では `□u=0` の波動方程式となり、`u` は速度 `c` で伝播する。  
放射の多極展開を考えると、以下が重要になる。

まず、上の波動方程式の遅延解（retarded solution）は Green 関数を用いて

$$
u(t,\\mathbf{x})=\\frac{G}{c^2}\\int d^3x'\\,\\frac{\\rho(t_{\\rm ret},\\mathbf{x}')}{|\\mathbf{x}-\\mathbf{x}'|}
,\\quad
t_{\\rm ret}=t-\\frac{|\\mathbf{x}-\\mathbf{x}'|}{c}
$$

と書ける。系の大きさを `R`、観測点までの距離を `r=|x|` とし、遠方場 `r≫R` で `1/|x-x'|` と `t_ret` を展開すると、`u` の `1/r` 項はソースの多極モーメント（単極・双極・四重極…）で表される。ここで

$$
M\\equiv \\int d^3x\\,\\rho,\\quad
D_i\\equiv \\int d^3x\\,\\rho x_i,\\quad
Q_{ij}\\equiv \\int d^3x\\,\\rho\\left(x_i x_j-\\frac{1}{3}\\delta_{ij} x_k x_k\\right)
$$

を質量の単極・双極・（トレースを除いた）四重極とする。

孤立系では質量保存により `dM/dt=0` が成り立つため、単極の時間変化は放射を生まない。さらに運動量保存から

$$
\\frac{d^2 D_i}{dt^2}=0
$$

となり、重心系では `D_i` は定数（通常 `0` に選べる）である。したがって、遠方で時間的に変動し得る最初の多極は四重極であり、放射の主項は **四重極**になる。

この点は二重パルサーが与える強い制約（特に NS–WD 系で顕在化する双極放射の上限）と整合するために、P-model が満たすべき最小要件でもある。すなわち、P-model では「全ての物質が同一の `P` に普遍的に結合し、天体の種類によって“荷”が変わらない」ことを仮定し、双極放射が出ない状況を採用する（双極を許す理論は観測で即棄却されやすい）。[Will2014][WeisbergHuang2016][Kramer2021][Freire2012J1738][Antoniadis2013J0348]

四重極放射のエネルギー流束は（係数まで含めた標準形として）

$$ \dot{E}_{\rm rad} = -\frac{G}{5c^5}\left\langle \dddot{Q}_{ij}\dddot{Q}_{ij}\right\rangle $$

で与えられる（`Q_ij` は質量四重極モーメント、上付きドットは時間微分）。[PetersMathews1963][Peters1964]  
この結果から、連星の軌道周期 `P_b` の変化（円軌道〜離心軌道）として、いわゆる Peters–Mathews の式

$$
\dot{P}_b^{\rm (quad)}
= -\frac{192\pi}{5}\left(\frac{2\pi G M_c}{c^3 P_b}\right)^{5/3}
\frac{1+\frac{73}{24}e^2+\frac{37}{96}e^4}{(1-e^2)^{7/2}}
$$

（`M_c` は chirp 質量、`e` は離心率）が得られる。[PetersMathews1963][Peters1964]

同様に、重力波の周波数 `f(t)` は四重極放射のチャープ則

$$
\frac{df}{dt}
= \frac{96}{5}\pi^{8/3}\left(\frac{G M_c}{c^3}\right)^{5/3} f^{11/3}
$$

に従い、積分すると

$$
t_c-t
= \frac{5}{256}\left(\frac{G M_c}{c^3}\right)^{-5/3}(\pi f)^{-8/3}
$$

の形が現れる（`t_c` は合体時刻）。[PetersMathews1963][Peters1964]

四重極則が与える「スケール感」の目安（軌道減衰と chirp の典型レンジ）を次図に示す。

`output/theory/dynamic_p_quadrupole_scalings.png`

本稿の段階では、P-model の動的方程式から「強場での波形」を完全に計算することは主張しない。  
ただし、P-model が相対論の置換候補であるためには、この弱場極限（四重極放射の支配）と整合する必要がある。  
以降の 4.10 節（二重パルサー）と 4.11 節（GW150914）では、上の観測量（`Pdot_b` と chirp）を用いて、現状の一次ソースに基づく整合性を示す。

### 2.7 核力（近接場での波動干渉）

2.1〜2.6節では、主に遠距離場（重力・放射）を `P(x)` と `P(x,t)` の写像で扱った。  
核力スケールでは、同じ写像をそのまま「静的ポテンシャルの遠距離近似」として使うだけでは不十分で、  
`r` が局所モードの代表長（`λ_c`）に近づく近接場で、位相干渉を明示的に扱う必要がある。

ここでは最小仮定として、スケール分岐を次で固定する。

- `r≫λ_c`：位相平均化が有効で、遠距離場の有効記述（重力）を使う。  
- `r≲λ_c`：位相干渉が残り、`Ψ_total=Ψ1+Ψ2` の重ね合わせを局所近接場の入口として使う。

この近接場の入口では、単独核子モードを

$$
\\Psi=A\\exp(i\\omega t-ikr)
$$

と書き、2体系では同相/逆相モードの分裂（`ω_bound` と `ω_free`）を束縛の最小表現として扱う。  
最小の定式化として、相対座標 `r` の動径方程式を

$$
\left[-\frac{\hbar^2}{2\mu}\frac{d^2}{dr^2}+U_{\rm eff}(r)\right]u_\ell(r)=E\,u_\ell(r),
\quad u_\ell(r)\equiv rR_\ell(r)
$$

で固定する（`μ` は換算質量）。  
ここで、束縛状態は `E<0`、散乱状態は `E>=0` とし、  
`s` 波の最小モデル（例：square-well）では、境界条件から

$$
k\cot(kR)=-\kappa,\quad
k=\sqrt{2\mu(V_0-E_B)}/\hbar,\quad
\kappa=\sqrt{2\mu E_B}/\hbar
$$

の形が得られる（`E_B>0` は結合エネルギー）。  
解析解を持たない場合は、この固有値条件を数値的に解いて `ω_bound` を定義する。

周波数差は

$$
\Delta\omega\equiv \omega_{\rm free}-\omega_{\rm bound}
=\frac{E_{\rm free}-E_{\rm bound}}{\hbar}
$$

と定義する。閾値基準 `E_{\rm free}=0` をとると  
`Δω=E_B/ħ` となり、結合エネルギーと質量欠損は

$$
B.E.\equiv E_B=\hbar\Delta\omega,\qquad
\Delta m=\frac{E_B}{c^2}=\frac{\hbar\Delta\omega}{c^2}
$$

で結ばれる。代表的な自由モード周波数 `ω_{\rm free}` を基準にすれば、

$$
\frac{\Delta m}{m}
=\frac{E_B/c^2}{\hbar\omega_{\rm free}/c^2}
=\frac{\Delta\omega}{\omega_{\rm free}}
$$

となる。  
さらに、近接場の幾何因子を

$$
\frac{\Delta\omega}{\omega_{\rm free}}
=F_{\rm geom}\!\left(\frac{R}{\lambda_c},n,\ell\right)\,C_{\rm coh}
$$

で定義し、`F_{\rm geom}` は半径比と量子数依存、`C_{\rm coh}` は位相整合度を表す無次元係数として運用する。  
これにより、`Δω` の観測I/F（`B.E.`、`Δm/m`）と形状・量子数依存を同一式で管理できる。  
この段階では「完全な核力理論」を主張せず、Part III 4.2.7 の検証I/Fに接続するための最小仮定を固定する。

Out of Scope（この節では扱わない項目）は次の通り。

- Coulomb 補正（pp 系の詳細補正）  
- スピン・軌道相互作用の詳細  
- 多体相関の高次項（3体力の完全理論化、殻模型の完全再構成）

これらは Part III 側で「有効モデルによる操作的I/F」と「波動干渉の第一原理的解釈」を分けて検証し、  
差分予測と反証条件（3σ運用）で段階的に固定する。

### 2.8 回転（フレームドラッグ）

静的なスカラー場 `P(x)` だけでは、回転質量が作る効果（フレームドラッグ / 重力磁場）を表現できない。  
相対論の置換候補としては、少なくとも弱場では既知の Lense-Thirring 極限と矛盾しないことが必要になる。[Will2014]

本稿では「最小の拡張」として、回転する質量（角運動量 `J`）の周りで、局所慣性系が角速度 `Ω_LT` で引きずられる（歳差する）ことを仮定し、弱場での代表式を検証量として固定する。

代表的には、ジャイロのフレームドラッグ歳差（Lense-Thirring）として

$$
\\boldsymbol{\\Omega}_{\\rm LT}
= \\frac{G}{c^2 r^3}\\left[3(\\mathbf{J}\\cdot\\hat{\\mathbf{r}})\\hat{\\mathbf{r}}-\\mathbf{J}\\right]
$$

が現れる。  
また人工衛星の軌道要素では、昇交点経度（ノード）の歳差が

$$
\\dot{\\Omega}_{\\rm node}
= \\frac{2GJ}{c^2 a^3(1-e^2)^{3/2}}
$$

で与えられる（`a` は長半径、`e` は離心率）。[Will2014]

GR/SR と P-model の位置づけは 1.1 節で述べた通りである。ここでの回転項は、その「弱場で一致すべき必須要件」として扱う。

### 2.9 量子（局所P構造・相関・selection）

本稿の中心は重力・時計・光伝播であるが、P-model の出発点「全ての物質は波であり、その媒質は時間である」は、量子現象（干渉・相関）とも接続し得る。  
ここでは量子力学の再導出を主張せず、**最小の立場と検証入口**（何を仮定し、どこが反証点になるか）を理論側に明記する。

- 粒子像：粒子は「波の反射／境界条件により束縛されたモード」として現れる（最小デモは `output/quantum/particle_reflection_demo.png`）。
- 相関像：もつれは「局所P構造の共有（共通の生成過程・共通履歴）」により生じる相関として再解釈し得る。ただし本稿は、この機構が量子力学の全予測（振幅則など）を再現できるとは主張しない。
- 反証入口：ベル不等式の議論では、観測手続き（time-tag の coincidence window、trial 定義、event-ready window）が **採用される事象集合（統計母集団）** を変え得る。これを選別重み `w_ab(λ)`（設定 `a,b` と隠れ変数 `λ` に依存）として表すと、

$$
P_{\\rm obs}(x,y\\mid a,b)
=\\frac{\\int d\\lambda\\,\\rho(\\lambda)\\,w_{ab}(\\lambda)\\,P(x\\mid a,\\lambda)P(y\\mid b,\\lambda)}
{\\int d\\lambda\\,\\rho(\\lambda)\\,w_{ab}(\\lambda)}
$$

となり、`w_ab` が設定に依存しない（同一母集団）という前提が崩れる場合、標準的な CHSH 導出の適用条件が満たされない。  
この「どの仮定が、どの解析手続きで破れ得るか」を一次公開データで検証し、反証可能性へ接続する（詳細：`doc/quantum/06_chsh_assumptions_and_selection.md`、`doc/quantum/07_pmodel_bell_chsh_notes.md`）。

---

## 3. 方法（Methods）

### 3.1 再現性（実行と生成物）

本研究では、各テーマについて一次データから図表と主要数値を生成する処理条件を固定し、同じ入力から同じ出力が得られるようにした。  
再現に必要な情報（指標定義、処理の段階分解、一次ソースと参照日）は本章と第8章、および付録（データ出典）にまとめる。

<!-- INTERNAL_ONLY_START -->
本リポジトリは「同じ入力→同じ出力」を再現できることを優先し、以下のパイプラインに統一している：

- 入力データ（キャッシュ含む）：`data/<topic>/`
- 計算スクリプト：`scripts/<topic>/`
- 生成物（固定ファイル名）：`output/<topic>/`

代表的な再現コマンド（推奨）：

- 一括実行（キャッシュのみ・ネットワークを使わない）：`python -B scripts/summary/run_all.py --offline --jobs 2`
- 一括実行（必要に応じて取得も行う）：`python -B scripts/summary/run_all.py --online --jobs 2`
- 資料生成（軽量；HTMLのみ）：`cmd /c output\summary\build_materials.bat quick-nodocx`
- 資料生成（Full；重い）：`cmd /c output\summary\build_materials.bat`
- 公開レポート再生成（単体）：`python -B scripts/summary/public_dashboard.py`
- 論文HTML再生成（単体；HTMLのみ）：`python -B scripts/summary/paper_build.py --mode publish --outdir output/summary --skip-docx`
- 論文化の整合チェック（引用/図表インデックス）：`python -B scripts/summary/paper_lint.py`

テーマ別の再現（単体実行例）：

- LLR（月レーザー測距）：`python -B scripts/llr/llr_batch_eval.py --offline`
- Cassini（太陽会合）：`python -B scripts/cassini/cassini_fig2_overlay.py`
- Viking（太陽会合）：`python -B scripts/viking/viking_shapiro_check.py`
- 太陽光偏向（VLBI, PPN γ）：`python -B scripts/theory/solar_light_deflection.py`
- Mercury（近日点移動）：`python -B scripts/mercury/mercury_precession_v3.py`
- GPS（衛星時計）：`python -B scripts/gps/compare_clocks.py`（必要に応じて `scripts/gps/plot.py`）
- EHT（ブラックホール影）：`python -B scripts/eht/eht_shadow_compare.py`

本稿の Table 1 は、確定出力（固定パスのCSV/JSON）から集計して作る（再計算は行わない）。  
Table 1 自体も固定名で出力する：`output/summary/paper_table1_results.md`。
<!-- INTERNAL_ONLY_END -->

### 3.2 データ取得とローカルキャッシュ

一次ソースから取得したデータは参照日（Accessed）を明記し、取得したファイルは内容同一性（ハッシュ等）で識別できる形で管理する。  
これにより、配布元が更新された場合でも、本稿の結果がどのデータに基づくかを追跡できる。

<!-- INTERNAL_ONLY_START -->
- 取得URL/sha256/保存先の固定：各テーマでマニフェスト（JSON）を用意し、`data/<topic>/` に保存する（例：`data/llr/llr_edc_batch_manifest.json`）。
- オフライン再現：`--offline` を指定すると、キャッシュが無い取得系タスクはスキップされる（スキップ理由は `output/summary/run_all_status.json` に残る）。
- 取得結果の再利用：HORIZONSなどのクエリ結果も保存して再利用する（例：`output/viking/horizons_cache/`）。
<!-- INTERNAL_ONLY_END -->

一次ソースの一覧は付録（データ出典）にまとめる。

### 3.3 共通パラメータ（β, δ）と定数

- `β`：光伝播（屈折/偏向/Shapiro）の強さを担う自由度。PPN対応の係数2は `β` 側で表し、GR整合は `β=1` とする。
- `δ`：速度項の飽和パラメータ（正・無次元）。`v→c` で発散（無限大）が現れないようにするために導入する（上限制約は結果節で示す）。
- `φ`：`P` から定義されるニュートン形ポテンシャル（2.1節）。係数を吸収するために `φ` の定義式を変更しない（必要なら `β` を微調整する）。

### 3.4 指標（誤差）と外れ値ポリシー

テーマごとに観測量の定義が異なるため、指標は「観測量に対する誤差」を直接集計する：

- LLR：`残差 = 観測 - モデル`（定数オフセット整列後）を用い、RMS（ns）を指標とする。バッチでは station×reflector×月（定数整列）などの単位で集計し、代表値として中央値RMSを採用する。
- Cassini：`y(t)`（周波数比）の時系列に対して、RMSE と相関係数を用いる（観測は一次データ処理、モデルは P-model の計算）。
- 太陽光偏向（PPN γ）：観測 `γ±σ` と P-model 予測 `γ=2β-1` の差を、zスコア（`(γ_obs-γ_pred)/σ`）で評価する。
- GPS：IGS Final（準実測）に対する時計残差のRMSを比較し、放送暦（BRDC）と P-model の差を評価する。
- EHT：公表値（リング直径）を参照し、「リング直径≒影直径」の近似を含む整合性チェックとして扱う（系統不確かさは結果節で明示する）。
- 二重パルサー：一致度 `R = Pdot_b(obs) / Pdot_b(GR)` を用い、`R=1`（四重極放射予測と一致）からのズレで「追加放射（双極など）」の上限を評価する。
- 重力波（GW150914）：公開strainから瞬時周波数 `f(t)` を抽出し、四重極チャープ則（Newton近似）への当てはめで位相（chirp）の整合性を評価する（本稿では簡易指標として `R^2` を併記）。

外れ値については「黙って除外」せず、出典行（source_file:lineno）まで追跡できる形で診断し、基準統計からの除外ポリシーを明示する。

### 3.5 テーマ別の処理概要（固定出力）

各テーマの一次ソースは付録（データ出典）にまとめ、ここでは手順の骨格のみを記す。

- LLR（月レーザー測距）：
  - 入力：EDC CRD Normal Point（NPT/NP2）。
  - 処理：time-tag（tx/rx/mid）を局ごとに最適化し、対流圏遅延・潮汐（固体地球/海洋荷重/月体）・Shapiro（太陽/地球）を段階追加して寄与を分解する。
- Cassini（太陽会合）：
  - 入力：PDS SCE1 の TDF（必要に応じて Ka/X を併用）。
  - 処理：観測量定義（pseudo-residual）を踏まえ、参照Shapiro（`β=1`）を足し戻して `y(t)` を復元し、binning/平滑化/デトレンド条件を固定して比較する。
- Viking（太陽会合）：
  - 入力：JPL Horizons（天体暦ベクトル）。
  - 処理：幾何（太陽中心ICRF）から往復 Shapiro 遅延を計算し、ピーク値と時刻を固定する。
- 太陽光偏向（VLBI, PPN γ）：
  - 入力：VLBI 等で推定された PPN パラメータ `γ` の代表値（公表値）。
  - 処理：P-model の対応 `γ=2β-1` と比較し、zスコア（`(γ_obs-γ_pred)/σ`）を算出する。併せて弱場の偏向角 `α(b)` を可視化する。
- Mercury（近日点移動）：
  - 入力：内部定数（G, M_sun, 初期条件など）。
  - 処理：運動方程式を数値積分し、近日点移動（角秒/世紀）を推定する。
- GPS（衛星時計）：
  - 入力：IGS Final（CLK/SP3）と放送暦（BRDC）。
  - 処理：同一期間（例：1日）で残差時系列とRMSを算出し、放送暦と P-model を比較する。
- EHT（ブラックホール影）：
  - 入力：EHT公表値（リング直径、質量、距離）。
  - 処理：リング直径≒影直径（κ=1）という近似の下で整合性を確認し、係数差（P-model vs GR）の差分予測を示す。
- 二重パルサー（軌道減衰）：
  - 入力：論文で公表された一致度 `R=Ṗ_b(obs)/Ṗ_b(GR)`（観測/GR）を使用する。
  - 処理：`R≈1` の一致を確認し、代替理論で問題になりやすい追加放射（双極放射など）の上限を、`|R-1|` と誤差から概算する。
- 重力波（GW150914, chirp）：
  - 入力：GWOSC公開strain（H1/L1, 32秒, 4 kHz）。
  - 処理：bandpass→Hilbert位相→瞬時周波数 `f(t)` を抽出し、四重極チャープ則（Newton近似）で位相（chirp）の整合性を確認する。

---

## 4. 結果（Results）

検証結果の要点は Table 1 に要約し、各テーマについて代表的な比較図を示す。

### 4.1 Table 1（検証サマリ）

Table 1 は「固定した処理条件で得た結果」を集計した要約であり、本文中では各節の理解の入口として用いる。

#### 4.1.1 総合スコアボード（全検証の現状）

Table 1 の全検証を「OK/要改善/不一致（目安）」で色分けし、現時点の到達点を俯瞰する。

`output/summary/validation_scoreboard.png`

（色の意味）
- 緑：OK
- 黄：要改善（要検証）
- 赤：不一致
- 灰：参考（z等の共通指標が定義できない／補助情報）

### 4.2 LLR（月レーザー測距）

各テーマで読み方を揃えるため、本節は **入力→処理→指標→図→注意** の順で要点をまとめる。

- **入力（一次ソース）**：EDC 公開の CRD Normal Point（station×reflector の多数サンプル）。[EDC][ILRSCRD]
- **処理（固定条件）**：幾何＋太陽Shapiroに、対流圏遅延・潮汐（固体地球＋月体）・海洋荷重を段階追加し、外れ値（スパイク）は統計から除外して一次データ行で追跡する（自動再割当はしない）。
- **指標**：定数オフセット整列後の残差（観測 - モデル）の RMS（ns）。代表値として station×reflector 分布の **中央値RMS** を用いる。
- **結果（主要数値）**：最良モデル（対流圏＋潮汐＋海洋荷重）で中央値RMSは **4.324546 ns**（N=13106）。
- **図（観測と差）**：
  - `output/llr/out_llr/llr_primary_overlay_tof.png`（観測 vs モデル）
  - `output/llr/out_llr/llr_primary_residual.png`（観測 - モデル：差が最も直感的に見える代表図）
  - `output/llr/out_llr/llr_primary_residual_compare.png`（モデル改善：地球中心→観測局→反射器の効果）
  - `output/llr/batch/llr_residual_distribution.png`（残差分布：段階追加での収束）
- **図（診断・寄与分解）**：
  - `output/llr/batch/llr_outliers_overview.png`（外れ値の俯瞰）
  - `output/llr/batch/llr_outliers_time_tag_sensitivity.png`（time-tag: tx/rx/mid の感度）
  - `output/llr/batch/llr_outliers_target_mixing_sensitivity.png`（ターゲット混入感度：現ターゲット vs 推定ターゲット）
  - `output/llr/batch/llr_apol_residual_vs_elevation.png`（APOL 残差 vs 仰角（SR+Tropo+Tide））
  - `output/llr/batch/llr_grsm_residual_vs_elevation.png`（GRSM 残差 vs 仰角（SR+Tropo+Tide））
  - `output/llr/batch/llr_matm_residual_vs_elevation.png`（MATM 残差 vs 仰角（SR+Tropo+Tide））
  - `output/llr/batch/llr_wetl_residual_vs_elevation.png`（WETL 残差 vs 仰角（SR+Tropo+Tide））
  - `output/llr/batch/llr_station_coord_delta_pos_eop.png`（局座標差分（pos+eop vs site log））
  - `output/llr/coord_compare/llr_station_coords_rms_compare.png`（局座標ソース比較（RMS））
  - `output/llr/batch/llr_tide_ablations_overall.png`（潮汐の寄与（固体潮汐/海洋荷重/月体潮汐））
  - `output/llr/batch/llr_shapiro_ablations_overall.png`（重力遅延の寄与（Shapiroの拡張））
  - `output/llr/batch/llr_rms_ablations_overall.png`（切り分け（Shapiro / 対流圏 / 月回転モデル））
  - `output/llr/coord_compare/llr_grsm_monthly_rms_pos_eop_vs_slrlog.png`（GRSM 月別RMS（局座標ソース比較））
  - `output/llr/batch/llr_rms_by_station_month_models.png`（局別×月別：期間依存の確認）
- **注意（系統誤差）**：LLR の ns 級残差は局座標/EOP/対流圏/潮汐/海洋荷重など複数の系統要因の重ね合わせで決まるため、RMSの低下だけでなく外れ値を根拠付きで説明できる状態が重要である。[Pearlman2019][IERS2010][Saastamoinen1972][Niell1996]

### 4.3 Cassini（太陽会合・ドップラー）

- **入力（一次ソース）**：Cassini 太陽会合のドップラー系列として、PDS SCE1 の TDF（一次データ）を用いる。[PDSCassiniSCE1]
- **処理（固定条件）**：TDFの `y` が “観測-予測” の疑似残差である点を踏まえ、参照Shapiroの足し戻し、欠測区間の扱い、符号などの前提を固定して比較する。
- **指標**：観測 vs P-model の RMSE / 相関（±10日などの窓）。
- **結果（主要数値）**：±10日の窓で **RMSE=7.069e-11 / corr=0.96308**（N=190, β=1）。
- **図（観測と差）**：
  - `output/cassini/cassini_fig2_overlay_zoom10d.png`（観測 vs P-model）
  - `output/cassini/cassini_fig2_overlay_full.png`（観測 vs P-model（全期間））
  - `output/cassini/cassini_pds_vs_digitized.png`（PDS一次データ（処理後） vs 論文図デジタイズ）
  - `output/cassini/cassini_fig2_residuals.png`（残差：観測 - P-model）
  - `output/cassini/cassini_beta_sweep_rmse.png`（βスイープ：RMSEの感度）
- **補足（図の読み方）**：
  - `cassini_pds_vs_digitized` は「PDS一次データの処理結果が、文献図のデジタイズと整合しているか（符号・時間軸・窓）」の確認。
  - `cassini_beta_sweep_rmse` は、βを変えてもRMSEがほぼ変わらない（=Cassini単体でβを強く拘束しない）ことを可視化する。
- **注意（β拘束の強さ）**：`β` の微調整（PPN: `(1+γ)=2β`）は可能だが、現状の窓・処理条件では RMSE が `β` に対してほぼフラットであり、Cassini単体で `β` を強く拘束するというより「一次データで形状とスケールが整合する」ことの確認として扱う。[Will2014][Bertotti2003]

### 4.4 太陽光偏向（観測γ vs P-model）

- **入力（一次ソース）**：VLBI 等で推定された PPN パラメータ `γ`（太陽重力による光偏向の強さ）の公表値を用いる。[Lebach1995][Shapiro2004VLBI][Fomalont2009]
- **処理（固定条件）**：P-model の `β` を固定し、対応 `γ=2β-1` により観測 `γ±σ` と比較する。あわせて弱場の偏向角 `α(b) ≈ 4β GM/(c^2 b)` の曲線を示す。
- **指標**：観測 `γ` と予測 `γ` の zスコア（`(γ_obs-γ_pred)/σ`）。参考として太陽縁の偏向角 `α_limb` の一致も示す。
- **結果（主要数値, β=1）**：最も小さなσの観測は **γ=0.99983±0.00026** で、予測 **γ=1.0** との差は **-0.00017（z=-0.65）**となり、1σ以内で整合する。
- **図**：
  - `output/theory/solar_light_deflection.png`（左：偏向角 α(b)、右：観測γ±σ と 予測γ=2β-1）
- **注意（扱いの範囲）**：ここで用いる `γ` は各解析が推定した公表値であり、一次の時系列データの直接フィットではない。主目的は PPN 対応（`γ=2β-1`）の整合性チェックである。

### 4.5 Viking（太陽会合・Shapiro遅延）

- **入力（一次ソース/参照）**：Sun-center ICRF での Earth/Mars ベクトルを JPL HORIZONS から取得し、太陽会合の往復幾何を再構成する。[JPLHorizons]  観測の代表値として、文献に示される往復 Shapiro 遅延ピーク（概ね 200–250 μs）を参照する。[Shapiro1977][Reasenberg1979]
- **処理（固定条件）**：往復路の Shapiro 遅延を幾何から計算し、`β` を固定して P-model の遅延ピークと比較する。
- **指標**：ピーク遅延（μs）とピーク時刻（UTC）。
- **結果（主要数値）**：ピークは **247.96 μs**（ピーク時刻 **1976-11-25 UTC**）となった。
- **図**：
  - `output/viking/viking_p_model_vs_measured_no_arrow.png`（P-model の往復遅延カーブと、文献の代表値レンジの比較）
- **注意（観測の粒度）**：ここでの「観測」は生データの時系列ではなく文献の代表値に基づくため、目的は “係数スケールとピーク位置” の整合性チェックである（一次データでの時系列フィットは別課題）。
<!-- INTERNAL_ONLY_START -->
（内部）集計値（固定）：`output/viking/viking_shapiro_result.csv`
<!-- INTERNAL_ONLY_END -->

### 4.6 Mercury（近日点移動）

- **入力（参照）**：近日点移動の観測残差の代表値として **42.98 ″/世紀** を参照する。[NobiliWill1986]
- **処理（固定条件）**：太陽中心の二体近似で水星軌道を数値積分し、周回ごとの近日点方向角から世紀あたりの移動量を推定する。
- **指標**：近日点移動の推定値（″/世紀）と、周回数に対する線形性（累積がほぼ直線か）。
- **結果（主要数値）**：P-model 推定は **42.992913 ″/世紀**（代表値との差 **0.012913 ″/世紀**）となった。
- **図**：
  - `output/mercury/mercury_orbit.png`（水星軌道と、近日点移動の累積）
- **注意（モデルの範囲）**：他惑星摂動・太陽四重極・小天体摂動などを含む精密暦モデルではないため、ここでの一致は主に “弱場での補正項のスケール” を見る目的である。
<!-- INTERNAL_ONLY_START -->
（内部）集計値（固定）：`output/mercury/mercury_precession_metrics.json`
<!-- INTERNAL_ONLY_END -->

### 4.7 GPS（衛星時計）

- **入力（一次ソース）**：IGS Final の CLK/SP3（準実測）を基準とし、同日の放送暦（BRDC）と比較する。[Dow2009][IGSBKGRootFTP]
- **処理（固定条件）**：2025-10-01 の 31 衛星について、IGS Final に対する時計残差（バイアス+ドリフト除去後）を算出し、BRDC と P-model の RMS を比較する。[Ashby2003]
- **指標**：衛星ごとの残差RMSと、その中央値（全衛星）。補助指標として、標準相対論補正 `δt_rel` の周期成分と P-model の周期成分の一致度（相関/RMSE）を示す。
- **結果（主要数値）**：中央値RMSは **BRDC=0.3278 ns**、**P-model=0.4022 ns**。P-model の方が小さい衛星は **6/31**。周期成分は **corr=0.999818 / RMSE=0.4776 ns** と高一致。
- **図**：
  - `output/gps/gps_rms_compare.png`（衛星ごとのRMS比較：BRDC vs P-model）
  - `output/gps/gps_residual_compare_G01.png`（時計残差の比較例：G01（観測IGSに対する比較））
  - `output/gps/gps_clock_residuals_all_31.png`（放送暦の時計残差：BRDC - IGS（全衛星））
  - `output/gps/gps_relativistic_correction_G02.png`（周期成分の比較例：標準式 vs P-model）
- **補足（図の読み方）**：
  - `gps_clock_residuals_all_31` は「放送暦（BRDC）が観測IGSからどの程度ずれているか」を全衛星で俯瞰する図（P-model図と対で読む）。
  - `gps_residual_compare_G01` は「単一衛星の時系列で、ずれの形（トレンド/周期）」を確認する代表例。
- **注意（解釈の難しさ）**：GPSの残差には時計モデル・軌道誤差・電離層/対流圏・アンテナ位相中心など複数要因が混在し、“P-model が観測より良い” を単一指標で言い切りにくい。ここでは同一入力（IGS Final固定）で再現可能な比較枠を提供することを優先した。[Ashby2003]
<!-- INTERNAL_ONLY_START -->
（内部）集計値（固定）：`output/gps/gps_compare_metrics.json`
<!-- INTERNAL_ONLY_END -->

### 4.8 EHT（ブラックホール影）

- **入力（一次ソース）**：EHT が公表するリング直径を用いる（M87*: 2017/2018、Sgr A*: 2017）。[EHTM87I2019][EHTM87VI2019][EHTM87PersistentI2024][EHTM87PersistentII2025][EHTSgrAI2022]  質量・距離は M87* は EHT、Sgr A* は GRAVITY の推定を用いる。[Gravity2019]
- **処理（固定条件）**：リング直径とシャドウ直径の関係を `θ_ring = κ θ_sh` とおく。P-model は `β` を固定（既定：`β=1`）し、観測リング直径から **`κ_fit`（リング/シャドウ比）** を推定する。整合性チェックとして、`κ=1` を置いた zスコアも参考として併記する。標準理論（GR）側は Schwarzschild 近似に加え、参考として Kerr（スピン/傾斜）による係数レンジを併記する（定義は付録）。
- **指標**：観測−モデルの zスコア（`κ=1` 仮定）と、`κ_fit`。
- **結果（主要数値, β=1）**：M87*（2017）は **42.0±3.0 µas（観測）**に対して **41.53±4.89 µas（P-model のシャドウ）**で、`κ_fit(P)` は **1.011±0.139**。Sgr A* は **51.8±2.3 µas（観測）**に対して **54.52±0.25 µas（P-model のシャドウ）**で、`κ_fit(P)` は **0.950±0.042**。また M87* の 2018 campaign では **43.3 µas（−3.1/+1.5）**が報告され、2017 と整合する（Δ=+1.3 µas; z≈0.3）。[EHTM87PersistentI2024]
- **図**：
  - `output/eht/eht_shadow_compare.png`（リング直径（観測）と影直径（P-model, κ=1）の比較）
  - `output/eht/eht_m87_persistent_shadow_ring_diameter.png`（M87*: リング直径の multi-epoch 整合（2017 vs 2018））
  - `output/eht/eht_shadow_zscores.png`（zスコア：κ=1仮定での整合性）
  - `output/eht/eht_kappa_fit.png`（κ_fit：観測リング直径と理論シャドウ直径から逆算した κ）
  - `output/eht/eht_ring_morphology.png`（リング形状指標：幅 W/d と非対称 A の一次ソースレンジ）
  - `output/eht/eht_scattering_budget.png`（散乱blur（230 GHz）のスケール感：リング直径/リング幅との比較）
  - `output/eht/eht_refractive_scattering_limits.png`（屈折散乱（refractive）による像ゆらぎのスケール：必要精度との比較）
  - `output/eht/eht_shadow_systematics.png`（κとKerrスピンレンジなどの系統不確かさの可視化）
  - `output/eht/eht_kerr_shadow_coeff_grid.png`（Kerr shadow 係数の (a*,inc) 依存（reference systematic））
  - `output/eht/eht_kerr_shadow_coeff_definition_sensitivity.png`（Kerr shadow 係数の定義依存（effective diameter）感度）
- **補足（図の読み方）**：
  - `eht_ring_morphology` は「直径」以外の追加観測量（幅・非対称）を示し、放射モデル/散乱/幾何（Kerr）の系統と結びつく入口になる。
  - `eht_m87_persistent_shadow_ring_diameter` は M87* の multi-epoch（2017/2018）の再現性チェックで、リング直径そのものが大きく変動していないことを確認する。
  - `eht_scattering_budget` と `eht_refractive_scattering_limits` は、Sgr A* の散乱が κ（リング/シャドウ変換）に与え得る系統のスケール感（約1%〜10%級）を示す（一次ソース：Zhu/Johnson/Psaltis）。
- **注意（系統誤差の支配）**：EHTは放射モデル・散乱・画像再構成・スピン/傾斜角依存が大きく、ここでの比較は “強い証明” というより、`κ` を主たる系統として明示した上での整合性チェックと差分予測の入口である。[Psaltis2018][Johnson2018][Zhu2018][Vincent2022ThickDisk][Johnson2020PhotonRing][Gralla2020PhotonRingShape]
  - 追加で、Kerr shadow は非円形であり「単一の直径係数」への写像は定義依存になり得るため、複数定義でのレンジ（definition envelope）も系統として併記した（`output/eht/eht_kerr_shadow_coeff_definition_sensitivity.png`）。
- **要点（検出可能性）**：M87* は θ_unit（質量・距離）が律速、Sgr A* は ring σ と κσ が律速になる（定量化は 5.1節）。
- **補足（Sgr A*, Paper V：M3 / 2.2 μm の near-passing と救済条件）**：Sgr A* の放射モデル検証（Paper V）では、near-passing を支配する観測側の key constraints として、(i) mm light curves（230 GHz）の 3h modulation index（M3）と、(ii) 2.2 μm flux 分布が残りやすい。本リポジトリでは、観測側前提（サンプル数、独立性、digitize、閾値のσ換算）を分解し、reconnection 条件図として固定した。[EHTSgrAV2022][Wielgus2022][Gravity2020Flux]
  - 図：`output/eht/eht_sgra_paper5_m3_nir_reconnection_conditions.png`
  - 代表値：2017（7 samples）vs historical（n=42; pre-2017-04-11）での KS は p_asymptotic≈0.0106（α=0.01）と境界に近い。D は 26/42（=0.619…）で、最小ステップ 1/42 だけ増えると p_asymptotic≈0.007 へ落ちる。この 1-step の反転は、0.051 近傍の historical 側 mi3 値が Δ≈0.0014（約+2.8%）ずれる程度で起こり得る（digitize 系統の感度）。
  - 独立性の扱い（effective n）でも p は変化し得る（例：curves n=30 と見なすと p_asymptotic≈0.0138）。
- **補足（S2/GRAVITY：強場の独立テスト）**：S2 の重力赤方偏移（f）と Schwarzschild 歳差（f_SP）は、現状は 1PN の整合性チェックとしては有効だが、P-model の時計項 `exp(-GM/(c^2 r))` と GR の `√(1-2GM/(c^2 r))` の差は 2PN（ε^2）で微小である。S2 近日点（r≈1400R_S, v≈7650 km/s）では f の差分予測は Δf≈−1.9×10^-4 程度で、3σ判別には σ(f)≲6×10^-5 が必要（観測は σ_total≈0.175）。歳差も同様に桁違いの精度が必要（order estimate）。[Gravity2018S2Redshift][Gravity2020S2Precession]
  - `output/eht/gravity_s2_pmodel_projection.json`
<!-- INTERNAL_ONLY_START -->
（内部）集計値（固定）：`output/eht/eht_shadow_compare.json`
（内部）詳細メモ：`doc/eht/README.md`
<!-- INTERNAL_ONLY_END -->

### 4.9 重力赤方偏移（GP-A / Galileo）

- **入力（一次ソース）**：GP-A（1976）と Galileo 5/6（GREAT）の重力赤方偏移テストで公表された偏差パラメータ ε を用いる。[Vessot1980][Delva2018]
- **注意（記号）**：ここでの ε は重力赤方偏移テストの偏差パラメータであり、2.5.1節で用いた BAO/AP の ε（warping）とは別定義である。
- **処理（固定条件）**：観測では `z_obs=(1+ε)ΔU/c^2`（ここで `z≡Δf/f`）と定義する。P-model（弱場・静止時計）では `dτ/dt=P0/P≈1+φ/c^2` となり、一次の重力赤方偏移は GR と同じ形（ε=0）を予測する。
- **指標**：各実験の ε と σ、および zスコア（`ε/σ`）。
- **結果（主要数値）**：GP-A は **ε=0 ± 7.0×10^-5**（z=0）。Galileo 5/6 は **ε=(0.19 ± 2.48)×10^-5**（z=+0.08）。いずれも ε=0 と整合。
- **図**：
  - `output/theory/gravitational_redshift_experiments.png`（観測 ε と誤差棒、0線=予測）
- **注意（誤差の解釈）**：GP-A の「70×10^-6 level」は論文表現のため、ここでは便宜上 σ=70×10^-6 を 1σ 相当の目安として扱っている（厳密な統計解釈は一次ソースを参照）。[Vessot1980]
<!-- INTERNAL_ONLY_START -->
（内部）入力固定：`data/theory/gravitational_redshift_experiments.json`
（内部）集計値（固定）：`output/theory/gravitational_redshift_experiments.json`
<!-- INTERNAL_ONLY_END -->

### 4.10 二重パルサー（軌道減衰：放射の制約）

- **理論（2.6節との接続）**：動的Pの最小仮定（`u=ln(P/P0)` が遠方で波動方程式を満たし、保存則により単極・双極放射が出ない）により、弱場・遠方では **四重極放射が主項**となる。したがって、軌道周期の減衰 `Pdot_b` は Peters–Mathews の四重極公式と同形（係数の正規化まで含めて既知の標準式）になり、P-model（弱場四重極）としては `R=1` を予測する必要がある。[PetersMathews1963][Peters1964]
- **入力（一次ソース）**：各系（NS-NS/NS-WD）について、一次ソースから軌道要素（`P_b`, `e`）と質量（`m1`, `m2`）、および外部補正後の `Pdot_b(int)`（運動学補正/銀河加速度など）を取り、P-model（弱場四重極）の `Pdot_b` を **再計算**して `R` を導出する。[WeisbergHuang2016][Kramer2021][Freire2012J1738][Antoniadis2013J0348]
- **再計算（固定式）**：弱場四重極の `Pdot_b` は

$$
\\dot{P}_b^{\\rm quad}
= -\\frac{192\\pi}{5} T_{\\odot}^{5/3}
\\left( \\frac{2\\pi}{P_b} \\right)^{5/3}
\\frac{m_1 m_2}{(m_1+m_2)^{1/3}}
\\frac{1+\\frac{73}{24}e^2+\\frac{37}{96}e^4}{(1-e^2)^{7/2}}
$$

で与えられる（`T_⊙ ≡ GM_⊙/c^3`、質量は `M_⊙` 単位）。本リポジトリでは `data/pulsar/binary_pulsar_orbital_decay.json` に一次ソースの数値を固定し、`scripts/pulsar/binary_pulsar_orbital_decay.py` が `Pdot_b(Peters–Mathews)` と `R` を自動導出する。
- **指標**：`R=1` が「四重極放射（弱場・遠方のP-model）と一致」を意味する。代替理論で問題になりやすい **双極放射**（追加のエネルギー損失）が存在すると、`R` は `1` から系統的にずれる（観測が許すズレの幅が“追加放射の上限”を与える）。
- **結果（主要数値）**：一次ソースの数値から再計算すると、
  - PSR B1913+16：**R=0.998292**（文献の **0.9983 ± 0.0016（1σ）**と整合）
  - PSR J0737-3039A/B：**R=0.999953**（文献の「四重極放射を **1.3×10^-4（95%）**で検証」という主張と整合）
  - PSR J1738+0333（NS-WD）：**R=0.943158**（1σ≈0.117）
  - PSR J0348+0432（NS-WD）：**R=1.059881**（1σ≈0.174）
  となり、いずれも `R=1` と矛盾しない（誤差の大きさは系により異なる）。観測誤差から概算すると、四重極以外の追加放射（双極放射など）は **<0.65%（3σ, B1913+16）**、**<0.025%（3σ, J0737）**、**<41%（3σ, J1738）**、**<58%（3σ, J0348）**程度までに制限される（ここでの“追加”は `Pdot_b` の差分を分率で表した粗い上限）。
- **図**：
  - `output/pulsar/binary_pulsar_orbital_decay.png`
- **注意（限界）**：本稿の 2.6 節は「弱場・遠方で四重極が主項になる」ことを最小仮定で整理した段階であり、強場（合体直前）や波形全体をP-modelから独立に計算できたことを主張しない。ここでは、少なくとも **“双極が出ない”という必要条件**と、観測が要求する **四重極支配**が矛盾しないことを確認する。なお NS-WD 連星は双極放射の有無に敏感な系だが、厳密な棄却は「差分の分率」だけではなく、スカラー荷（感度）や理論パラメータ空間への写像を伴う。

### 4.11 重力波（GW150914：chirp位相の整合性）

注意：本節は、LIGO/Virgo の標準的な matched filtering（テンプレート推定）を置換するものではない。公開 strain から抽出できる最小量（瞬時周波数 f(t)）を用いた **整合性チェック（sanity check）**として位置づける。

- **理論（2.6節との接続）**：四重極放射のチャープ則は

$$
\\frac{df}{dt}
= \\frac{96}{5}\\pi^{8/3}\\left(\\frac{G M_c}{c^3}\\right)^{5/3} f^{11/3}
$$

で与えられ、積分すると

$$
t_c-t
= \\frac{5}{256}\\left(\\frac{G M_c}{c^3}\\right)^{-5/3}(\\pi f)^{-8/3}
\\equiv A\\,f^{-8/3}
$$

となる。したがって、位相（chirp）の一致を確認する最小チェックは「観測から抽出した `f(t)` が `f^{-8/3}` の直線関係（`t=t_c-A f^{-8/3}`）を満たすか」である。[PetersMathews1963][Peters1964][Abbott2016GW150914]
- **入力（一次ソース）**：GW150914 の公開 strain（GWOSC, H1/L1, 32秒, 4 kHz）を使用する。[GWOSC][Abbott2016GW150914]
- **処理**：bandpass→Hilbert位相→瞬時周波数 `f(t)` を抽出し、`t=t_c-A f^{-8/3}`（四重極チャープ則, Newton近似）に当てはめる。
- **指標**：本稿では当てはめの `R^2` と、`A` から求めた `M_c`（チャープ質量）の概算値を併記する（厳密な推定ではない）。
- **結果（要点）**：公開データから抽出した `f(t)` は、イベント近傍の区間で四重極チャープ則と整合する傾向を示す（位相整合の入口）。
- **図**：
  - `output/gw/gw150914_chirp_phase.png`
  - `output/gw/gw150914_waveform_compare.png`（bandpass後の波形 vs 単純モデル（四重極）テンプレート）
- **補足（多イベント）**：同じ手順を複数イベント（GW151226 / GW170104 / GW170817 / GW190425 / GW200115_042309 / GW200129_065458 / GW200224_222234 / GW191216_213338 / GW200311_115853）に適用し、`R^2` と match を要約図と Table 1 にまとめた。BNS（GW170817/GW190425）や NSBH（GW200115_042309）は信号が長く弱い場合があるため、周波数トラック抽出はホワイトニング＋guided STFT を用いてロバスト性を確認した。個別イベントの図は付録（図表一覧）に掲載する。
- **要約図**：`output/gw/gw_multi_event_summary.png`（R^2 と match の一覧）
- **注意（位置づけ）**：ここでの `M_c` はデモ目的の簡易推定であり、LIGO公式の波形テンプレート＋事前分布を用いた厳密推定の代替ではない。P-model としては、最終的に位相（chirp）や波形まで再導出し、強場での差分予測を与える必要がある。

### 4.12 回転（フレームドラッグ：GP-B / LAGEOS）

- **入力（一次ソース）**：GP-B のフレームドラッグ歳差（drift rate）と、LAGEOS/LAGEOS II による Lense-Thirring 効果の一致度（要旨に明記された比）を用いる。[Everitt2011GPB][CiufoliniPavlis2004]
- **処理（固定条件）**：比較用の比 `μ ≡ |Ω_obs|/|Ω_pred|`（符号は座標系依存のため絶対値で比較）を用いる。P-model（弱場・回転項の最小拡張）は `μ=1` を予測する。
- **指標**：`μ` と `σ`、および zスコア（`(μ-1)/σ`）。
- **結果（主要数値）**：GP-B は **μ=0.949±0.184（z=-0.28）**。LAGEOS は **μ=0.99±0.05（z=-0.20）**。いずれも `μ=1` と整合する。
- **図**：
  - `output/theory/frame_dragging_experiments.png`（観測 μ と誤差棒、1線=予測）
- **注意（位置づけ）**：ここでは回転項の最小拡張が弱場の既知テストと矛盾しないことを確認する。強場（回転ブラックホール近傍）での差分予測は今後の課題である。
<!-- INTERNAL_ONLY_START -->
（内部）入力固定：`data/theory/frame_dragging_experiments.json`
（内部）集計値（固定）：`output/theory/frame_dragging_experiments.json`
<!-- INTERNAL_ONLY_END -->

### 4.13 量子（前提検証）

本稿の結論は量子現象の再導出ではないが、公開一次データに基づき「観測手続き（selection）が統計量を動かし得る入口」を定量化し、反証へ接続できる状態にした。以下では、量子テストを項目別に要約する。

#### 4.13.1 ベルテスト（time-tag / trial-based）

- **入力（一次ソース）**：NIST belltestdata（photon time-tag）、Delft（event-ready; trial table）、Weihs 1998（photon time-tag）。一次ソース一覧は `doc/PRIMARY_SOURCES.md` に固定する。
- **処理（固定条件）**：
  - NIST：GPS PPS 整列＋click delay（sync基準）＋coincidence window sweep（greedy pairing）。併せて build（hdf5）を用いた trial-based 集計（sync×slot）を行う。
  - Delft：event-ready window start（offset）を掃引し、有効trial数と CHSH を評価する。
  - Weihs：coincidence window sweep により CHSH |S| の感度を評価する。
- **指標**：NIST は click delay の setting 依存（KS距離）と CH の `J_prob`、Weihs/Delft は CHSH `S`（固定variant）。
- **結果（主要数値）**：
  - NIST：KS(A)=**0.0287**、KS(B)=**0.0411**。`J_prob` は window sweep により **-1.41e-3 → 2.10e-3** と変化し、trial-based best は **6.52e-6**。
  - Weihs：|S| は window sweep で **2.62 → 2.90** の範囲をとる。
  - Delft 2015：`S` は offset sweep で **1.71 → 2.50**、baseline は **2.422±0.204**。
  - Delft 2016（Sci Rep 30289；combined）：`S` は offset sweep で **1.10 → 2.54**、baseline combined は **2.346±0.184**。
- **図**：
  - `output/quantum/bell_selection_sensitivity_summary.png`（横断まとめ：selectionノブで統計量がどれだけ動くか）
  - `output/quantum/nist_belltest_time_tag_bias__03_43_afterfixingModeLocking_s3600.png`（NIST：delay分布）
  - `output/quantum/nist_belltest_trial_based__03_43_afterfixingModeLocking_s3600.png`（NIST：trial-based vs coincidence-based）
  - `output/quantum/delft_hensen2015_chsh.png`（Delft 2015：offset sweep）
  - `output/quantum/delft_hensen2016_srep30289_chsh.png`（Delft 2016：offset sweep）
  - `output/quantum/weihs1998_chsh_sweep_summary__multi_subdirs.png`（Weihs：複数条件での sweep）
- **注意（位置づけ）**：ここで固定するのは「selection が入る位置」とその感度であり、量子力学の全予測の再導出を主張しない（反証への入口）。

#### 4.13.2 重力×量子干渉（COW／原子干渉計／量子時計）

- **入力**：公開論文の式・代表値（本稿ではスケーリングと規模感を固定する）。
- **結果（主要数値）**：
  - COW（中性子干渉）：代表値（H=3 cm, v0=2000 m/s）で位相の振幅は **約11.16 cycles**（`phi0_cycles=-11.155...`）。
  - 原子干渉計重力計：`T=0.4 s` の例で重力項位相は **2.31e7 rad**（`phi_ref_rad`）。
  - 光格子時計（chronometric leveling）：時計由来 ΔU と測地 ΔU の差は **z=0.85σ**、有効εは **5.67e-4±6.68e-4**。
- **図**：
  - `output/quantum/cow_phase_shift.png`
  - `output/quantum/atom_interferometer_gravimeter_phase.png`
  - `output/quantum/optical_clock_chronometric_leveling.png`

#### 4.13.3 物質波干渉（電子二重スリット／de Broglie 精密）

- **結果（主要数値）**：
  - 電子二重スリット（600 eV）：de Broglie 波長 **50.05 pm**、縞間隔 **0.179 mrad**（角度）。
  - de Broglie 精密（原子反跳→α）：独立な α との差は **z=0.59σ**、有効εは **-5.33e-9±9.08e-9**。
- **図**：
  - `output/quantum/electron_double_slit_interference.png`
  - `output/quantum/de_broglie_precision_alpha_consistency.png`

#### 4.13.4 重力誘起デコヒーレンス（可視度低下）

- **指標**：可視度 `V` と、追加の時間構造ノイズ（fractional rate noise）`σ_y` の必要条件。
- **結果（主要数値）**：高さ分布の幅 `σ_z` を 0.1/1/10 mm としたとき、`V=0.5` に対応する時間スケールはそれぞれ **4.0e4 s / 4.0e3 s / 4.0e2 s**（同等に `σ_y` は 10倍刻みで増える）。
- **図**：`output/quantum/gravity_induced_decoherence.png`

#### 4.13.5 光の量子干渉（単一光子／HOM／スクイーズド光）

- **結果（主要数値）**：
  - 単一光子干渉（V≥0.8, λ=1550 nm）：等価な光路長雑音は **σ_L≈165 nm**。
  - HOM：reported visibility は **0.982±0.013（D=13 ns）**, **0.987^{+0.013}_{-0.020}（D=1 μs）**。
  - スクイーズド光（10 dB）：loss-only の下限として **η≥0.900**。
- **図**：`output/quantum/photon_quantum_interference.png`

#### 4.13.6 真空・QED精密（Casimir／Lamb shift）

- **結果（主要数値）**：
  - Casimir：sphere-plane の力スケールと、代表的な相対精度（最短距離で **~1%**）の入口を固定した（球直径 **201.7 μm**）。
  - Lamb shift：Lamb shift のスケーリング（`Z^4`）と、未確定な2-loop項のスケーリング（`Z^6`）を整理し、核サイズ項（非QED系統）の寄与例を示した。
- **図**：`output/quantum/qed_vacuum_precision.png`

<!-- INTERNAL_ONLY_START -->
### 4.14 代表図（固定パス：再掲）

LLR（外れ値の俯瞰）：
- `output/llr/batch/llr_outliers_overview.png`
- `output/llr/batch/llr_outliers_time_tag_sensitivity.png`
- `output/llr/batch/llr_residual_distribution.png`

Cassini（観測 vs P-model / 残差）：
- `output/cassini/cassini_fig2_overlay_zoom10d.png`
- `output/cassini/cassini_fig2_residuals.png`

Viking（観測 vs P-model）：
- `output/viking/viking_p_model_vs_measured_no_arrow.png`

EHT（観測 vs P-model / 差分予測）：
- `output/eht/eht_shadow_compare.png`
- `output/eht/eht_shadow_zscores.png`
- `output/eht/eht_shadow_differential.png`
- `output/eht/eht_shadow_systematics.png`
<!-- INTERNAL_ONLY_END -->

---

## 5. 差分予測（Differential Predictions / Falsifiability）

### 5.1 EHT：影直径係数の差

P-model（最小延長）の影直径係数は `4eβ`、Schwarzschild近似（GR）は `2√27` であり、`β` を固定しても数%の差が残る。  
リング直径（観測量）と影直径（理論量）の関係を

$$ \\theta_{\\rm ring} = \\kappa\\,\\theta_{\\rm sh}, \\qquad \\theta_{\\rm sh} = C\\,\\frac{r_s}{D}, \\qquad r_s \\equiv \\frac{2GM}{c^2} $$

と置く（`κ` は放射モデル/散乱/定義差などに由来する系統係数）。  
`β=1` としたとき、係数は

$$ C_{\\rm P} = 4e \\simeq 10.873, \\qquad C_{\\rm GR} = 2\\sqrt{27} \\simeq 10.392 $$

であり、相対差は約 **4.63%** となる。  
従って、`β` を他の弱場テストで固定した上で EHT による判別を目指す場合、目安として「総合の 1σ 不確かさが ~1.5% 以下（≈3σで係数差に届く）」が必要になる（主な不確かさ源は `κ` と Kerr スピン/傾斜依存、および質量・距離の系統）。  
差分予測として、影直径係数の差と、判別に必要な観測精度の目安（現状と必要精度のギャップ）を次図に示す。

必要精度の内訳（θ_unit / ring σ / κσ）とトレードオフは以下に固定出力した。

この内訳を読むと、M87* は θ_unit（=GM/(c^2 D) の角スケール）の相対誤差が約11.8%と大きく、3σ判別には約1.1%が必要（≈11×の改善）となり、ring σ や κσ を詰める前に質量・距離の精度改善が律速になる。一方 Sgr A* は θ_unit は要求を満たすが、現状の ring σ は約4.4%で、κ の系統も2%級になり得るため、ring σ を ~0.5 µas（≈1%）まで、κ を ~1% レベルまで拘束できるかが主なボトルネックになる。

`output/eht/eht_kappa_tradeoff.png`

`output/eht/eht_kappa_precision_required.png`

`output/eht/eht_delta_precision_required.png`

`output/summary/decisive_falsification.png`

### 5.2 速度飽和 δ：γ上限

速度項に `δ>0` を導入すると `v→c` で `γ` が発散せず、`γ_max ≈ 1/sqrt(δ)` に飽和する。  
一方で既存の高γ観測と矛盾しないためには `δ` は非常に小さくなければならない。本稿での採用値（例）と、既存観測からの上限制約を次図に示す。

`output/theory/delta_saturation_constraints.png`

### 5.3 宇宙論：距離二重性 η（棄却条件；距離指標前提に依存）

距離二重性（Etheringtonの関係）は

$$ \\eta(z) \\equiv \\frac{D_L}{(1+z)^2 D_A} $$

で定義され、標準（FRW + 光子数保存）では `η(z)=1` となる。  
観測は一般に

$$ D_L(z) = (1+z)^{2+\\epsilon_0}\\,D_A(z) \\quad (\\eta=(1+z)^{\\epsilon_0}) $$

でパラメータ化し、`ε0=0` からのずれとして制約する。[Martinelli2021DDR]  
現在データ（SNIa+BAO）では `ε0 = 0.013 ± 0.029 (1σ)` と報告され、標準（`ε0=0`）は整合する。  
一方で、静的背景Pの最小モデル（`D_L=(1+z)D_A` → `ε0=-1`）は、現行の距離指標前提のまま比較すると **約35σ** で外れる。これは「膨張なし＋静的幾何＋距離指標の基準が不変」という最小仮定の組が、現行の距離指標（SNIa/BAO）の前提をそのまま採る限り成立しにくいことを意味する。  
この張力の解釈（静的背景Pの最小仮定の棄却 vs 距離指標前提の再検証）は 2.5.3 節で述べた通りである。本稿では後者を必要補正（α, s_L, s_R）に落とし込み、固定出力している。現状制約（SNIa+BAO）に合わせる目安は **Δε≈1.013±0.029（z=1で約2倍）**である。  
同じ Δε は、灰色不透明度（光子数非保存）としては `τ(z)=2Δε ln(1+z)`（z=1で等価 τ≈1.40）に、標準光源の光度進化としては `s_L=-2Δε`（z=1で約1.5等級の系統）に対応する。

`output/cosmology/cosmology_distance_duality_constraints.png`

`output/cosmology/cosmology_ddr_reconnection_conditions.png`

距離二重性（DDR）の一次ソース（距離指標）依存を俯瞰する図（同じ静的最小でも結論がどれだけ動くか）：

`output/cosmology/cosmology_distance_duality_source_sensitivity.png`

さらに、DDRの ε0 制約どうしの **内部整合性（相互矛盾）**を、相関を無視した独立近似で

$$ z_{ij} = \frac{\epsilon_i-\epsilon_j}{\sqrt{\sigma_i^2+\sigma_j^2}} $$

として可視化すると、最大で **約3.25σ** の不一致が見える（例：Clusters+SNIa (2012, non-iso) と SGL+SNIa+GRB (2016, SIS)）。  
これは「静的背景P最小が整合する/しない」の議論以前に、距離指標側の系統（クラスター幾何モデル、レンズ質量モデル、校正・共分散、パラメータ化の差）が結論を大きく揺らし得ることを示す。

`output/cosmology/cosmology_distance_duality_internal_consistency.png`

また、一次ソースの中には `η(z)=1+η0 z` や `η(z)=1+η0 z/(1+z)` の形で制約が報告され、`ε0` へ換算する際に **z_ref（換算アンカー）**の仮定が入るものがある。  
この z_ref の取り方だけでも、静的最小（ε0=-1）の張力度（|z|）が大きく動きうる（定義差の寄与）。

`output/cosmology/cosmology_distance_duality_anchor_sensitivity.png`

次に、こうした「同一カテゴリ内のモデル差」を **系統幅 σ_cat** として（観測σに二乗和で加算し）張力度を再評価すると、
一部の BAO 非依存テストは 張力度が 3σ 以上から 3σ 未満へ移るケースがある。  
また SNIa+BAO / SNIa+H(z) についても、一次ソース追加により同一カテゴリ内の分散を見積もれるようになり、現状は σ_cat≈0.13（SNIa+BAO; n=2）、σ_cat≈0.05（SNIa+H(z); n=4）程度になる。代表の SNIa+BAO (2021) の張力度は 35σ → 約7.8σ まで緩むが、依然として強い張力が残る（ここでの σ_cat は「モデル差」の代理であり、距離指標全体の系統の上限を与えるものではない）。

`output/cosmology/cosmology_distance_duality_systematics_envelope.png`

`output/cosmology/cosmology_reconnection_parameter_space.png`

---

## 6. 議論（Discussion）

### 6.1 現状の達成範囲

本リポジトリの必須セット（初版）として、弱場テスト（LLR/Cassini/Viking/Mercury/GPS）と強場の代表例（EHT）を同一の枠組み（P, β, δ）で比較できる状態を整えた。  
ただし各テーマには「観測量の定義」や「系統誤差モデル」が存在し、特に EHT は放射モデル・散乱・スピン依存が残るため、差分予測の有意性評価は今後の精密化が必要である。

EHTについては、現状の比較は「リング直径 ≒ 影直径（κ=1）」という近似の上に成り立つ。  
この仮定のもとでは、P-model/GRともに観測と大きく矛盾しない一方で、統計的に決定的な優劣（置換可能性の証拠）を主張できる段階ではない。  
整合性チェックの指標（zスコア）を次図に示す。

`output/eht/eht_shadow_zscores.png`

従って現時点の位置づけは、(i) `κ_fit` を通じた「リング/シャドウの系統スケール把握」と粗い整合性チェック、(ii) `β` を固定した上で残る「係数差（数%）」の差分予測（`output/eht/eht_shadow_differential.png`）である。

### 6.2 制限事項と今後

- LLR：外れ値は出典行（source_file:lineno）まで追跡し、time-tag感度（`output/llr/batch/llr_outliers_time_tag_sensitivity.png`）とターゲット混入感度（`output/llr/batch/llr_outliers_target_mixing_sensitivity.png`）で切り分けた。最大級スパイクは「ターゲット混入（反射器取り違え）」の可能性が高いため、混入疑いは基準統計から除外し、一次データ行で確認する（自動再割当はしない）。
<!-- INTERNAL_ONLY_START -->
外れ値診断CSV（source_file:lineno 追跡）：`output/llr/batch/llr_outliers_diagnosis.csv`
<!-- INTERNAL_ONLY_END -->
- Cassini：TDFの観測量定義（予測との差、プラズマ補正、欠測補完）を一次ソースで追跡し続ける。
- Viking：本稿では幾何（HORIZONS）から往復Shapiro遅延のピークを再現し、文献代表値と比較した。一次の“時系列”検証に踏み込むには、当時の観測データ体系（レンジ/ドップラー、各種補正、推定パラメータ）まで含む再解析が必要になる。
- Mercury：近日点移動は縮約した数値モデルで再現している。厳密な検証には、n体摂動・太陽J2・座標時系/エフェメリス整合などを含む枠組みへ拡張し、独立な参照（近代エフェメリスや文献）との比較を系統立てる必要がある。
- GPS：現状は IGS Final を基準とした「残差RMSの比較」であり、衛星/期間/補正体系に依存して見え方が変わる。より強い主張には、複数日・複数アークでの統計、観測モデル（電離圏/対流圏/相対論補正）の扱い、推定（推定しない/する）量の固定などの設計を明確化する必要がある。
- EHT：κ（リング/影）とKerrスピン依存を系統不確かさとして扱い、(i) κの許容範囲の明示、(ii) `β` を固定した上で残る係数差の評価、(iii) 判別に必要な観測精度要件（`output/eht/eht_kappa_tradeoff.png` / `output/eht/eht_kappa_precision_required.png`）を整理した（系統の俯瞰：`output/eht/eht_shadow_systematics.png` / `output/eht/eht_shadow_differential.png`）。残る課題は κ の物理（放射モデル/散乱/定義差）を 1〜2% レベルで拘束できるかである。
- 量子（前提検証の入口）：本稿の結論ではないが、公開一次データ（NIST time-tag / Delft event-ready / Weihs 1998 time-tag）を用いた前提検証の入口（time-tag選別／trial 定義）と、CHSH 前提（同一母集団）条件の数式化を整備した（`output/quantum/nist_belltest_time_tag_bias.png` / `output/quantum/delft_hensen2015_chsh.png` / `output/quantum/weihs1998_chsh_sweep__weihs1998_longdist_longdist1.png`）。また重力×量子干渉の入口として、COW（中性子干渉）・原子干渉計重力計・光格子時計（chronometric leveling）に加え、物質波干渉（電子二重スリット）と de Broglie 精密（原子反跳→αの整合）まで含む最小再現（スケーリング＋一致度）を固定した（`output/quantum/cow_phase_shift.png` / `output/quantum/atom_interferometer_gravimeter_phase.png` / `output/quantum/optical_clock_chronometric_leveling.png` / `output/quantum/electron_double_slit_interference.png` / `output/quantum/de_broglie_precision_alpha_consistency.png`）。さらに重力誘起デコヒーレンスについて、観測量（V）と P-model 側の最小パラメータ（σy）の関係を固定し、必要条件の規模感を可視化した（`output/quantum/gravity_induced_decoherence.png`）。加えて光の量子干渉（単一光子visibility／HOM visibility／スクイーズド光（dB））の観測量を一次ソースで固定し、time-tag/遅延が解析に入る位置づけを整理した（`output/quantum/photon_quantum_interference.png`）。さらに真空・QED精密（Casimir/Lamb）の一次ソースと観測量の入口を固定し、「ここを再現できない量子解釈は棄却」という最低要件を明文化した（`output/quantum/qed_vacuum_precision.png`）。今後は NIST の追加run（装置条件依存）、trial-based vs coincidence-based の拡張、および Giustina 2015 等の photon time-tag 一次データ統合へ拡張する。
- 速度飽和 δ：本稿では「無限大（100%）は現実に出ない」という仮定を `δ>0` で表現したが、δ自体は現状“測定”されたものではない。既知の高γ観測と矛盾しないための上限（`output/theory/delta_saturation_constraints.png`）は整理したが、今後は δ が観測へ与える具体的な差分（どの実験で、どの精度で判別可能か）を明文化する必要がある。

### 6.3 判別条件（反証可能性の形）

- EHT：`β` を太陽系検証（Cassini等）で固定した上で、`κ` と Kerr 系統（スピン/傾斜/散乱）を 1〜2% レベルで拘束し、リング直径の統計誤差を同程度まで下げられれば、係数差（約4.6%）の有意な判別が視野に入る。M87* は θ_unit（質量/距離）が律速、Sgr A* は ring σ と κσ が律速（`output/eht/eht_kappa_tradeoff.png` / `output/eht/eht_kappa_precision_required.png`；詳細は 5.1節）。
- 速度飽和 δ：普遍定数として仮定する限り、観測される最大 `γ`（高エネルギー粒子/宇宙ニュートリノ等）により `δ` の上限が更新される。δによる“差”を直接測るには、`γ` が `γ_max≈1/√δ` に近づく領域の観測が必要である（現状の採用値（例）では実用上きわめて遠い）。

必要精度（EHT）と棄却条件（EHT/δ）の要点は、反証条件パックとして次図にまとめた。

`output/summary/decisive_falsification.png`

---

## 7. 結論（Conclusion）

P-model は、時間波密度 `P` を介して重力・時計・光伝播を統一的に表現する枠組みである。  
弱場テストの代表例に対して再現可能な検証パイプラインを整え、差分予測（EHTの係数差・速度飽和δ）を提示した。  
量子側については本稿の結論ではないが、ベルテスト公開データ（NIST/Delft/Weihs 1998）を用いた観測手続き（time-tag選別／trial 定義）の前提検証パイプラインと、棄却条件の整理を整備した。  
宇宙論では、DESI DR1 の raw から ξ0/ξ2+cov を自前生成し BAO ε fit を行った結果、pbg 座標で複数トレーサにまたがる張力が安定に現れることを確認した。ただし本稿ではこれを直ちに P-model の棄却とは解釈せず、距離指標前提の再接続条件（α, s_L, s_R；Δε）として定量化し、反証可能な予測として固定出力した。  
今後は、差分予測が観測で判別可能になる条件（不確かさの支配要因）を絞り込み、強場・宇宙論領域での反証可能性をより直接の観測提案へ落とし込む。

---

## 8. データ/コードの入手性（Reproducibility / Availability）

本稿で用いたデータは公表されている一次ソースに基づき、出典は付録にまとめる。  
解析コードと処理条件は「同じ入力→同じ結果」が再現できる形で整理している。一次ソース（URL/参照日/ローカル保存先）は付録「データ出典（一次ソース）」にまとめる。

再現の入口（推奨）：
- HTMLのみ更新（軽量；Windows）：`cmd /c output\summary\build_materials.bat quick-nodocx`
- HTMLのみ更新（軽量；任意OS）：`python -B scripts/summary/paper_build.py --mode publish --outdir output/summary --skip-docx` と `python -B scripts/summary/public_dashboard.py`
- publish QC（軽量）：`python -B scripts/summary/paper_qc.py`
- 量子（ベル；一次データ再解析）：`python -B scripts/quantum/nist_belltest_time_tag_reanalysis.py` / `python -B scripts/quantum/delft_hensen2015_chsh_reanalysis.py` / `python -B scripts/quantum/weihs1998_time_tag_reanalysis.py`（取得：`python -B scripts/quantum/fetch_nist_belltestdata.py` / `python -B scripts/quantum/fetch_delft_hensen2015.py` / `python -B scripts/quantum/fetch_weihs1998_zenodo_7185335.py`）
- 公開マニフェスト：`python -B scripts/summary/release_manifest.py --no-hash`
- フル再現（重い）：`python -B scripts/summary/run_all.py --offline --jobs 2`

---

<!-- INTERNAL_ONLY_START -->
## 9. 参考文献（References）

- 作業用リスト：`doc/paper/30_references.md`
<!-- INTERNAL_ONLY_END -->
