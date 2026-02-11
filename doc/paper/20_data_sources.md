本付録は、本稿で扱う検証が依拠する **一次ソース**（公表データ/仕様/代表文献）を列挙する。

## LLR（月レーザー測距）

- [EDC] DGFI-TUM EDC（ILRSデータミラー）。Web: `https://edc.dgfi.tum.de/`（ftpミラーもあり）。Accessed: 2026-01-03.
- [ILRSCRD] ILRS “CRD - Consolidated Ranging Data Format”. Web: `https://ilrs.gsfc.nasa.gov/data_and_products/formats/crd.html`. Accessed: 2026-01-03.

## Cassini（太陽会合）

- [PDSCassiniSCE1] NASA Planetary Data System (PDS). “CASSINI RSS RAW DATA SET - SCE1 V1.0” (CO-SS-RSS-1-SCE1-V1.0). DOI: 10.17189/ske1-av41.  
  Landing page: `https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=CO-SS-RSS-1-SCE1-V1.0`. Accessed: 2026-01-03.
<!-- INTERNAL_ONLY_START -->
  Mirror used in this repo: `https://atmos.nmsu.edu/pdsd/archive/data/co-ss-rss-1-sce1-v10/`.
<!-- INTERNAL_ONLY_END -->

## Viking（太陽会合）

- [JPLHorizons] NASA/JPL SSD HORIZONS. Web: `https://ssd.jpl.nasa.gov/horizons/`（API: `https://ssd.jpl.nasa.gov/api/horizons.api`）。Accessed: 2026-01-03.

## 太陽光偏向（VLBI, PPN γ）

- [Lebach1995] Lebach, D. E., et al. “Measurement of the Solar Gravitational Deflection of Radio Waves Using Very-Long-Baseline Interferometry.” *Physical Review Letters* **75**, 1439 (1995). DOI: 10.1103/PhysRevLett.75.1439. Accessed: 2026-01-04.
- [Shapiro2004VLBI] Shapiro, I. I., et al. “Measurement of the Solar Gravitational Deflection of Radio Waves Using Geodetic Very-Long-Baseline Interferometry Data, 1979–1999.” *Physical Review Letters* **92**, 121101 (2004). DOI: 10.1103/PhysRevLett.92.121101. Accessed: 2026-01-04.
- [Fomalont2009] Fomalont, E. B., et al. “Measurement of the Solar Gravitational Deflection of Radio Waves in Agreement with General Relativity.” arXiv:0904.3992 (2009). Accessed: 2026-01-04.

## GPS（衛星時計）

- [IGSBKGRootFTP] IGS 公開データ（BKG mirror）. Web: `https://igs.bkg.bund.de/root_ftp/IGS/`. Accessed: 2026-01-03.
- GPS 放送暦（RINEX, BRDC）は同サイトの公開プロダクトを使用する。

## 重力赤方偏移（GP-A / Galileo 5/6）

- [Vessot1980] Vessot, R. F. C., et al. “Test of Relativistic Gravitation with a Space-Borne Hydrogen Maser.” *Physical Review Letters* **45**, 2081 (1980). DOI: 10.1103/PhysRevLett.45.2081.  
  公開コピー（NASA NTRS）: `https://ntrs.nasa.gov/citations/19810000898`. Accessed: 2026-01-04.
- [Delva2018] Delva, P., et al. “A gravitational redshift test using eccentric Galileo satellites.” *Physical Review Letters* **121**, 231101 (2018). DOI: 10.1103/PhysRevLett.121.231101.  
  arXiv: `https://arxiv.org/abs/1810.12191`. Accessed: 2026-01-04.

## 回転（フレームドラッグ：GP-B / LAGEOS）

- [Everitt2011GPB] Everitt, C. W. F., et al. “Gravity Probe B: Final Results of a Space Experiment to Test General Relativity.” *Physical Review Letters* **106**, 221101 (2011). DOI: 10.1103/PhysRevLett.106.221101.  
  arXiv: `https://arxiv.org/abs/1105.3456`. Accessed: 2026-01-08.
- [CiufoliniPavlis2004] Ciufolini, I., and Pavlis, E. C. “A confirmation of the general relativistic prediction of the Lense-Thirring effect.” *Nature* **431**, 958-960 (2004). DOI: 10.1038/nature03007.  
  Web: `https://www.nature.com/articles/nature03007`. Accessed: 2026-01-08.

## Mercury（近日点移動）

- 外部の時系列観測データには依存せず、重力定数・太陽質量などの定数と初期条件を用いて数値積分する（代表値の比較は [NobiliWill1986]）。

## EHT（ブラックホール影）

- [EHTM87I2019] M87* リング直径（EHT Paper I）
- [EHTSgrAI2022] Sgr A* リング直径（EHT Paper I）
- [Gravity2019] Sgr A* 質量・距離の拘束（幾何学的）
- [Gravity2018S2Redshift] S2 星軌道：重力赤方偏移（GRAVITY Collaboration 2018）
- [Gravity2020S2Precession] S2 星軌道：Schwarzschild 歳差（GRAVITY Collaboration 2020）
- [Johnson2018] Sgr A* の散乱と固有構造（散乱カーネル/パワースペクトルの基礎）
- [Psaltis2018] Sgr A* の異方性散乱モデル（散乱モデルの一次ソース）
- [Zhu2018] 散乱による制限（Sgr A* 影サイズ/非円形性でのGRテストが制限される点の整理）

## 二重パルサー（軌道減衰）

- [WeisbergHuang2016] Weisberg, J. M., and Huang, Y. “Relativistic Measurements from Timing the Binary Pulsar PSR B1913+16.” *The Astrophysical Journal* **829**, 55 (2016). DOI: 10.3847/0004-637X/829/1/55. Accessed: 2026-01-05.
- [Kramer2021] Kramer, M., et al. “Strong-field Gravity Tests with the Double Pulsar.” *Physical Review X* **11**, 041050 (2021). DOI: 10.1103/PhysRevX.11.041050. Accessed: 2026-01-05.

## 重力波（GW150914）

- [GWOSC] Gravitational Wave Open Science Center (GWOSC). “GW150914 (GWTC-1-confident, v3) event API / strain.” Web: `https://gwosc.org/eventapi/`. Accessed: 2026-01-06.
- [Abbott2016GW150914] Abbott, B. P., et al. “Observation of Gravitational Waves from a Binary Black Hole Merger.” *Physical Review Letters* **116**, 061102 (2016). DOI: 10.1103/PhysRevLett.116.061102. Accessed: 2026-01-06.

## 宇宙論（赤方偏移：背景P）

- 外部の観測データには依存せず、背景の時間波密度 `P_bg(t)` が時間変化する場合に `1+z=P_em/P_obs` が得られることを可視化する（概念図）。  
  `H0^(P)` は可視化用の代表値（既定 70 km/s/Mpc）として与え、スクリプト引数で変更できる。
- 同じ仮定（宇宙膨張なし＋背景P）を置いたとき、距離二重性 `η(z)` や Tolman 表面輝度などの観測量スケーリングが FRW とどう“差”になるかを、定義と最小仮定から導いた（データフィットではない）。
- 距離二重性（DDR）の観測制約（棄却条件の固定入力）：
  - [Martinelli2021DDR] Martinelli, M., et al. “Euclid: Forecast constraints on the cosmic distance duality relation with complementary external probes.” *Astronomy & Astrophysics* (2021). DOI: 10.1051/0004-6361/202039637.  
    arXiv: `https://arxiv.org/abs/2007.16153`. Accessed: 2026-01-08.
  - [Avgoustidis2010Opacity] Avgoustidis, A., et al. “Constraints on cosmic opacity and beyond the standard model physics from cosmological distance measurements.” *JCAP* (2010). DOI: 10.1088/1475-7516/2010/10/024.  
    arXiv: `https://arxiv.org/abs/1004.2053`. Accessed: 2026-01-09.（SNIa+H(z) による ε0 制約としても利用）
  - [Holanda2012ClustersDDR] Holanda, R. F. L., et al. “Probing the cosmic distance duality relation with the Sunyaev-Zel’dovich effect, X-ray observations and supernovae Ia.” *Astronomy & Astrophysics* (2012). DOI: 10.1051/0004-6361/201118343.  
    arXiv: `https://arxiv.org/abs/1104.3753`. Accessed: 2026-01-09.（クラスター+SNe Ia の η0 制約を z=1 で ε0 に換算）
  - [LiWuYu2011DDR] Li, Z., et al. “Cosmological-model-independent tests for the distance-duality relation from Galaxy Clusters and Type Ia Supernova.” *The Astrophysical Journal Letters* (2011). DOI: 10.1088/2041-8205/729/1/L14.  
    arXiv: `https://arxiv.org/abs/1101.5255`. Accessed: 2026-01-09.（クラスター+SNe Ia の η0 制約を z=1 で ε0 に換算）
  - [Holanda2016SGLDDR] Holanda, R. F. L., et al. “Probing the distance-duality relation with high-z data.” arXiv: `https://arxiv.org/abs/1611.09426`. Accessed: 2026-01-10.（SGL+SNe Ia+GRB。power-law `η=(1+z)^η0` は本稿の `ε0` と一致）
  - [LiLin2017RadioDDR] Li, X., and Lin, H.-N. “Testing the distance duality relation using type-Ia supernovae and ultra-compact radio sources.” arXiv: `https://arxiv.org/abs/1710.11361`. Accessed: 2026-01-10.（SNe Ia + 電波コンパクトソース。一次ソースの η は `D_A(1+z)^2/D_L` 定義のため、本稿の η へは逆数換算が必要）
- 不透明度（cosmic opacity）の一次ソース制約（αの固定入力）：
  - [LvXia2016CDDR] Lv, M.-Z., and Xia, J.-Q. “Constraints on Cosmic Distance Duality Relation from Cosmological Observations.” *Physics of the Dark Universe* (2016). DOI: 10.1016/j.dark.2016.06.003.  
    arXiv: `https://arxiv.org/abs/1606.08102`. Accessed: 2026-01-09.
  - [Avgoustidis2010Opacity] Avgoustidis, A., et al. “Constraints on cosmic opacity and beyond the standard model physics from cosmological distance measurements.” *JCAP* (2010). DOI: 10.1088/1475-7516/2010/10/024.  
    arXiv: `https://arxiv.org/abs/1004.2053`. Accessed: 2026-01-09.
  - [HolandaCarvalhoAlcaniz2014Opacity] Holanda, R. F. L., Carvalho, J. C., and Alcaniz, J. S. “Model-independent constraints on the cosmic opacity.” arXiv: `https://arxiv.org/abs/1207.1694`. Accessed: 2026-01-10.（Union2.1+H(z), SDSS+H(z)。τ(z)=2ǫ z を z=1 で α に換算。平坦（Ωk=0）のサンプルを採用）
  - [WangWeiLiXiaZhu2017CurvatureOpacity] Wang, G.-J., et al. “Model-independent Constraints on Cosmic Curvature and Opacity.” arXiv: `https://arxiv.org/abs/1709.07258`. Accessed: 2026-01-10.（H(z)+JLA。曲率Kと不透明度εを同時推定し、H0事前分布（無し/Planck/局所）の感度も併記）
  - （観測; 近傍）GW170817×NGC4993：EM距離とGW距離の比から α を換算した参考入力（z≪1のため α は実質未拘束）。
    - [Abbott2017GW170817] Abbott, B. P., et al. “GW170817: Observation of Gravitational Waves from a Binary Neutron Star Inspiral.” *Physical Review Letters* **119**, 161101 (2017). DOI: 10.1103/PhysRevLett.119.161101.  
      arXiv: `https://arxiv.org/abs/1710.05832`. Accessed: 2026-01-10.（GW距離 D_L の一次ソース）
    - [Hjorth2017NGC4993Distance] Hjorth, J., et al. “The distance to NGC 4993: the host galaxy of the gravitational-wave event GW170817.” arXiv: `https://arxiv.org/abs/1710.05856`. Accessed: 2026-01-10.（z_cosmic と距離推定）
    - [Cantiello2018NGC4993SBF] Cantiello, M., et al. “A precise distance to the host galaxy of the binary neutron star merger GW170817 using surface brightness fluctuations.” arXiv: `https://arxiv.org/abs/1801.06080`. Accessed: 2026-01-10.（SBF距離）
  - [QiCaoPanLi2019GWOpacity] Qi, J.-Z., Cao, S., Pan, Y., and Li, J. “Cosmic opacity: cosmological-model-independent tests from gravitational waves and Type Ia Supernova.” arXiv: `https://arxiv.org/abs/1902.01702`. Accessed: 2026-01-10.（GW標準サイレン+SNe Ia による不透明度テスト。Pantheon+ET（forecast）の ϵ 拘束を α に換算して参考入力として使用）
  - [LiaoAvgoustidisLi2015Transparency] Liao, K., Avgoustidis, A., and Li, Z. “Is the Universe Transparent?” arXiv: `https://arxiv.org/abs/1512.01861`. Accessed: 2026-01-09.（JLA+H(z); τ(z)=2ϵ z を z=1 で α に換算）
  - [NairJhinganJain2013Transparency] Nair, R., Jhingan, S., and Jain, D. “Cosmic distance duality and cosmic transparency.” arXiv: `https://arxiv.org/abs/1210.2642`. Accessed: 2026-01-09.（BAO+SNe Ia の不透明度差 Δτ から ϵ を推定）
- Tolman表面輝度（SB dimming）の一次ソース制約（有効指数 n の固定入力）：
  - [LubinSandage2001Tolman] Lubin, L. M., and Sandage, A. “The Tolman Surface Brightness Test for the Reality of the Expansion. IV. ...” *The Astronomical Journal* (2001). DOI: 10.1086/322134.  
    arXiv: `https://arxiv.org/abs/astro-ph/0106566`. Accessed: 2026-01-08.
- SN time dilation（距離指標と独立な時間伸長の一次ソース制約）：
  - [Blondin2008TimeDilation] Blondin, S., et al. “Time Dilation in Type Ia Supernova Spectra at High Redshift.” arXiv: `https://arxiv.org/abs/0804.3595`. Accessed: 2026-01-09.
- CMB温度スケーリング T(z)（距離指標と独立な一次ソース制約）：
  - [Avgoustidis2011CMBTz] Avgoustidis, A., et al. “Constraints on the CMB temperature-redshift dependence from SZ and distance measurements.” arXiv: `https://arxiv.org/abs/1112.1862`. Accessed: 2026-01-09.
- SNe Ia の標準光源進化（magnitude evolution）の一次ソース制約（s_Lの目安）：
  - [Scolnic2018Pantheon] Scolnic, D. M., et al. “The Complete Light-curve Sample of Spectroscopically Confirmed SNe Ia from Pan-STARRS1 and Cosmological Constraints from the Combined Pantheon Sample.” arXiv: `https://arxiv.org/abs/1710.00845`. Accessed: 2026-01-09.（Pantheon lcparam を誤差予算推定（dmb binning）に使用）
  - Pantheon 公開データ（系統共分散; sys covariance, mag^2）：`https://raw.githubusercontent.com/dscolnic/Pantheon/master/sys_full_long.txt`. Accessed: 2026-01-09.（誤差予算の上限側（stat+sys）の目安に使用）
  - [Brout2022PantheonPlus] Brout, D., et al. “The Pantheon+ Analysis: The Full Data Set and Light-curve Release.” arXiv: `https://arxiv.org/abs/2202.04077`. Accessed: 2026-01-09.（Pantheon+ 距離（μ）と共分散の一次ソース）
    - Data release（PantheonPlusSH0ES）：`https://github.com/PantheonPlusSH0ES/DataRelease`. Accessed: 2026-01-09.
    - 距離/共分散は DataRelease の配布ファイルを使用する。
  - [Linden2009SNEvo] Linden, S., et al. “Cosmological parameter extraction and biases from type Ia supernova magnitude evolution.” arXiv: `https://arxiv.org/abs/0907.4495`. Accessed: 2026-01-09.
  - [Kumar2021SNEvoCCDR] Kumar, D., et al. “A non-parametric test of variability of Type Ia supernovae luminosity and CDDR.” arXiv: `https://arxiv.org/abs/2107.04784`. Accessed: 2026-01-09.（Pantheon+CC。Part II は CDDR を仮定し、M_B(z)=M_B0+M_B1 ln(1+z) を制約）
  - [Hu2023SNEvoSGLS] Hu, J., et al. “Calibrating the effective magnitudes of type Ia supernovae with a model-independent method.” arXiv: `https://arxiv.org/abs/2309.17163`. Accessed: 2026-01-09.（Pantheon+強い重力レンズ; δM=ε ln(1+z) を s_L に換算。CDDR仮定のため注意）
  - [Kang2020SNEvoAgeHR] Kang, Y., et al. “Early-type Host Galaxies of Type Ia Supernovae. II. Evidence for Luminosity Evolution in Supernova Cosmology.” arXiv: `https://arxiv.org/abs/1912.04903`. Accessed: 2026-01-10.（HR-age の傾きを z=0→1 の Δage≈5.3 Gyr に当てはめ、z=1 の ΔM として s_L へ換算）
  - [Dainotti2021PantheonH0Evo] Dainotti, M. G., et al. “On the Hubble Constant Tension in the SNe Ia Pantheon Sample.” arXiv: `https://arxiv.org/abs/2103.02117`. Accessed: 2026-01-09.（H0(z)=H0/(1+z)^α を s_L=2α として「標準光源の見かけの進化」に換算。CDDR仮定ではないが、距離スケールとの退化に注意）
  - [Ferramacho2009SNEvo] Ferramacho, L. D., Blanchard, A., and Zolnierowski, Y. “Constraints on CDM cosmology from galaxy power spectrum, CMB and SNIa evolution.” arXiv: `https://arxiv.org/abs/0807.4608`. Accessed: 2026-01-09.
- BAO標準定規（音響地平線 r_drag）の一次ソース値（校正スケールの目安）：
  - [Planck2018Params] Planck Collaboration. “Planck 2018 results. VI. Cosmological parameters.” arXiv: `https://arxiv.org/abs/1807.06209`. Accessed: 2026-01-09.
- Alcock-Paczynski（AP）: BAO異方性からの F_AP=D_M H / c（距離二重性とは別の幾何プローブ）：
  - [Alam2017BOSS] Alam, S., et al. “The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic Survey: cosmological analysis of the DR12 galaxy sample.” arXiv: `https://arxiv.org/abs/1607.03155`. Accessed: 2026-01-09.
    - Table 8（BAO+FS consensus; reduced covariance 10^4 c_ij）を BAO側の共分散（D_Mの相関）として使用（表の数値を転記）。
  - BOSS DR12（post-recon）相関関数 multipoles ξℓ（ℓ=0,2; 解析の一次統計）:
    - [Ross2016CorrfuncData] Ross, A. J., et al. “COMBINEDDR12 correlation function multipoles (post-reconstruction) data package.” SDSS SAS: `https://data.sdss.org/sas/dr12/boss/papers/clustering/Ross_etal_2016_COMBINEDDR12_corrfunc.zip`. Accessed: 2026-01-11.
  - BOSS DR12（pre-recon）相関関数 multipoles ξℓ（ℓ=0,2; full-shape, Satpathy et al. 2016）:
    - [Satpathy2016CorrfuncData] Satpathy, S., et al. “COMBINEDDR12 full-shape correlation function multipoles (pre-reconstruction) data package.” SDSS SAS: `https://data.sdss.org/sas/dr12/boss/papers/clustering/Satpathy_etal_2016_COMBINEDDR12_full_shape_corrfunc_multipoles.zip`. Accessed: 2026-01-11.
  - BOSS DR12 P(k) multipoles（P0/P2; Beutler et al. 公開パッケージ。窓関数 RR multipoles（Beutleretal_window_z*）も含む）:
    - [Beutler2016PowspecData] Beutler, F., et al. “COMBINEDDR12 BAO power spectrum multipoles data package.” SDSS SAS: `https://data.sdss.org/sas/dr12/boss/papers/clustering/Beutler_etal_DR12COMBINED_BAO_powspec.tar.gz`. Accessed: 2026-01-11.
  - eBOSS DR16（異方的BAO; D_M/r_d, D_H/r_d）:
    - [Bautista2020eBOSSLRG] Bautista, J. E., et al. “The Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey: Baryon Acoustic Oscillations with Luminous Red Galaxies.” arXiv: `https://arxiv.org/abs/2007.08993`. Accessed: 2026-01-10.
    - [Neveux2020eBOSSQSO] Neveux, R., et al. “The Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey: Baryon Acoustic Oscillations and Redshift Space Distortions from Quasar Clustering.” arXiv: `https://arxiv.org/abs/2007.08998`. Accessed: 2026-01-10.
    - [duMasdesBourboux2020eBOSSLya] du Mas des Bourboux, H., et al. “The Completed SDSS-IV Extended Baryon Oscillation Spectroscopic Survey: Baryon Acoustic Oscillations with Lyα Forests.” arXiv: `https://arxiv.org/abs/2007.08995`. Accessed: 2026-01-10.

## 量子（ベル / 干渉 / QED精密）

### 原子核（AME2020 / ENSDF / IAEA NDS）

- [IAEAAMDCAME2020] IAEA NDS/AMDC の AME2020 を一次質量表として採用する。Web: `https://www-nds.iaea.org/amdc/ame2020/`. Accessed: 2026-02-07.
- ローカル固定入力（キャッシュ）として AME2020 mass table を固定し、参照先と同一性（ハッシュ）は検証用資料へ集約する。[PModelVerificationMaterials]
- [IAEAENSDF] ENSDF（Evaluated Nuclear Structure Data File）を核構造 cross-check の一次出典として採用する。Web: `https://www-nds.iaea.org/relnsd/ensdf/`. Accessed: 2026-02-07.
- [IAEANDSPortal] IAEA Nuclear Data Services のポータルを核データ出典の上位入口として固定する。Web: `https://www-nds.iaea.org/`. Accessed: 2026-02-07.
- [NNDCNuDat3] NuDat 3.0 を ENSDF 系の操作的参照I/Fとして採用する。Web: `https://www.nndc.bnl.gov/nudat3/`. Accessed: 2026-02-07.
- ローカル固定入力（キャッシュ）として NuDat 3.0 由来の抽出結果を固定し、参照先と同一性（ハッシュ）は検証用資料へ集約する。[PModelVerificationMaterials]

### ベルテスト（time-tag / trial log）

- [NISTBelltestdata] NIST belltestdata（Shalm et al. 2015 の time-tag 公開データ）. Bucket: `https://s3.amazonaws.com/nist-belltestdata/`（prefix: `belldata/`）。Accessed: 2026-01-25.
- [DelftHensen2015] Delft loophole-free Bell test dataset（Hensen et al. 2015；event-ready / CHSH）。4TU: `https://data.4tu.nl/datasets/e8cf2991-3153-48ad-b67d-dfd7d7d97fd3`. Accessed: 2026-01-25.
- [DelftHensen2016Srep30289] Delft second loophole-free Bell test dataset（Hensen et al. 2016；Sci Rep 6, 30289；event-ready / CHSH）。4TU: `https://data.4tu.nl/articles/dataset/Second_loophole-free_Bell_test_with_electron_spins_separated_by_1_3_km/12694403`. Accessed: 2026-01-25.
- [Zenodo7185335] Weihs et al. 1998（photon time-tag）dataset. DOI: 10.5281/zenodo.7185335. Web: `https://zenodo.org/records/7185335`. Accessed: 2026-01-25.
- [IllinoisQIBellTestdata] Christensen et al. 2013（PRL 111, 130406；Kwiat group / photon / CH）photon time-tag dataset（CH_Bell_Data.zip）. Illinois QI: `https://research.physics.illinois.edu/QI/BellTest/`（files: `CH_Bell_Data.zip`, `data_organization.txt`）。Accessed: 2026-01-25.

### 原子スペクトル（NIST ASD）

- [NISTASDLines] NIST Atomic Spectra Database（ASD；Lines）. Web: `https://physics.nist.gov/PhysRefData/ASD/lines_form.html`. Accessed: 2026-01-25.

### 原子スペクトル（NIST AtSpec handbook）

- [NISTAtSpecHandbook] NIST “AtSpec” handbook（AtSpec.PDF）. Web: `https://www.physics.nist.gov/Pubs/AtSpec/AtSpec.PDF`. Accessed: 2026-01-26.

### 同位体質量（NIST Atomic Weights and Isotopic Compositions）

- [NISTIsotopicCompositions] NIST Atomic Weights and Isotopic Compositions（Hydrogen; relative atomic mass）. Web: `https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=H`. Accessed: 2026-01-25.

### 分子定数（NIST Chemistry WebBook）

- [NISTWebBookDiatomic] NIST Chemistry WebBook “Constants of diatomic molecules”（Huber & Herzberg compilation）. Web: `https://webbook.nist.gov/cgi/cbook.cgi?Mask=1000`. Accessed: 2026-01-25.

### 熱化学（NIST Chemistry WebBook）

- [NISTWebBookThermo] NIST Chemistry WebBook “Gas phase thermochemistry data”. Web: `https://webbook.nist.gov/cgi/cbook.cgi?Mask=1`. Accessed: 2026-01-25.

### 分光学的解離エネルギー D0（0 K）

- [ArXiv1902_09471] arXiv:1902.09471 “Benchmarking theory with an improved measurement of the ionization and dissociation energies of H2”. arXiv: `https://arxiv.org/abs/1902.09471`. Accessed: 2026-01-25.
- [ArXiv2202_00532] arXiv:2202.00532 “Improved Ionization and Dissociation Energies of the Deuterium Molecule”. arXiv: `https://arxiv.org/abs/2202.00532`. Accessed: 2026-01-25.
- [Holsch2023HD] Hölsch et al. 2023 “Ionization and dissociation energies of HD and dipole-induced g/u -symmetry breaking”. *Phys. Rev. A* **108**, 022811 (2023). DOI: 10.1103/PhysRevA.108.022811. Accessed: 2026-01-25.

### 分子遷移（一次線リスト）

- [ExoMolRACPPK] ExoMol database: H2 line list “RACPPK” (states/trans/pf). Web: `https://exomol.com/data/molecules/H2/1H2/RACPPK`. Accessed: 2026-01-25.
- [ExoMolADJSAAM] ExoMol database: HD isotopologue line list “ADJSAAM” (states/trans/pf). Web: `https://exomol.com/data/molecules/H2/1H-2H/ADJSAAM`. Accessed: 2026-01-25.
- [MOLATD2ARLSJ1999] MOLAT（OBSPM）: D2 FUV emission line lists（transition energies + A；B, B′, C±, D± → X）. Web: `https://molat.obspm.fr/index.php?page=pages/Molecules/D2/D2.php`. Accessed: 2026-01-26.
- [ARLSJ1999] Abgrall et al. 1999（D2 FUV emission; dataset description for MOLAT）. J. Phys. B: At. Mol. Opt. Phys. 32 (1999) 3813–3838.

### 重力×量子干渉 / 物質波 / 光 / 真空

- [Mannheim1996COW] Mannheim 1996（COW 理論整理）. arXiv: `https://arxiv.org/abs/gr-qc/9611037`. Accessed: 2026-01-25.
- [ColellaOverhauserWerner1975] Colella, Overhauser, Werner 1975（COW 原典）. *Phys. Rev. Lett.* **34**, 1472 (1975). DOI: 10.1103/PhysRevLett.34.1472. Accessed: 2026-01-25.
- [Mueller2007AtomInterferometer] Mueller, H., et al. 2007（原子干渉計重力計）. arXiv: `https://arxiv.org/abs/0710.3768`. Accessed: 2026-01-25.
- [ArXiv2309_14953] arXiv:2309.14953v3 “Long-distance chronometric leveling with a portable optical clock”. arXiv: `https://arxiv.org/abs/2309.14953`. Accessed: 2026-01-25.
- [Bach2012ElectronDoubleSlit] Bach, R., et al. 2012（電子二重スリット）. arXiv: `https://arxiv.org/abs/1210.6243`. Accessed: 2026-01-25.
- [Bouchendira2008AlphaRecoil] Bouchendira, R., et al. 2008（原子反跳→α）. arXiv: `https://arxiv.org/abs/0812.3139`. Accessed: 2026-01-25.
- [Gabrielse2008AlphaG2] Gabrielse, G., et al. 2008（電子 g-2 → α；独立参照）. arXiv: `https://arxiv.org/abs/0801.1134`. Accessed: 2026-01-25.
- [Pikovski2013TimeDilationDecoherence] Pikovski, I., et al. 2013（time dilation decoherence）. arXiv: `https://arxiv.org/abs/1311.1095`. Accessed: 2026-01-25.
- [Bonder2015Decoherence] Bonder, Y., et al. 2015（decoherence 議論）. arXiv: `https://arxiv.org/abs/1509.04363`. Accessed: 2026-01-25.
- [AnastopoulosHu2015Decoherence] Anastopoulos, C., and Hu, B. L. 2015（decoherence 議論）. arXiv: `https://arxiv.org/abs/1507.05828`. Accessed: 2026-01-25.
- [Pikovski2015Reply] Pikovski, I., et al. 2015（reply）. arXiv: `https://arxiv.org/abs/1509.07767`. Accessed: 2026-01-25.
- [Hasegawa2021LatticeClockDephasing] Hasegawa, Y., et al. 2021（光格子時計：重力赤方偏移→dephasing 提案）. arXiv: `https://arxiv.org/abs/2107.02405`. Accessed: 2026-01-25.
- [Kimura2004SinglePhotonInterference] Kimura, T., et al. 2004（単一光子干渉）. arXiv: `https://arxiv.org/abs/quant-ph/0403104`. Accessed: 2026-01-25.
- [ArXiv2106_03871] arXiv:2106.03871v2 “Quantum interference of identical photons from remote GaAs quantum dots”. arXiv: `https://arxiv.org/abs/2106.03871`. Accessed: 2026-01-25.
- [Zenodo6371310] Zenodo（arXiv:2106.03871 source data）. DOI: 10.5281/zenodo.6371310. Accessed: 2026-01-25.
- [Vahlbruch2007Squeezed10dB] Vahlbruch, H., et al. 2007（10 dB squeezing benchmark）. arXiv: `https://arxiv.org/abs/0706.1431`. Accessed: 2026-01-25.
- [RoyLinMohideen2000Casimir] Roy, A., Lin, C.-Y., and Mohideen, U. 2000（Casimir sphere-plane; AFM）. arXiv: `https://arxiv.org/abs/quant-ph/9906062`. Accessed: 2026-01-25.
- [IvanovKarshenboim2000LambShift] Ivanov, A. I., and Karshenboim, S. G. 2000（Lamb shift）. arXiv: `https://arxiv.org/abs/physics/0009069`. Accessed: 2026-01-25.
- [Parthey2011Hydrogen1S2S] Parthey, C. G., Matveev, A., Alnis, J., et al. 2011（Hydrogen 1S–2S absolute frequency; uncertainty budget）. arXiv: `https://arxiv.org/abs/1107.3101`. Accessed: 2026-01-26.

## ブロック中/保留

本稿では扱わない（将来、一次データが入手でき次第、検証候補として追加する）。

<!-- INTERNAL_ONLY_START -->

# データ出典（一次ソース）とローカルキャッシュ

本ドキュメントは、本論文における「どの結果が、どの一次ソースに基づくか」を固定するための一覧です。  
本リポジトリでは再現性のため、取得したデータは `data/` にキャッシュし、生成物は `output/` に固定名で保存します。

入口：
- 一般向け統一レポート（図のカタログ）：`output/summary/pmodel_public_report.html`
- Table 1（検証サマリ）：`output/summary/paper_table1_results.md`

---

## 共通ルール（このリポジトリの約束）

- 入力データ：`data/<topic>/`
- 実行スクリプト：`scripts/<topic>/`
- 生成物：`output/<topic>/`
- 一括再現（推奨）：
   - `python -B scripts/summary/run_all.py --online --jobs 2`
   - `python -B scripts/summary/run_all.py --offline --jobs 2`（キャッシュがある場合）
   - ※推奨並列数：`--jobs 2`（不安定/メモリ不足が疑われる場合のみ `--jobs 1` で切り分け）
   - 資料生成（軽量；HTMLのみ）：`cmd /c output\summary\build_materials.bat quick-nodocx`

---

## LLR（月レーザー測距）

対象：
- 月レーザー測距（LLR: Lunar Laser Ranging）の CRD Normal Point（NPT/NP2）を用いた、観測TOF（往復光時間）比較。

一次ソース：
- DGFI-TUM EDC（ILRS/SLR系の配布ミラー）：`ftp://edc.dgfi.tum.de` / `https://edc.dgfi.tum.de`

ローカルキャッシュ（入力）：
- バッチ入力マニフェスト：`data/llr/llr_edc_batch_manifest.json`（URL/sha256/保存先を含む）
- ダウンロード済みデータ：`data/llr/edc/`
- 局座標（一次ソース pos+eop）など：`data/llr/stations/`、`data/llr/pos_eop/`

スクリプト：
- バッチ評価：`scripts/llr/llr_batch_eval.py`

主要生成物：
- バッチ要約：`output/llr/batch/llr_batch_summary.json`
- 外れ値俯瞰：`output/llr/batch/llr_outliers_overview.png`

---

## Cassini（太陽会合）

対象：
- Cassini の太陽会合におけるドップラー観測量 `y(t)`（Shapiro遅延の時間微分）を、一次データ（PDS TDF）から再構成して P-model と比較。

一次ソース：
- PDS3（Cassini SCE1、CO-SS-RSS-1-SCE1-V1.0）  
  本リポジトリではダウンロード可能なミラー（NMSU atmos）を使用し、URL/sha256 をマニフェストに固定する。

ローカルキャッシュ（入力）：
- TDF マニフェスト：`data/cassini/pds_sce1/manifest_tdf.json`（base_url + 各ファイルURL/sha256）
- ODF マニフェスト：`data/cassini/pds_sce1/manifest_odf.json`

スクリプト：
- 主要比較：`scripts/cassini/cassini_fig2_overlay.py`

主要生成物：
- 代表図（観測 vs P-model）：`output/cassini/cassini_fig2_overlay_zoom10d.png`
- 残差：`output/cassini/cassini_fig2_residuals.png`
- 指標：`output/cassini/cassini_fig2_metrics.csv`

---

## Viking（太陽会合）

対象：
- Viking（Earth–Mars）の往復 Shapiro 遅延を、幾何（太陽中心・ICRF）から計算し、代表値（200–250 μs）と整合するかを確認。

一次ソース：
- NASA JPL Horizons API（天体暦ベクトル）

ローカルキャッシュ（入力）：
- Horizons キャッシュ：`output/viking/horizons_cache/`（クエリ結果の保存）

スクリプト：
- `scripts/viking/viking_shapiro_check.py`

主要生成物：
- 数値：`output/viking/viking_shapiro_result.csv`
- 図：`output/viking/viking_p_model_vs_measured_no_arrow.png`

---

## 太陽光偏向（VLBI, PPN γ）

対象：
- VLBI 等で推定された PPN パラメータ `γ`（太陽重力による光偏向の強さ）を、P-model の予測 `γ=2β-1` と比較する。

一次ソース：
- [Lebach1995][Shapiro2004VLBI][Fomalont2009]（各論文における `γ` の推定値。生データ再解析ではなく、公表値の比較として扱う）

ローカルキャッシュ（入力）：
- 観測 `γ` の代表値（一次ソースの要約）：`data/theory/solar_light_deflection_measurements.json`

スクリプト：
- `scripts/theory/solar_light_deflection.py`

主要生成物：
- 図：`output/theory/solar_light_deflection.png`（α(b) と 観測γ±σ）
- 指標：`output/theory/solar_light_deflection_metrics.json`

---

## GPS（衛星時計）

対象：
- IGS Final（CLK/SP3）を準実測として、放送暦（BRDC）および P-model による時計残差を比較。

一次ソース：
- IGS（International GNSS Service）の最終解（Final）プロダクト
- GPS 放送暦（RINEX）

ローカルキャッシュ（入力）：
- 取得元URL/sha256（固定）：`data/gps/igs_bkg_sources.json`（IGS BKG root_ftp）
- `data/gps/IGS0OPSFIN_20252740000_01D_05M_CLK.CLK`
- `data/gps/IGS0OPSFIN_20252740000_01D_15M_ORB.SP3`
- `data/gps/BRDC00IGS_R_20252740000_01D_MN.rnx`

スクリプト：
- 取得：`scripts/gps/fetch_igs_bkg.py`（例：`python -B scripts/gps/fetch_igs_bkg.py --date 2025-10-01`）
- 計算：`scripts/gps/compare_clocks.py`
- 図化：`scripts/gps/plot.py`

主要生成物：
- 指標：`output/gps/gps_compare_metrics.json`
- 図：`output/gps/gps_rms_compare.png` ほか

---

## Mercury（近日点移動）

対象：
- P-model の運動方程式で水星軌道を数値積分し、近日点移動（角秒/世紀）を算出して代表値と比較。

入力：
- 内部定数（G, M_sun, 初期条件など。外部データ依存は最小）

スクリプト：
- `scripts/mercury/mercury_precession_v3.py`

主要生成物：
- 指標：`output/mercury/mercury_precession_metrics.json`

---

## EHT（ブラックホール影）

対象：
- EHT の公表するリング直径（M87*, Sgr A*）を、P-model の最小延長（影直径係数）と比較し、差分予測（係数差）を示す。

一次ソース（公表値）：
- `data/eht/eht_black_holes.json` の `sources` を参照（URLを固定）

ローカルキャッシュ（入力）：
- `data/eht/eht_black_holes.json`（リング直径・質量・距離と、出典URLを固定）
- Sgr A* 質量・距離（GRAVITY 2019）：`data/eht/sources/arxiv_1904.05721.pdf`
- S2 星軌道（GRAVITY 2018/2020）：`data/eht/sources/arxiv_1807.09409.pdf` / `data/eht/sources/arxiv_2004.07187.pdf`
- 参照マニフェスト（URL/sha256固定）：`output/eht/gravity_s2_sources_manifest.json`

スクリプト：
- `scripts/eht/eht_shadow_compare.py`
  - M87* multi-epoch（2017/2018）整合：`scripts/eht/eht_m87_persistent_shadow_metrics.py`

主要生成物：
- 比較：`output/eht/eht_shadow_compare.json`
- 図：`output/eht/eht_shadow_compare.png` / `output/eht/eht_shadow_differential.png`
  - 図（M87* multi-epoch）：`output/eht/eht_m87_persistent_shadow_ring_diameter.png`

---

## 連星パルサー（軌道減衰）

対象：
- 連星（NS-NS / NS-WD）の軌道周期減少について、一次ソースの軌道要素・質量・外部補正後 `Pdot_b` から四重極予測を再計算し、「観測/四重極」の一致度 `R` と追加放射上限（概算）を整理する。
  - NS-NS：PSR B1913+16, PSR J0737-3039A/B
  - NS-WD：PSR J1738+0333, PSR J0348+0432

一次ソース：
- [WeisbergHuang2016][Kramer2021][Freire2012J1738][Antoniadis2013J0348]

ローカルキャッシュ（入力）：
- `data/pulsar/binary_pulsar_orbital_decay.json`（R値・不確かさ・DOIを固定）

スクリプト：
- `scripts/pulsar/binary_pulsar_orbital_decay.py`

主要生成物：
- 図：`output/pulsar/binary_pulsar_orbital_decay.png`
- 指標：`output/pulsar/binary_pulsar_orbital_decay_metrics.json`

---

## 重力波（GW150914）

対象：
- GW150914 の公開 strain から、bandpass→位相→瞬時周波数 `f(t)` を抽出し、四重極チャープ則（Newton近似）の整合性を簡易的に可視化する（本稿では“入口”として位置づけ、厳密推定は別途）。

一次ソース：
- [GWOSC]（event API / strain）

ローカルキャッシュ（入力）：
- `data/gw/gw150914/`（URL/sha256固定: `data/gw/gw150914/gw150914_sources.json`）

スクリプト：
- `scripts/gw/gw150914_chirp_phase.py`

主要生成物：
- 図：`output/gw/gw150914_chirp_phase.png`
- 指標：`output/gw/gw150914_chirp_phase_metrics.json`

---

## ブロック中/保留

（本稿では、ブロック中テーマは扱わない。詳細は `doc/BLOCKED.md` に集約する。）
<!-- INTERNAL_ONLY_END -->
