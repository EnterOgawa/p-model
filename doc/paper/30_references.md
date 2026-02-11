この文書は本論文における参考文献リストの作業用です。  
現時点では「どの主張/観測が、どの一次ソース/代表文献に紐づくか」を見失わないことを優先し、正式な書式（BibTeX/出版社指定）への整形は後段で行う。  
Webページ等のリソースは Accessed（参照日）を併記する。

<!-- INTERNAL_ONLY_START -->
入口（データ一次ソースの固定）：`doc/paper/20_data_sources.md`
<!-- INTERNAL_ONLY_END -->

---

## 0) 検証用資料（Verification Materials）

- [PModelVerificationMaterials] P-model Verification Materials（再現コマンド、固定アーティファクト一覧、監査ゲート、ハッシュ/マニフェスト等の補助資料）。Web: `https://github.com/EnterOgawa/p-model`（Releases: `https://github.com/EnterOgawa/p-model/releases`）。Accessed: 2026-02-11.

---

## 1) 一般相対論 / PPN / 標準式（背景）

- [Will2014] Will, C. M. “The Confrontation between General Relativity and Experiment.” *Living Reviews in Relativity* **17**, 4 (2014). DOI: 10.12942/lrr-2014-4.
- [Ashby2003] Ashby, N. “Relativity in the Global Positioning System.” *Living Reviews in Relativity* **6**, 1 (2003). DOI: 10.12942/lrr-2003-1.
- [Shapiro1964] Shapiro, I. I. “Fourth Test of General Relativity.” *Physical Review Letters* **13**, 789-791 (1964). DOI: 10.1103/PhysRevLett.13.789.
- [BornWolf1999] Born, M., and Wolf, E. *Principles of Optics* (7th ed., expanded). Cambridge University Press (1999).
- 太陽光偏向（VLBI, PPN γ）
  - [Lebach1995] Lebach, D. E., et al. “Measurement of the Solar Gravitational Deflection of Radio Waves Using Very-Long-Baseline Interferometry.” *Physical Review Letters* **75**, 1439 (1995). DOI: 10.1103/PhysRevLett.75.1439.
  - [Shapiro2004VLBI] Shapiro, I. I., et al. “Measurement of the Solar Gravitational Deflection of Radio Waves Using Geodetic Very-Long-Baseline Interferometry Data, 1979–1999.” *Physical Review Letters* **92**, 121101 (2004). DOI: 10.1103/PhysRevLett.92.121101.
  - [Fomalont2009] Fomalont, E. B., et al. “Measurement of the Solar Gravitational Deflection of Radio Waves in Agreement with General Relativity.” arXiv:0904.3992 (2009).

---

## 2) Cassini（太陽会合・Shapiro/Doppler）

- [Bertotti2003] Bertotti, B., Iess, L., and Tortora, P. “A test of general relativity using radio links with the Cassini spacecraft.” *Nature* **425**, 374-376 (2003). DOI: 10.1038/nature01997.
- [PDSCassiniSCE1] NASA Planetary Data System (PDS). “CASSINI RSS RAW DATA SET - SCE1 V1.0” (CO-SS-RSS-1-SCE1-V1.0). DOI: 10.17189/ske1-av41. Landing page: `https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=CO-SS-RSS-1-SCE1-V1.0`. Accessed: 2026-01-03.
<!-- INTERNAL_ONLY_START -->
  - Mirror used in this repo: `https://atmos.nmsu.edu/pdsd/archive/data/co-ss-rss-1-sce1-v10/`.
<!-- INTERNAL_ONLY_END -->

---

## 3) Viking（太陽会合・Shapiro 遅延）

- Viking の太陽会合 Shapiro 遅延（代表値の根拠）
  - [Shapiro1977] Shapiro, I. I., et al. “Fourth test of general relativity: New radar results.” *Journal of Geophysical Research* **82**(28), 4329-4334 (1977). DOI: 10.1029/JS082i028p04329.
  - [Reasenberg1979] Reasenberg, R. D., et al. “Viking relativity experiment: Verification of signal retardation by solar gravity.” *The Astrophysical Journal Letters* **234**, L219-L221 (1979). DOI: 10.1086/183144.
- 幾何の一次ソース（天体暦ベクトル）
  - [JPLHorizons] NASA/JPL SSD. “HORIZONS system.” Web: `https://ssd.jpl.nasa.gov/horizons/` (API: `https://ssd.jpl.nasa.gov/api/horizons.api`). Accessed: 2026-01-03.

---

## 4) Mercury（近日点移動）

- [NobiliWill1986] Nobili, A. M., and Will, C. M. “The real value of the perihelion advance of Mercury.” *Nature* **320**, 39-41 (1986). DOI: 10.1038/320039a0.

---

## 5) GPS（衛星時計）

- [Dow2009] Dow, J. M., Neilan, R. E., and Rizos, C. “The International GNSS Service in a changing landscape of Global Navigation Satellite Systems.” *Journal of Geodesy* **83**, 191-198 (2009). DOI: 10.1007/s00190-008-0300-3.
- [IGSBKGRootFTP] IGS Public Data (BKG mirror) “root_ftp/IGS” (products/BRDC/CLK/SP3). Web: `https://igs.bkg.bund.de/root_ftp/IGS/`. Accessed: 2026-01-03.

---

## 6) 重力赤方偏移（GP-A / Galileo 5/6）

- [Vessot1980] Vessot, R. F. C., et al. “Test of Relativistic Gravitation with a Space-Borne Hydrogen Maser.” *Physical Review Letters* **45**, 2081 (1980). DOI: 10.1103/PhysRevLett.45.2081.  
  NASA NTRS copy: `https://ntrs.nasa.gov/citations/19810000898`. Accessed: 2026-01-04.
- [Delva2018] Delva, P., et al. “A gravitational redshift test using eccentric Galileo satellites.” *Physical Review Letters* **121**, 231101 (2018). DOI: 10.1103/PhysRevLett.121.231101. arXiv:1810.12191. Accessed: 2026-01-04.

---

## 6b) 回転（フレームドラッグ：GP-B / LAGEOS）

- [Everitt2011GPB] Everitt, C. W. F., et al. “Gravity Probe B: Final Results of a Space Experiment to Test General Relativity.” *Physical Review Letters* **106**, 221101 (2011). DOI: 10.1103/PhysRevLett.106.221101. arXiv:1105.3456. Accessed: 2026-01-08.
- [CiufoliniPavlis2004] Ciufolini, I., and Pavlis, E. C. “A confirmation of the general relativistic prediction of the Lense-Thirring effect.” *Nature* **431**, 958-960 (2004). DOI: 10.1038/nature03007. Accessed: 2026-01-08.

---

## 7) LLR（月レーザー測距）

- 仕様（CRDフォーマット）
  - [ILRSCRD] International Laser Ranging Service (ILRS). “CRD - Consolidated Ranging Data Format.” Web: `https://ilrs.gsfc.nasa.gov/data_and_products/formats/crd.html`. Accessed: 2026-01-03.
- 一次データ配布（ILRSミラー）
  - [EDC] DGFI-TUM. “EDC - Eurolas Data Center (ILRS data mirror).” Web: `https://edc.dgfi.tum.de/`. Accessed: 2026-01-03.
- ILRS の概説（SLR/LLRの運用・データ体系）
  - [Pearlman2019] Pearlman, M. R., et al. “The ILRS: approaching 20 years and planning for the future.” *Journal of Geodesy* **93**, 2161-2180 (2019). DOI: 10.1007/s00190-019-01241-1.
- 大気遅延（対流圏）
  - [Saastamoinen1972] Saastamoinen, J. “Atmospheric correction for the troposphere and stratosphere in radio ranging of satellites.” In: *The Use of Artificial Satellites for Geodesy* (Geophysical Monograph Series, Vol. 15). AGU (1972).
  - [Marini1972] Marini, J. W. “Correction of satellite tracking data for an arbitrary tropospheric profile.” *Radio Science* **7**(2), 223-231 (1972). DOI: 10.1029/RS007i002p00223.
  - [Niell1996] Niell, A. E. “Global mapping functions for the atmosphere delay at radio wavelengths.” *Journal of Geophysical Research: Solid Earth* **101**(B2), 3227-3246 (1996). DOI: 10.1029/95JB03048.
- 地球自転・潮汐など（座標系/補正モデルの標準）
- [IERS2010] Petit, G., and Luzum, B. (eds.). *IERS Conventions (2010)*. IERS Technical Note No. 36 (2010).

---

## 8) EHT（ブラックホール影）

係数差の議論では「リング≒影」の近似（系統不確かさ κ）を明示し、過剰な結論を避ける（詳細は補足資料）。
<!-- INTERNAL_ONLY_START -->
詳細メモ：`doc/eht/README.md`
<!-- INTERNAL_ONLY_END -->

- M87*（EHT, Paper I）
  - [EHTM87I2019] Event Horizon Telescope Collaboration. “First M87 Event Horizon Telescope Results. I. The Shadow of the Supermassive Black Hole.” *The Astrophysical Journal Letters* **875**, L1 (2019). DOI: 10.3847/2041-8213/ab0ec7.
- M87*（EHT, Paper VI：質量・距離・影スケール）
  - [EHTM87VI2019] Event Horizon Telescope Collaboration. “First M87 Event Horizon Telescope Results. VI. The Shadow and Mass of the Central Black Hole.” *The Astrophysical Journal Letters* **875**, L6 (2019). DOI: 10.3847/2041-8213/ab1141.
- M87*（EHT 2018, Paper I：影の持続性）
  - [EHTM87PersistentI2024] Event Horizon Telescope Collaboration. “The persistent shadow of the supermassive black hole of M87. I. Observations, calibration, imaging, and analysis.” *Astronomy & Astrophysics* **681**, A79 (2024). DOI: 10.1051/0004-6361/202347932.
- M87*（EHT 2018, Paper II：理論解釈／GRMHD比較）
  - [EHTM87PersistentII2025] Event Horizon Telescope Collaboration. “The persistent shadow of the supermassive black hole of M87. II. Model comparisons and theoretical interpretations.” *Astronomy & Astrophysics* **693**, A265 (2025). DOI: 10.1051/0004-6361/202451296.
- Sgr A*（EHT, Paper I）
  - [EHTSgrAI2022] Event Horizon Telescope Collaboration. “First Sagittarius A* Event Horizon Telescope Results. I.” *The Astrophysical Journal Letters* **930**, L12 (2022). DOI: 10.3847/2041-8213/ac6674.
- Sgr A*（EHT, Paper V：放射モデル検証）
  - [EHTSgrAV2022] Event Horizon Telescope Collaboration, et al. “First Sagittarius A* Event Horizon Telescope Results. V. Testing Astrophysical Models of the Galactic Center Black Hole.” arXiv:2311.09478.
- リングと影の対応（厚いディスク／吸収の影響）
  - [Vincent2022ThickDisk] Vincent, F. H., et al. “Thick accretion discs and photon rings: the case for not-so-black shadows.” *Astronomy & Astrophysics* **667**, A170 (2022). DOI: 10.1051/0004-6361/202244339. arXiv:2206.12066.
- photon ring（干渉計シグナチャー；将来space-VLBIの理論基盤）
  - [Johnson2020PhotonRing] Johnson, M. D., et al. “Universal interferometric signatures of a black hole’s photon ring.” *Science Advances* **6**(12), eaaz1310 (2020). DOI: 10.1126/sciadv.aaz1310. arXiv:1907.04329.
- photon ring形状によるGRテスト（形状指標の頑健性）
  - [Gralla2020PhotonRingShape] Gralla, S. E., Holz, D. E., and Wald, R. M. “Black hole photon ring shape as a test of general relativity.” *Physical Review D* **102**, 124004 (2020). DOI: 10.1103/PhysRevD.102.124004. arXiv:2008.03879.
- Sgr A* 質量・距離（幾何学的拘束）
  - [Gravity2019] GRAVITY Collaboration, et al. “A geometric distance measurement to the Galactic center black hole with 0.3% uncertainty.” *Astronomy & Astrophysics* **625**, L10 (2019). DOI: 10.1051/0004-6361/201935656.
- Sgr A* 強場（S2 星軌道：重力赤方偏移/Schwarzschild 歳差）
  - [Gravity2018S2Redshift] GRAVITY Collaboration, et al. “Detection of the gravitational redshift in the orbit of the star S2 near the Galactic centre massive black hole.” arXiv:1807.09409 (2018). Accessed: 2026-01-22.
  - [Gravity2020S2Precession] GRAVITY Collaboration, et al. “Detection of the Schwarzschild precession in the orbit of the star S2 near the Galactic centre massive black hole.” arXiv:2004.07187 (2020). Accessed: 2026-01-22.
- Sgr A* 近赤外フラックス分布（2.2 μm; 閾値の保守性評価に利用）
  - [Gravity2020Flux] GRAVITY Collaboration, et al. “The Flux Distribution of Sgr A*.” arXiv:2004.07185.
- Sgr A* mm light curves（230 GHz; modulation index M3 の観測側前提）
  - [Wielgus2022] Wielgus, M., et al. “Millimeter light curves of Sagittarius A* observed during the 2017 Event Horizon Telescope campaign.” arXiv:2207.06829.
- Sgr A* の散乱（系統要因の一次ソース）
  - [Johnson2018] Johnson, M. D., et al. “The Scattering and Intrinsic Structure of Sagittarius A* at Radio Wavelengths.” *The Astrophysical Journal* **865**(2), 104 (2018). arXiv:1808.08966.
  - [Psaltis2018] Psaltis, D., et al. “A Model for Anisotropic Interstellar Scattering and its Application to Sgr A*.” arXiv:1805.01242 (2018).
  - [Zhu2018] Zhu, Z., Johnson, M. D., and Narayan, R. “Testing General Relativity with the Black Hole Shadow Size and Asymmetry of Sagittarius A*: Limitations from Interstellar Scattering.” arXiv:1811.02079 (2018).

---

## 9) 二重パルサー / 重力波（放射）

- [PetersMathews1963] Peters, P. C., and Mathews, J. “Gravitational Radiation from Point Masses in a Keplerian Orbit.” *Physical Review* **131**, 435 (1963). DOI: 10.1103/PhysRev.131.435.
- [Peters1964] Peters, P. C. “Gravitational Radiation and the Motion of Two Point Masses.” *Physical Review* **136**, B1224 (1964). DOI: 10.1103/PhysRev.136.B1224.
- [WeisbergHuang2016] Weisberg, J. M., and Huang, Y. “Relativistic Measurements from Timing the Binary Pulsar PSR B1913+16.” *The Astrophysical Journal* **829**, 55 (2016). DOI: 10.3847/0004-637X/829/1/55.
- [Kramer2021] Kramer, M., et al. “Strong-field Gravity Tests with the Double Pulsar.” *Physical Review X* **11**, 041050 (2021). DOI: 10.1103/PhysRevX.11.041050. arXiv:2112.06795.
- [Freire2012J1738] Freire, P. C. C., et al. “The relativistic pulsar-white dwarf binary PSR J1738+0333 II. The most stringent test of scalar-tensor gravity.” *Monthly Notices of the Royal Astronomical Society* (2012). DOI: 10.1111/j.1365-2966.2012.21253.x. arXiv:1205.1450. Accessed: 2026-01-08.
- [Antoniadis2013J0348] Antoniadis, J., et al. “A Massive Pulsar in a Compact Relativistic Binary.” *Science* **340**, 448-450 (2013). DOI: 10.1126/science.1233232. arXiv:1304.6875. Accessed: 2026-01-08.
- [Abbott2016GW150914] Abbott, B. P., et al. “Observation of Gravitational Waves from a Binary Black Hole Merger.” *Physical Review Letters* **116**, 061102 (2016). DOI: 10.1103/PhysRevLett.116.061102.
- [GWOSC] Gravitational Wave Open Science Center (GWOSC). Event API / strain data. Web: `https://gwosc.org/eventapi/`. Accessed: 2026-01-06.

---

## 10) 宇宙論（距離二重性 / DDR）

- [Martinelli2021DDR] Martinelli, M., et al. “Euclid: Forecast constraints on the cosmic distance duality relation with complementary external probes.” *Astronomy & Astrophysics* (2021). DOI: 10.1051/0004-6361/202039637. arXiv:2007.16153. Accessed: 2026-01-08.
- [Holanda2012ClustersDDR] Holanda, R. F. L., Lima, J. A. S., and Ribeiro, M. B. “Probing the cosmic distance duality relation with the Sunyaev-Zel’dovich effect, X-ray observations and supernovae Ia.” *Astronomy & Astrophysics* **538**, A83 (2012). DOI: 10.1051/0004-6361/201118343. arXiv:1104.3753. Accessed: 2026-01-09.
- [LiWuYu2011DDR] Li, Z., Wu, P., and Yu, H. “Cosmological-model-independent tests for the distance-duality relation from Galaxy Clusters and Type Ia Supernova.” *The Astrophysical Journal Letters* **729**, L14 (2011). DOI: 10.1088/2041-8205/729/1/L14. arXiv:1101.5255. Accessed: 2026-01-09.
- [Holanda2016SGLDDR] Holanda, R. F. L., Busti, V. C., Lima, F. S., and Alcaniz, J. S. “Probing the distance-duality relation with high-z data.” arXiv:1611.09426 (2016). Accessed: 2026-01-10.
- [LiLin2017RadioDDR] Li, X., and Lin, H.-N. “Testing the distance duality relation using type-Ia supernovae and ultra-compact radio sources.” arXiv:1710.11361 (2017). Accessed: 2026-01-10.
- [LvXia2016CDDR] Lv, M.-Z., and Xia, J.-Q. “Constraints on Cosmic Distance Duality Relation from Cosmological Observations.” *Physics of the Dark Universe* (2016). DOI: 10.1016/j.dark.2016.06.003. arXiv:1606.08102. Accessed: 2026-01-09.
- [Avgoustidis2010Opacity] Avgoustidis, A., Burrage, C., Redondo, J., Verde, L., and Jimenez, R. “Constraints on cosmic opacity and beyond the standard model physics from cosmological distance measurements.” *JCAP* (2010). DOI: 10.1088/1475-7516/2010/10/024. arXiv:1004.2053. Accessed: 2026-01-09.
- [HolandaCarvalhoAlcaniz2014Opacity] Holanda, R. F. L., Carvalho, J. C., and Alcaniz, J. S. “Model-independent constraints on the cosmic opacity.” arXiv:1207.1694 (2014). Accessed: 2026-01-10.
- [WangWeiLiXiaZhu2017CurvatureOpacity] Wang, G.-J., Wei, J.-J., Li, Z.-X., Xia, J.-Q., and Zhu, Z.-H. “Model-independent Constraints on Cosmic Curvature and Opacity.” arXiv:1709.07258 (2017). Accessed: 2026-01-10.
- [QiCaoPanLi2019GWOpacity] Qi, J.-Z., Cao, S., Pan, Y., and Li, J. “Cosmic opacity: cosmological-model-independent tests from gravitational waves and Type Ia Supernova.” arXiv:1902.01702 (2019). Accessed: 2026-01-10.
- [Abbott2017GW170817] Abbott, B. P., et al. (LIGO Scientific Collaboration and Virgo Collaboration). “GW170817: Observation of Gravitational Waves from a Binary Neutron Star Inspiral.” *Physical Review Letters* **119**, 161101 (2017). DOI: 10.1103/PhysRevLett.119.161101. arXiv:1710.05832. Accessed: 2026-01-10.
- [Hjorth2017NGC4993Distance] Hjorth, J., et al. “The distance to NGC 4993: the host galaxy of the gravitational-wave event GW170817.” arXiv:1710.05856 (2017). Accessed: 2026-01-10.
- [Cantiello2018NGC4993SBF] Cantiello, M., et al. “A precise distance to the host galaxy of the binary neutron star merger GW170817 using surface brightness fluctuations.” arXiv:1801.06080 (2018). Accessed: 2026-01-10.
- [LiaoAvgoustidisLi2015Transparency] Liao, K., Avgoustidis, A., and Li, Z. “Is the Universe Transparent?” arXiv:1512.01861 (2015). Accessed: 2026-01-09.
- [NairJhinganJain2013Transparency] Nair, R., Jhingan, S., and Jain, D. “Cosmic distance duality and cosmic transparency.” arXiv:1210.2642 (2013). Accessed: 2026-01-09.
- [LubinSandage2001Tolman] Lubin, L. M., and Sandage, A. “The Tolman Surface Brightness Test for the Reality of the Expansion. IV. A Measurement of the Tolman Signal and the Luminosity Evolution of Early-Type Galaxies.” *The Astronomical Journal* **122**(3), 1084-1103 (2001). DOI: 10.1086/322134. arXiv:astro-ph/0106566. Accessed: 2026-01-08.
- [Blondin2008TimeDilation] Blondin, S., et al. “Time Dilation in Type Ia Supernova Spectra at High Redshift.” Accepted for publication in *The Astrophysical Journal* (2008). arXiv:0804.3595. Accessed: 2026-01-09.
- [Avgoustidis2011CMBTz] Avgoustidis, A., et al. “Constraints on the CMB temperature-redshift dependence from SZ and distance measurements.” arXiv:1112.1862 (2011). Accessed: 2026-01-09.
- [Scolnic2018Pantheon] Scolnic, D. M., et al. “The Complete Light-curve Sample of Spectroscopically Confirmed SNe Ia from Pan-STARRS1 and Cosmological Constraints from the Combined Pantheon Sample.” *The Astrophysical Journal* (2018). arXiv:1710.00845. Accessed: 2026-01-09.
- [Brout2022PantheonPlus] Brout, D., et al. “The Pantheon+ Analysis: The Full Data Set and Light-curve Release.” arXiv:2202.04077 (2022). Accessed: 2026-01-09.
- [Linden2009SNEvo] Linden, S., et al. “Cosmological parameter extraction and biases from type Ia supernova magnitude evolution.” arXiv:0907.4495 (2009). Accessed: 2026-01-09.
- [Kumar2021SNEvoCCDR] Kumar, D., Rana, A., Jain, D., Mahajan, S., Mukherjee, A., and Holanda, R. F. L. “A non-parametric test of variability of Type Ia supernovae luminosity and CDDR.” arXiv:2107.04784 (2021). Accessed: 2026-01-09.
- [Hu2023SNEvoSGLS] Hu, J., Hu, J.-P., Li, Z., Zhao, W., and Chen, J. “Calibrating the effective magnitudes of type Ia supernovae with a model-independent method.” arXiv:2309.17163 (2023). Accessed: 2026-01-09.
- [Kang2020SNEvoAgeHR] Kang, Y., Lee, Y.-W., Kim, Y.-L., Chung, C., and Ree, C. H. “Early-type Host Galaxies of Type Ia Supernovae. II. Evidence for Luminosity Evolution in Supernova Cosmology.” arXiv:1912.04903 (2020). Accessed: 2026-01-10.
- [Dainotti2021PantheonH0Evo] Dainotti, M. G., De Simone, B., Schiavone, T., Montani, G., Rinaldi, E., and Lambiase, G. “On the Hubble Constant Tension in the SNe Ia Pantheon Sample.” arXiv:2103.02117 (2021). Accessed: 2026-01-09.
- [Ferramacho2009SNEvo] Ferramacho, L. D., Blanchard, A., and Zolnierowski, Y. “Constraints on CDM cosmology from galaxy power spectrum, CMB and SNIa evolution.” arXiv:0807.4608 (2009). Accessed: 2026-01-09.
- [Planck2018Params] Planck Collaboration. “Planck 2018 results. VI. Cosmological parameters.” arXiv:1807.06209 (2018). Accessed: 2026-01-09.
- [Alam2017BOSS] Alam, S., et al. “The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic Survey: cosmological analysis of the DR12 galaxy sample.” *Monthly Notices of the Royal Astronomical Society* (2017). arXiv:1607.03155. Accessed: 2026-01-09.
- [Ross2016CorrfuncData] Ross, A. J., et al. “COMBINEDDR12 correlation function multipoles (post-reconstruction) data package.” SDSS SAS (DR12 / BOSS / papers / clustering). Web: `https://data.sdss.org/sas/dr12/boss/papers/clustering/Ross_etal_2016_COMBINEDDR12_corrfunc.zip`. Accessed: 2026-01-11.
- [Satpathy2016CorrfuncData] Satpathy, S., et al. “COMBINEDDR12 full-shape correlation function multipoles (pre-reconstruction) data package.” SDSS SAS (DR12 / BOSS / papers / clustering). Web: `https://data.sdss.org/sas/dr12/boss/papers/clustering/Satpathy_etal_2016_COMBINEDDR12_full_shape_corrfunc_multipoles.zip`. Accessed: 2026-01-11.
- [Beutler2016PowspecData] Beutler, F., et al. “COMBINEDDR12 BAO power spectrum multipoles data package.” SDSS SAS (DR12 / BOSS / papers / clustering). Web: `https://data.sdss.org/sas/dr12/boss/papers/clustering/Beutler_etal_DR12COMBINED_BAO_powspec.tar.gz`. Accessed: 2026-01-11.
- [Bautista2020eBOSSLRG] Bautista, J. E., et al. “The Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey: Baryon Acoustic Oscillations with Luminous Red Galaxies.” arXiv:2007.08993 (2020). Accessed: 2026-01-10.
- [Neveux2020eBOSSQSO] Neveux, R., et al. “The Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey: Baryon Acoustic Oscillations and Redshift Space Distortions from Quasar Clustering.” arXiv:2007.08998 (2020). Accessed: 2026-01-10.
- [duMasdesBourboux2020eBOSSLya] du Mas des Bourboux, H., et al. “The Completed SDSS-IV Extended Baryon Oscillation Spectroscopic Survey: Baryon Acoustic Oscillations with Lyα Forests.” arXiv:2007.08995 (2020). Accessed: 2026-01-10.

---

## 11) 量子（ベル / 干渉 / QED精密）

### ベルテスト（一次データ / 公開データセット）

- [NISTBelltestdata] NIST belltestdata（Shalm et al. 2015 の time-tag 公開データ）. Bucket: `https://s3.amazonaws.com/nist-belltestdata/`（prefix: `belldata/`）。Accessed: 2026-01-25.
- [DelftHensen2015] Delft loophole-free Bell test dataset（Hensen et al. 2015；event-ready / CHSH）。4TU: `https://data.4tu.nl/datasets/e8cf2991-3153-48ad-b67d-dfd7d7d97fd3`. Accessed: 2026-01-25.
- [DelftHensen2016Srep30289] Delft second loophole-free Bell test dataset（Hensen et al. 2016；Sci Rep 6, 30289；event-ready / CHSH）。4TU: `https://data.4tu.nl/articles/dataset/Second_loophole-free_Bell_test_with_electron_spins_separated_by_1_3_km/12694403`. Accessed: 2026-01-25.
- [Zenodo7185335] Weihs et al. 1998（photon time-tag）dataset. DOI: 10.5281/zenodo.7185335. Web: `https://zenodo.org/records/7185335`. Accessed: 2026-01-25.
- [IllinoisQIBellTestdata] Christensen et al. 2013（PRL 111, 130406；Kwiat group / photon / CH）photon time-tag dataset（CH_Bell_Data.zip）. Illinois QI: `https://research.physics.illinois.edu/QI/BellTest/`（files: `CH_Bell_Data.zip`, `data_organization.txt`）。Accessed: 2026-01-25.

### ベルテスト（一次：論文）

- [Giustina2015PRL250401] Giustina, M., et al. “Significant-Loophole-Free Test of Bell’s Theorem with Entangled Photons.” *Phys. Rev. Lett.* **115**, 250401 (2015). DOI: 10.1103/PhysRevLett.115.250401. arXiv:1511.03190. Accessed: 2026-01-25.

### ベルテスト（解析・批判 / selection / time-coincidence）

- [AdenierKhrennikov2007FairSampling] Adenier, G., Khrennikov, A. Yu. “Is the Fair Sampling Assumption supported by EPR Experiments?” *Journal of Physics B: Atomic, Molecular and Optical Physics* **40** (2007) 131–141. DOI: 10.1088/0953-4075/40/1/012. arXiv: `https://arxiv.org/abs/quant-ph/0606122`. Accessed: 2026-01-28.
- [MichielsenDeRaedt2013EventByEventBell] Michielsen, K., De Raedt, H. “Event-by-event simulation of experiments to create entanglement and violate Bell inequalities.” arXiv: `https://arxiv.org/abs/1312.6360` (2013). Accessed: 2026-01-28.
- [DeRaedtMichielsenHess2016IrrelevanceBell] De Raedt, H., Michielsen, K., Hess, K. “Irrelevance of Bell’s Theorem for experiments involving correlations in space and time: a specific loophole-free computer-example.” arXiv: `https://arxiv.org/abs/1605.05237` (2016). Accessed: 2026-01-28.
- [Larsson2014LoopholesBellTests] Larsson, J.-Å. “Loopholes in Bell inequality tests of local realism.” *Journal of Physics A: Mathematical and Theoretical* **47**, 424003 (2014). DOI: 10.1088/1751-8113/47/42/424003. Accessed: 2026-01-28.

### 核構造（pairing / OES）

- [BohrMottelson1969] Bohr, A., and Mottelson, B. R. *Nuclear Structure, Vol. I: Single-Particle Motion*. W. A. Benjamin (1969).

### 核データ（AME2020 / ENSDF / IAEA NDS）

- [IAEAAMDCAME2020] IAEA Nuclear Data Services (NDS), Atomic Mass Data Center (AMDC). “AME2020 atomic mass evaluation (mass_1.mas20.txt).” Web: `https://www-nds.iaea.org/amdc/ame2020/`. Accessed: 2026-02-07.
- [IAEAENSDF] IAEA Nuclear Data Services (NDS). “Evaluated Nuclear Structure Data File (ENSDF).” Web: `https://www-nds.iaea.org/relnsd/ensdf/`. Accessed: 2026-02-07.
- [IAEANDSPortal] IAEA Nuclear Data Services portal. Web: `https://www-nds.iaea.org/`. Accessed: 2026-02-07.
- [NNDCNuDat3] NNDC (BNL). “NuDat 3.0.” Web: `https://www.nndc.bnl.gov/nudat3/`. Accessed: 2026-02-07.

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
- [Holsch2023HD] Hölsch et al. “Ionization and dissociation energies of HD and dipole-induced g/u -symmetry breaking.” *Phys. Rev. A* **108**, 022811 (2023). DOI: 10.1103/PhysRevA.108.022811.

### 分子遷移（一次線リスト）

- [ExoMolRACPPK] ExoMol database: H2 line list “RACPPK” (states/trans/pf). Web: `https://exomol.com/data/molecules/H2/1H2/RACPPK`. Accessed: 2026-01-25.
- [ExoMolADJSAAM] ExoMol database: HD isotopologue line list “ADJSAAM” (states/trans/pf). Web: `https://exomol.com/data/molecules/H2/1H-2H/ADJSAAM`. Accessed: 2026-01-25.
- [MOLATD2ARLSJ1999] MOLAT（OBSPM）: D2 FUV emission line lists（transition energies + A；B, B′, C±, D± → X）. Web: `https://molat.obspm.fr/index.php?page=pages/Molecules/D2/D2.php`. Accessed: 2026-01-26.
- [ARLSJ1999] Abgrall, H., Roueff, E., Liu, X., Shemansky, D. E., and James, G. K. “High-resolution far ultraviolet emission spectra of electron excited molecular deuterium.” *J. Phys. B: At. Mol. Opt. Phys.* **32**, 3813–3838 (1999).

### 重力×量子干渉 / 物質波 / 光 / 真空

- [Mannheim1996COW] Mannheim 1996（arXiv:gr-qc/9611037）. Accessed: 2026-01-25.
- [ColellaOverhauserWerner1975] Colella, R., Overhauser, A. W., and Werner, S. A. “Observation of gravitationally induced quantum interference.” *Phys. Rev. Lett.* **34**, 1472 (1975). DOI: 10.1103/PhysRevLett.34.1472.
- [Mueller2007AtomInterferometer] Mueller et al. 2007（arXiv:0710.3768）. Accessed: 2026-01-25.
- [ArXiv2309_14953] arXiv:2309.14953v3 “Long-distance chronometric leveling with a portable optical clock”. arXiv: `https://arxiv.org/abs/2309.14953`. Accessed: 2026-01-25.
- [Bach2012ElectronDoubleSlit] Bach et al. 2012（arXiv:1210.6243v1）. Accessed: 2026-01-25.
- [Bouchendira2008AlphaRecoil] Bouchendira et al. 2008（arXiv:0812.3139v1）. Accessed: 2026-01-25.
- [Gabrielse2008AlphaG2] Gabrielse et al. 2008（arXiv:0801.1134v2）. Accessed: 2026-01-25.
- [Rosenband2008AlHgClockAlpha] Rosenband, T., et al. “Frequency Ratio of Al+ and Hg+ Single-Ion Optical Clocks; Metrology at the 17th Decimal Place.” *Science* **319**, 1808–1812 (2008). DOI: 10.1126/science.1154622. Web (NIST PDF): `https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=50655`. Accessed: 2026-01-28.
- [Pikovski2013TimeDilationDecoherence] Pikovski et al. 2013（arXiv:1311.1095v2）. Accessed: 2026-01-25.
- [Bonder2015Decoherence] Bonder et al. 2015（arXiv:1509.04363v3）. Accessed: 2026-01-25.
- [AnastopoulosHu2015Decoherence] Anastopoulos & Hu 2015（arXiv:1507.05828v5）. Accessed: 2026-01-25.
- [Pikovski2015Reply] Pikovski et al. 2015（reply; arXiv:1509.07767v1）. Accessed: 2026-01-25.
- [Hasegawa2021LatticeClockDephasing] Hasegawa et al. 2021（arXiv:2107.02405v2）. Accessed: 2026-01-25.
- [Diosi1987] Diósi, L. “A universal master equation for the gravitational violation of quantum mechanics.” *Physics Letters A* **120**(8), 377–381 (1987). DOI: 10.1016/0375-9601(87)90681-5.
- [Penrose1996StateReduction] Penrose, R. “On Gravity's role in Quantum State Reduction.” *General Relativity and Gravitation* **28**, 581–600 (1996). DOI: 10.1007/BF02105068.
- [Donadi2020UndergroundCollapse] Donadi, S., Piscicchia, K., Curceanu, C., et al. “Underground test of gravity-related wave function collapse.” *Nature Physics* **17**, 74–78 (2021). DOI: 10.1038/s41567-020-1008-4. Accessed: 2026-01-26.
- [Kimura2004SinglePhotonInterference] Kimura et al. 2004（arXiv:quant-ph/0403104v2）. Accessed: 2026-01-25.
- [ArXiv2106_03871] arXiv:2106.03871v2 “Quantum interference of identical photons from remote GaAs quantum dots”. arXiv: `https://arxiv.org/abs/2106.03871`. Accessed: 2026-01-25.
- [Zenodo6371310] Zenodo（arXiv:2106.03871 source data）. DOI: 10.5281/zenodo.6371310. Accessed: 2026-01-25.
- [Vahlbruch2007Squeezed10dB] Vahlbruch et al. 2007（arXiv:0706.1431v1）. Accessed: 2026-01-25.
- [RoyLinMohideen2000Casimir] Roy, Lin, Mohideen 2000（arXiv:quant-ph/9906062v3）. Accessed: 2026-01-25.
- [IvanovKarshenboim2000LambShift] Ivanov & Karshenboim 2000（arXiv:physics/0009069v1）. Accessed: 2026-01-25.
- [Parthey2011Hydrogen1S2S] Parthey et al. 2011（arXiv:1107.3101v1）. Accessed: 2026-01-26.
- [Webb2011SpatialAlphaVariation] Webb, J. K., et al. “Indications of a Spatial Variation of the Fine Structure Constant.” *Phys. Rev. Lett.* **107**, 191101 (2011). DOI: 10.1103/PhysRevLett.107.191101. arXiv:1008.3907. Accessed: 2026-01-28.
