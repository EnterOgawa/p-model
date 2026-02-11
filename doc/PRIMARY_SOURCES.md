# 一次ソース参照先（データポータル一覧）

最終更新（UTC）：2026-01-31T19:56:26Z

目的：
- 本リポジトリで利用する「一次データ（観測そのもの）」や「一次論文ソース（TeX/PDF）」の入口（配布元URL）を、迷わないように一覧化する。
- 追加/変更があった場合は、`doc/WORK_HISTORY.md` に **必ず追記**し、可能なら `data/<topic>/sources/` に raw をキャッシュして offline 再現に備える。

原則：
- 一次データは「配布元のURL + 取得日（UTC） + ローカル保存先」をセットで記録する。
- `scripts/<topic>/fetch_*.py` が存在する場合は **そのスクリプトを正**とし、URLのハードコードは避ける。
- API 取得はレスポンス（raw）を `data/<topic>/sources/` か `output/<topic>/` に保存し、再実行で差分が追える形にする。

---

## 1) 太陽系・時計・レーザー測距（LLR/GPS/Cassini/Viking）

### JPL SSD Horizons（エフェメリス／幾何）
- Web / API：`https://ssd.jpl.nasa.gov/horizons/`（API: `https://ssd.jpl.nasa.gov/api/horizons.api`）
- 用途：LLR overlay / Cassini / Viking の幾何（light-time 等）
- 例：`scripts/llr/llr_pmodel_overlay_horizons_noargs.py`、`scripts/cassini/*horizons*`、`scripts/viking/*horizons*`
- 備考：SSL/Proxy 問題は環境変数（`HORIZONS_CA_BUNDLE` / `HORIZONS_INSECURE` / `HORIZONS_OFFLINE`）を優先。

### NAIF SPICE kernels（MOON PA 等）
- 公開ディレクトリ：`https://naif.jpl.nasa.gov/pub/naif/`
- 用途：月のPA座標系・時刻系など（LLR）
- 取得スクリプト：`scripts/llr/fetch_moon_kernels_naif.py`
- 保存：`data/llr/kernels/naif/`

### ILRS / EDC（CRD normal point 等）
- ILRS：`https://ilrs.gsfc.nasa.gov/`
- EDC（DGFI-TUM）：`https://edc.dgfi.tum.de/`（例：CRD normal point: `/pub/slr/data/npt_crd_v2/`）
- 用途：LLR（CRD normal point, station, pos+eop）
- 取得スクリプト例：`scripts/llr/fetch_llr_edc*.py`、`scripts/llr/fetch_pos_eop_edc.py`

### ILRS mission page（反射器の一次情報）
- NGLR-1：`https://ilrs.gsfc.nasa.gov/missions/satellite_missions/current_missions/ngl1_general.html`
- 用途：NGLR-1（nglr1）の反射器座標（PA）
- キャッシュ：`data/llr/sources/ilrs_ngl1_general.html`

### 海洋荷重（IMLS）
- 配布：`https://massloading.smce.nasa.gov/imls/`
- 用途：LLR の海洋荷重（harpos）
- 取得スクリプト：`scripts/llr/fetch_ocean_loading_imls.py`
- 保存：`data/llr/ocean_loading/`

### IGS（GPS products）
- BKG mirror：`https://igs.bkg.bund.de/root_ftp/IGS/`
- 用途：CLK/SP3/BRDC など（GPS時計比較）
- 取得スクリプト：`scripts/gps/fetch_igs_bkg.py`

### PDS（Cassini RSS）
- PDS archive（例）：`https://atmos.nmsu.edu/pdsd/archive/data/co-ss-rss-1-sce1-v10`
- 用途：Cassini SCE1（TDF/ODF 等）
- 取得スクリプト：`scripts/cassini/fetch_cassini_pds_sce1_*.py`

---

## 2) 重力波（GWOSC）

- Event API / strain：`https://gwosc.org/eventapi/`
- 用途：GW150914 等の公開 strain 取得（chirp/波形比較）
- 取得スクリプト例：`scripts/gw/gw150914_chirp_phase.py`
- GW250114 data release（Zenodo；GWOSC Event API の parameters が参照する公開プロダクト）：
  - DOI（GWOSC記載）：`https://doi.org/10.5281/zenodo.16877101`
  - API record（例）：`https://zenodo.org/api/records/16877102`
  - ローカル保存先：`data/gw/gw250114/GW250114_data_release.tar.gz`

---

## 3) 宇宙論（LSS/BAO：一次統計＝銀河+random）

### SDSS BOSS DR12（LSS）
- 推奨（高速ミラー）：`https://dr12.sdss3.org/sas/dr12/boss/lss/`
- 旧（遅い場合あり）：`https://data.sdss.org/sas/dr12/boss/lss/`
- 用途：BOSS DR12v5 LSS（銀河/ランダム）・関連zip（Ross/Satpathy/Beutler）
- 取得スクリプト：`scripts/cosmology/fetch_boss_dr12v5_lss.py` ほか

### SDSS eBOSS DR16（LSS）
- 例：`https://dr16.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/`
- 用途：eBOSS DR16（LRG/QSO/Lyα など）
- 取得スクリプト：`scripts/cosmology/fetch_eboss_dr16_lss.py`

### DESI DR1（LSS catalogs）
- public：`https://data.desi.lbl.gov/public/dr1/`
- LSS（iron; LSScats）：`https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/`
- mirror（WebDAV）：`https://webdav-hdfs.pic.es/data/public/DESI/DR1/`
- 用途：DESI DR1（銀河/ランダム）から ξ(s,μ)→ξℓ を再計算
- 取得スクリプト：`scripts/cosmology/fetch_desi_dr1_lss.py`、探索：`scripts/cosmology/discover_desi_public_2pt_products.py`

### DESI DR1 VAC（Lyα correlations; Y1 BAO）
- WebDAV（公開VAC）：`https://webdav-hdfs.pic.es/data/public/DESI/DR1/vac/dr1/lya-correlations/v1.0/`
- 用途：Lya QSO（Lyα forest auto + QSO cross）の公開 2pt（cf_*）と full covariance を取得し、peakfit 入口（ξ0/ξ2+cov）へ射影して ε fit を行う。
- 取得スクリプト：`scripts/cosmology/fetch_desi_dr1_vac_lya_correlations.py`
- 保存先（raw）：`data/cosmology/desi_dr1_vac_lya_correlations_v1p0/raw/`
- 備考：`full-covariance-smoothed.fits` は約 1.8GB（15000x15000）と大きい。

### Pantheon（SNe Ia; 参考）
- raw（GitHub）：`https://raw.githubusercontent.com/dscolnic/Pantheon/master/`
- 用途：距離指標の誤差予算（代表データ）
- 取得実装：`scripts/cosmology/cosmology_distance_indicator_error_budget.py`

### bao_data（BAO圧縮データ; 参考）
- raw：`https://raw.githubusercontent.com/CobayaSampler/bao_data/master/`
- repo：`https://github.com/CobayaSampler/bao_data`
- 用途：BAO「圧縮出力」側の比較（ただし本プロジェクトは一次統計（銀河+random）優先）
- 取得スクリプト：`scripts/cosmology/fetch_desi_dr1_bao_bao_data.py`

---

## 4) EHT（一次論文ソース）

- 方針：PDF/TeX を `data/eht/sources/` に保存して sha256 を固定する（詳細は `doc/eht/README.md`）。
- 固定キャッシュ（sha256）：
  - EHT M87* Paper I（arXiv:1906.11238）：`data/eht/sources/arxiv_1906.11238.pdf`（sha256=`BE6604770BB0D0A6C2A81071518E89D5BC2F9B460F91390518804E0A1F454472`）
  - EHT M87* Paper VI（arXiv:1906.11243）：`data/eht/sources/arxiv_1906.11243.pdf`（sha256=`51BC4254165360C1F22BF1E739BF88930AE0FAFD8FE2E960BB1FACF70F765E04`）
  - EHT M87* 2018 Paper I（A&A 681, A79, 2024; DOI: 10.1051/0004-6361/202347932）：`data/eht/sources/aanda_aa47932-23.pdf`（sha256=`8A972A997314088214F8CEBA4D52D2B1159ACC2BC7BE2CD2CC28E30A1BDF7B15`）
  - EHT M87* 2018 Paper II（A&A 693, A265, 2025; DOI: 10.1051/0004-6361/202451296）：`data/eht/sources/aanda_aa51296-24.pdf`（sha256=`52A2806AFB4D6DA458B83BD8867229741962004F32385BCE5D533ECA2415B4A3`）
  - Vincent et al. 2022（A&A 667, A170; arXiv:2206.12066；ring≒shadow の系統）：`data/eht/sources/arxiv_2206.12066.pdf`（sha256=`5DF77CDE0801423D062657E09075569340AD0100D9F5EABB05B7609E9A47FC1B`）
    - arXiv source：`data/eht/sources/arxiv_2206.12066_src.tar.gz`（sha256=`6613AAF540609BAD1668C20EAAC5FBAC086956A2DAFEA1B89B8FF7C550B389AF`）
      - 展開先：`data/eht/sources/arxiv_2206.12066/`
  - Johnson et al. 2020（Science Advances 6(12), eaaz1310; arXiv:1907.04329；photon ring干渉計シグナチャー）：`data/eht/sources/arxiv_1907.04329.pdf`（sha256=`721438EEE1C6F3D176AC5FDC3B196A20A9547CB778FAA071A9BB95128479B8C9`）
    - arXiv source：`data/eht/sources/arxiv_1907.04329_src.tar.gz`（sha256=`0A2731898760D1E166F3F1F180191E48112215F8B798961542D229D467637BC1`）
      - 展開先：`data/eht/sources/arxiv_1907.04329/`
  - Gralla et al. 2020（PRD 102, 124004; arXiv:2008.03879；photon ring形状テスト）：`data/eht/sources/arxiv_2008.03879.pdf`（sha256=`042B5F3DA9EB3C639383593F40EFEB7A44C69100BE0EB028B24D7EDC4A73F157`）
    - arXiv source：`data/eht/sources/arxiv_2008.03879_src.tar.gz`（sha256=`E0D1CDB11D0A89A03D077D925A0CB7303FDFAFBC64A1996C2284D4AA86007E89`）
      - 展開先：`data/eht/sources/arxiv_2008.03879/`
  - EHT Sgr A* Paper II（arXiv:2311.08679, source）：`data/eht/sources/arxiv_2311.08679_src.tar.gz`（sha256=`7DF0836BF79E6DA0EF9BFA2E490C3B9DAAAD9290C01CE4A9E608DA236C6B20CE`）
    - 展開先：`data/eht/sources/arxiv_2311.08679/`
  - EHT Sgr A* Paper II（arXiv:2311.08679, PDF）：`data/eht/sources/arxiv_2311.08679.pdf`（sha256=`514C0467C41FC6CFA3D88E1185C10D8F147C09CCB12020B1F325FE7FEDDC4E75`）
  - EHT Sgr A* Paper I（arXiv:2311.08680, source）：`data/eht/sources/arxiv_2311.08680_src.tar.gz`（sha256=`8A03702D3C2037D21EDD5B00C0FF97DA97EA126960DD3ED244BE4A4B63BA6C33`）
    - 展開先：`data/eht/sources/arxiv_2311.08680/`
  - EHT Sgr A* Paper I（arXiv:2311.08680, PDF）：`data/eht/sources/arxiv_2311.08680.pdf`（sha256=`DF498BCAA7F11A5CF6AD15CEC6F8D37CD0FAD05F6A251505637206CDC01E3C47`）
  - EHT Sgr A* Paper III（arXiv:2311.09479, source）：`data/eht/sources/arxiv_2311.09479_src.tar.gz`（sha256=`86DD920BE3724CB253261E13A51C1D4CF81F69BE5711129C4D62C36C71597E84`）
    - 展開先：`data/eht/sources/arxiv_2311.09479/`
  - EHT Sgr A* Paper III（arXiv:2311.09479, PDF）：`data/eht/sources/arxiv_2311.09479.pdf`（sha256=`DD40488CD9CC0F4BE94C60435ABC579C6D706FE4F15C06C45692C738668610C1`）
  - EHT Sgr A* Paper IV（arXiv:2311.08697, source）：`data/eht/sources/arxiv_2311.08697_src.tar.gz`（sha256=`F9768B53DC3D793E29CDB62A124B59E4C20C3C455720F65AB4E3F6C04A779177`）
    - 展開先：`data/eht/sources/arxiv_2311.08697/`
  - EHT Sgr A* Paper IV（arXiv:2311.08697, PDF）：`data/eht/sources/arxiv_2311.08697.pdf`（sha256=`42F25D9F11D26A073EDDD57BFF42FB8F21761EB4698512E9FE59EC2038BDFD1D`）
  - EHT Sgr A* Paper V（arXiv:2311.09478, source）：`data/eht/sources/arxiv_2311.09478_src.tar.gz`（sha256=`5143C78D40D7CB316D279087506334B536CDC654D9FEE66F1EF45403EC6BD554`）
    - 展開先：`data/eht/sources/arxiv_2311.09478/`
  - EHT Sgr A* Paper V（arXiv:2311.09478, PDF）：`data/eht/sources/arxiv_2311.09478.pdf`（sha256=`CA438CE4EAB1FD1337D96E308EF63FC28E512667803F4890D83BBA0AE7548864`）
  - EHT Sgr A* Paper VI（arXiv:2311.09484, source）：`data/eht/sources/arxiv_2311.09484_src.tar.gz`（sha256=`99EC01FFB9D50B8DA1EF5A6FE465802D16D502D6EC472E35A86CC0A0FFDB9EFE`）
    - 展開先：`data/eht/sources/arxiv_2311.09484/`
  - EHT Sgr A* Paper VI（arXiv:2311.09484, PDF）：`data/eht/sources/arxiv_2311.09484.pdf`（sha256=`12A674DC2CDB233021CEBBBD91D5E4EAEB865C433EEE871581FA0DBAA5E7B83D`）
  - GRAVITY Collaboration（2018; arXiv:1807.09409）：`data/eht/sources/arxiv_1807.09409.pdf`（sha256=`BD620D1136EB396BEEE040A6A4CFC39AC980B8B3A1369A867839B0283577D185`）
    - arXiv source：`data/eht/sources/arxiv_1807.09409_src.tar.gz`（sha256=`FCEDE8B2AF8EFA458AA2A3747975F7B09051C55FEB5096F1C52CA6B76A15DF42`）
      - 展開先：`data/eht/sources/arxiv_1807.09409/`
  - GRAVITY Collaboration（2020; arXiv:2004.07187）：`data/eht/sources/arxiv_2004.07187.pdf`（sha256=`FD662D1B585106220DD19922963038ADF2A0BFDDDD2B10F03EA9191ADDC19398`）
    - arXiv source：`data/eht/sources/arxiv_2004.07187_src.tar.gz`（sha256=`BDA7CC9454C57C8D2BFDD93E67D1C6735A516A401A97052BAF34C253A765E23E`）
      - 展開先：`data/eht/sources/arxiv_2004.07187/`
  - GRAVITY Collaboration（2020; arXiv:2004.07185; The flux distribution of Sgr A*）：`data/eht/sources/arxiv_2004.07185.pdf`（sha256=`4269A14DF8A327E27A6579221A8BF7C1853D7B05184BDCE19958F17DD5D87EF4`）
    - arXiv source：`data/eht/sources/arxiv_2004.07185_src.tar.gz`（sha256=`78681682A9BBFC73C8E4943D4756E5A83A9A3976D6BF9777FA8F0755A7EA5685`）
      - 展開先：`data/eht/sources/arxiv_2004.07185/`
  - Wielgus et al.（ApJL 930, L19, 2022; arXiv:2207.06829）：`data/eht/sources/arxiv_2207.06829.pdf`（sha256=`A0BC35AD1FD7B6CA5C5D93F2563C725A4B131B18CBD436F88BAE3DBE5F687057`）
    - arXiv source：`data/eht/sources/arxiv_2207.06829_src.tar.gz`（sha256=`43DCD6B1969B44EA945B8C771F9FDAC4D22EFF4B50DB09661195791C201AA9C7`）
      - 展開先：`data/eht/sources/arxiv_2207.06829/`
  - GRAVITY Collaboration（A&A 625, L10, 2019; arXiv:1904.05721）：`data/eht/sources/arxiv_1904.05721.pdf`（sha256=`C9A033D3753754740FE994E1A9D20B9B5C9B5D55C34B401802BD9139653062FF`）
  - Mertens et al.（A&A 595, A54, 2016; arXiv:1608.05063）：`data/eht/sources/arxiv_1608.05063.pdf`（sha256=`85FFF49C3C2D801B5B9A67FB14B6F6EBA64ED488EDA30CCFC9A887588A69CDE0`）
  - Psaltis et al. 2018（arXiv:1805.01242）：`data/eht/sources/arxiv_1805.01242.pdf`（sha256=`F7297397F150B6FCE4249BC8DFB185D2DCA4A7DC46E48F13B8C49F1082ED5B33`）
  - Johnson et al. 2018（arXiv:1808.08966）：`data/eht/sources/arxiv_1808.08966.pdf`（sha256=`9A041884770A9ED2224ECE118AECDE47D758824AC6C5185B4D7BC048A0C3F349`）
    - arXiv source：`data/eht/sources/arxiv_1808.08966_src.tar.gz`（sha256=`2D4DA3172C6B3730678DE7146DF21A0288F2B7814C7B553F917BD06A0681015F`）
      - 展開先：`data/eht/sources/arxiv_1808.08966/`
  - Zhu et al. 2018（arXiv:1811.02079）：`data/eht/sources/arxiv_1811.02079.pdf`（sha256=`2070909C73598E66141012B1C385EB59635C114D566DAA32DC92DB86DC2FFB24`）
    - arXiv source：`data/eht/sources/arxiv_1811.02079_src.tar.gz`（sha256=`E98DE933C2DBB2966C379AD987A0BF18F23B58A1762BA0120A12AFB138D3EE0E`）
      - 展開先：`data/eht/sources/arxiv_1811.02079/`
  - 追加（Paper V “historical distribution” の一次データ基盤）：
    - Marrone（2006; thesis, SMA 2005）：Daniel P. Marrone thesis（CfA）をローカル固定（light curve digitize に利用）。
      - URL（PDF）：`https://lweb.cfa.harvard.edu/~dmarrone/dpm_thesis.pdf`
      - ローカル（PDF）：`data/eht/sources/marrone2006_thesis_dpm_thesis.pdf`（sha256=`7B4096A59487B920CD245FB92DE9634D63A927119013785E53BF8FF5ECE8FFEA`）
      - URL（PS.gz）：`https://lweb.cfa.harvard.edu/~dmarrone/dpm_thesis.ps.gz`
      - ローカル（PS.gz）：`data/eht/sources/marrone2006_thesis_dpm_thesis.ps.gz`（sha256=`78F6E6E41FD0A8858BFD4EBF8C7D90D7EDB3707EA49B926C49D419F7D0CD6FB4`）
    - Marrone et al.（2006; arXiv:astro-ph/0607432）：`data/eht/sources/arxiv_astroph_0607432.pdf`（sha256=`AE3A305277BB559198C0AF0F02D8C5965AAF51F4D8674DA22875A5554B9BA2FC`）
      - arXiv source：`data/eht/sources/arxiv_astroph_0607432_src.tar.gz`（sha256=`8AF2830DE3ADBFBD1A57586184A42A31F1C4E9C95E992A09B5FD8FC6C7120535`）
        - 展開先：`data/eht/sources/arxiv_astroph_0607432/`
    - Marrone et al. 2008（arXiv:0712.2877; SMA 2006-07-17）：`data/eht/sources/arxiv_0712.2877.pdf`（sha256=`EC5CCFE382E46532D5AB239164F075B5015B543D0C7A50A5B6D2825332D08605`）
      - arXiv source：`data/eht/sources/arxiv_0712.2877_src.tar.gz`（sha256=`3F045C97736131A6E45F8167A2F686B4EBB5A89B20EEED58631F52F3DFD49698`）
        - 展開先：`data/eht/sources/arxiv_0712.2877/`
    - Yusef-Zadeh et al. 2009（arXiv:0907.3786; SMA 2007-04-01/03/04/05）：`data/eht/sources/arxiv_0907.3786.pdf`（sha256=`15002DD390288A3652B1F4F31AF0C84854746A25C6DB89826BBCFCAAF4D6A6E8`）
      - arXiv source：`data/eht/sources/arxiv_0907.3786_src.tar.gz`（sha256=`E243C9A1B295E567D0F950FC2ABED71192DA23371B57631CAEB0136578FBB8CF`）
        - 展開先：`data/eht/sources/arxiv_0907.3786/`
    - Dexter et al. 2014（arXiv:1308.5968; CARMA 2009/2011, SMA 2012）：`data/eht/sources/arxiv_1308.5968.pdf`（sha256=`F02C49CFD074081B5129961D80E38FA16F03357D174CF067090A8E19A94E6A26`）
      - arXiv source：`data/eht/sources/arxiv_1308.5968_src.tar.gz`（sha256=`2640F1DDED60C468757138B9DFE118E402FDAC488E181A2BBFE95E31E7CFFD1E`）
        - 展開先：`data/eht/sources/arxiv_1308.5968/`
    - Fazio et al. 2018（arXiv:1807.07599; SMA 2015-05-14）：`data/eht/sources/arxiv_1807.07599.pdf`（sha256=`080240EF36DE1BDB8D582D8C82A1B7BD10100D10D3BEB78E9C12908C0B8B4C03`）
      - arXiv source：`data/eht/sources/arxiv_1807.07599_src.tar.gz`（sha256=`F0DDDB999CD5664BB623AF94008D06B52B325656648B36091EB71F986CD9597A`）
        - 展開先：`data/eht/sources/arxiv_1807.07599/`
    - Bower et al. 2018（arXiv:1810.07317; ALMA 2016-03-03/08-13）：`data/eht/sources/arxiv_1810.07317.pdf`（sha256=`703742EE1D17B5CF1BCB6CCC20480B8718349B150B623334000543AC803214B4`）
      - arXiv source：`data/eht/sources/arxiv_1810.07317_src.tar.gz`（sha256=`01090FAF6CE4C7085A16460BECBA0F7668BAE8F795FBDFDCD3DFC56F282F5F44`）
        - 展開先：`data/eht/sources/arxiv_1810.07317/`
    - Witzel et al. 2021（arXiv:2011.09582; ALMA 2016-07-12, SMA 2016-07-13）：`data/eht/sources/arxiv_2011.09582.pdf`（sha256=`000E32F0F81828DD17CC8607538DD2E846BB3859987E2468C319A0954BBA3B20`）
      - arXiv source：`data/eht/sources/arxiv_2011.09582_src.tar.gz`（sha256=`9234EC4554347E5BFF0FB9122E8E6535819F6A805EDAF4A19117FBD848896E0A`）
        - 展開先：`data/eht/sources/arxiv_2011.09582/`
    - IOP “data behind figure”（機械可読の light curve）：
      - Witzel+2021 dbf1：`data/eht/lightcurves/witzel2021/apjac0891f1.tar.gz`（sha256=`F8792637E832397129D1EAC5660C99192AC372C0B85BD9D7D55D6B81BAC344D4`）→ `data/eht/lightcurves/witzel2021/dbf1.txt`（sha256=`CA5EA706A6D5481ECF383707EA8EE078668BEDD8AEB8AF6E631CE48AFE255CC7`）
      - Fazio+2018 dbf3：`data/eht/lightcurves/fazio2018/apjaad4a2f3.tar.gz`（sha256=`8B2C01C84E9E567B8ABCA979B7420851F129A55AB28A2765A7EDD05AE84BC7E8`）→ `data/eht/lightcurves/fazio2018/dbf3.txt`（sha256=`9CB0CA922440ABC5D185B3C51BADF02F4F2903BE92AE6145334F8A7E5EFFDCB8`）
    - 追加（補強論文の取得・キャッシュ）：`python -B scripts/eht/fetch_eht_supporting_papers.py`

---

## 5) 量子（ベルテスト等：将来の一次データ）

- 参照メモ：`doc/quantum/P-model_BellTest_AI_Brief.md`

### 量子測定（Born則・状態更新）：位置づけ（Step 7.10）
- 基準書（資料）：`doc/quantum/15_quantum_measurement_born_rule.md`
- 固定出力（位置づけ図＋要点）：`python -B scripts/quantum/quantum_measurement_born_rule.py`
  - 出力：`output/quantum/quantum_measurement_born_rule_flow.png`（sha256=`A58E5189DB42087F9CF6C478B42853400EC2EF79CE3376CBD4ACE55C788E5808`）
  - 出力：`output/quantum/quantum_measurement_born_rule_metrics.json`（sha256=`439199AE84F37574B533730F4C643350F237B5ADE2F918927BCDFBB72F87276F`）

### 電磁気（電荷・Maxwell/光子）：最小導入（Step 7.11）
- 基準書（資料）：`doc/quantum/16_electromagnetism_charge_maxwell_photon.md`
- 固定出力（スケール確認＋位置づけ図＋要点）：`python -B scripts/quantum/electromagnetism_minimal.py`
  - 出力：`output/quantum/electromagnetism_minimal.png`（sha256=`E519B14365267A734F93E8E22195FA4A72B2CA39F0C381EBDE53AC980D3FBEDB`）
  - 出力：`output/quantum/electromagnetism_minimal_metrics.json`（sha256=`DBBB8801687BAA99A8A6E0C72CB202187012D526C6D3F27FAF5761ACB58C9268`）
- 固定出力（αのP依存性チェック；κ制約）：`python -B scripts/quantum/alpha_p_dependence_constraint.py`
  - 出力：`output/quantum/alpha_p_dependence_constraint.png`（sha256=`317021287BCE2991A7E1B8EAC31CF7552866DA482519B923FA6E9519163A1245`）
  - 出力：`output/quantum/alpha_p_dependence_constraint.json`（sha256=`C21DA156FB30346EDCACE3C239F0E55AECCA2D2BB6395EA48FC9D60CA9AD05AA`）

### 原子・分子（原子スペクトル基準値；Step 7.12）
- NIST Atomic Spectra Database（ASD；Lines Form）：`https://physics.nist.gov/PhysRefData/ASD/lines_form.html`
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nist_asd_lines.py --spectra "H I"`（保存先：`data/quantum/sources/nist_asd_h_i_lines/`）
  - 取得結果マニフェスト：`data/quantum/sources/nist_asd_h_i_lines/manifest.json`（sha256=`E2184894A0A245545151B92C5C9B2D621ABF9F9E3A9D318EEF8D652BBB714D05`）
  - 抽出値：`data/quantum/sources/nist_asd_h_i_lines/extracted_values.json`（sha256=`5A506E9BCCD07F85451EB09362C73B8276F67EED0A9F027CB29237F4895FCA0F`）
  - 取得TSV：`data/quantum/sources/nist_asd_h_i_lines/nist_asd_lines__h_i__format3__obs_ritz_unc.tsv`（sha256=`B3415AC650B641C2125A836AA494C8CA6927B6AA29B81D93E246145EE3FCC6D0`）
- 固定出力（基準値）：`python -B scripts/quantum/atomic_hydrogen_baseline.py`
  - 出力：`output/quantum/atomic_hydrogen_baseline.png`（sha256=`72210FF7DC74F86AA3042623F01059B69E6ACE9EAF93D81D4C592DEE6304E0BC`）
  - 出力：`output/quantum/atomic_hydrogen_baseline_metrics.json`（sha256=`38FAF84B25C0FA213C469DE66B7331F725D6FD4D16D443061A993A45D3291F62`）
- NIST AtSpec handbook（H I hyperfine; 21 cm）：`https://www.physics.nist.gov/Pubs/AtSpec/AtSpec.PDF`
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nist_atspec_handbook.py`（保存先：`data/quantum/sources/nist_atspec_handbook/`）
  - 取得結果マニフェスト：`data/quantum/sources/nist_atspec_handbook/manifest.json`（sha256=`ACE3EA49DDD99D64D29FB284011CE7C078D9C153C1477528D03EFCAAB81CFC40`）
  - 抽出値：`data/quantum/sources/nist_atspec_handbook/extracted_values.json`（sha256=`4C8EEC89415FA3570483DAB837FB24246FDA28E1CE92AFE7E3DBA169BBFECBF9`）
  - 取得PDF：`data/quantum/sources/nist_atspec_handbook/AtSpec.PDF`（sha256=`4920AE3A9DD60351D9918BA53A0586CD11D50CEBE95336E0282E019448362C4D`）
- 固定出力（基準値）：`python -B scripts/quantum/atomic_hydrogen_hyperfine_baseline.py`
  - 出力：`output/quantum/atomic_hydrogen_hyperfine_baseline.png`（sha256=`9A73BCEE8920551AADA1A088EE161FE72949B3D6D479308664DC962B3C8E1A88`）
  - 出力：`output/quantum/atomic_hydrogen_hyperfine_baseline_metrics.json`（sha256=`4AB071F9B6A38E6F5F8515C750769A25146A56FA0361BE1BCDCF4F2AF4D8F883`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nist_asd_lines.py --spectra "He I"`（保存先：`data/quantum/sources/nist_asd_he_i_lines/`）
  - 取得結果マニフェスト：`data/quantum/sources/nist_asd_he_i_lines/manifest.json`（sha256=`7BC2B52240E841E039EB529969E62CDCFD26AEA0D54E9AF640486596863E5EB0`）
  - 抽出値：`data/quantum/sources/nist_asd_he_i_lines/extracted_values.json`（sha256=`2830AA4E9E4CC3923E28B49B18AC21E70B0D091E91947CE7D675B2064DA4ADDC`）
  - 取得TSV：`data/quantum/sources/nist_asd_he_i_lines/nist_asd_lines__he_i__format3__obs_ritz_unc.tsv`（sha256=`2D1F55956A907F82BF8DF8EE9C54513D20A6573B7AECBCE2A7E4DFBFAA86F61A`）
- 固定出力（基準値）：`python -B scripts/quantum/atomic_helium_baseline.py`
  - 出力：`output/quantum/atomic_helium_baseline.png`（sha256=`CC0BECC7629ED07561FB4EBFE4751723F39EEC2163D2AAF92CDFEF6295358D3A`）
  - 出力：`output/quantum/atomic_helium_baseline_metrics.json`（sha256=`5A23FEF694A3038E820076EEF35FED0100EF5C62C28B977E77817EB8DF8DF746`）
- NIST Chemistry WebBook（Constants of diatomic molecules）：`https://webbook.nist.gov/cgi/cbook.cgi?Mask=1000`
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nist_webbook_diatomic_constants.py --id C1333740 --slug h2`（保存先：`data/quantum/sources/nist_webbook_diatomic_h2/`）
  - 取得結果マニフェスト：`data/quantum/sources/nist_webbook_diatomic_h2/manifest.json`（sha256=`7ED3AEF7598AE57185323D9FEFF12402C7C8CDE12EC0D5F527483CD5BA742074`）
  - 抽出値：`data/quantum/sources/nist_webbook_diatomic_h2/extracted_values.json`（sha256=`A514FD722904116780046EA26289AEC0F5ABA0E802BA4F0178BD9061BF5D61A9`）
  - 取得HTML：`data/quantum/sources/nist_webbook_diatomic_h2/nist_webbook_diatomic_constants__h2__c1333740.html`（sha256=`016071DDFEC6443A1D76C97ED33BE8811E6B49A550D1AAB79EE5B52AC90DB486`）
- 固定出力（基準値）：`python -B scripts/quantum/molecular_h2_baseline.py --slug h2`
  - 出力：`output/quantum/molecular_h2_baseline.png`（sha256=`06F077CF3E121BD5A389F8C02FC3ED601C1B25A8209C9536F64AF3B425769902`）
  - 出力：`output/quantum/molecular_h2_baseline_metrics.json`（sha256=`77E264C3A57548710BA5DC89EDAB74AFDEE150F18A1D0AC3276F1D437CBA98C4`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nist_webbook_diatomic_constants.py --id C13983205 --slug hd`（保存先：`data/quantum/sources/nist_webbook_diatomic_hd/`）
  - 取得結果マニフェスト：`data/quantum/sources/nist_webbook_diatomic_hd/manifest.json`（sha256=`7BC6D9C77883B5364A1CF6EDFF2BF7B24275E48C2C4F4AD08B807C1B2F7BD099`）
  - 抽出値：`data/quantum/sources/nist_webbook_diatomic_hd/extracted_values.json`（sha256=`CF369493191C06E668DBBCADAE6600892204856D10C32A7E1231E12FCEEA6386`）
  - 取得HTML：`data/quantum/sources/nist_webbook_diatomic_hd/nist_webbook_diatomic_constants__hd__c13983205.html`（sha256=`9DD2A7FF8F19343177935A2753FB83D2264E2DA5432FB7804236DEE636C62894`）
- 固定出力（基準値）：`python -B scripts/quantum/molecular_h2_baseline.py --slug hd`
  - 出力：`output/quantum/molecular_hd_baseline.png`（sha256=`AF507BEE6172D02A8BE9B1F352D181A51C54B20F9F090D3698A752B98FA086E0`）
  - 出力：`output/quantum/molecular_hd_baseline_metrics.json`（sha256=`27AD7929383208ED3F47F71FF6511C7F24BBDC5980CA602BC60E9A3FA177D6C3`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nist_webbook_diatomic_constants.py --id C7782390 --slug d2`（保存先：`data/quantum/sources/nist_webbook_diatomic_d2/`）
  - 取得結果マニフェスト：`data/quantum/sources/nist_webbook_diatomic_d2/manifest.json`（sha256=`0EB8354936810E0947C850C781806A33BDEE538B185CF966DF4AC2C0C84FB6F4`）
  - 抽出値：`data/quantum/sources/nist_webbook_diatomic_d2/extracted_values.json`（sha256=`450132B31351DD09D2CE8F49D9A6E8BEAF8DA4479AE141CB20C34E0D0DE29CF9`）
  - 取得HTML：`data/quantum/sources/nist_webbook_diatomic_d2/nist_webbook_diatomic_constants__d2__c7782390.html`（sha256=`AC4C69921562B5B7C9B68D14631F4EE119EC82B414426626398D19AC62C4B725`）
- 固定出力（基準値）：`python -B scripts/quantum/molecular_h2_baseline.py --slug d2`
  - 出力：`output/quantum/molecular_d2_baseline.png`（sha256=`A23DC2599163D6563E7F4946E016C335A24F967002B228B7C4B7EA8B596E36B3`）
  - 出力：`output/quantum/molecular_d2_baseline_metrics.json`（sha256=`1DA9969C47EFBAB8054AF6F536FFAB2F5A9CAEB0750421C1337F6021EDFED9E5`）
- 固定出力（同位体依存チェック）：`python -B scripts/quantum/molecular_isotopic_scaling.py`
  - 出力：`output/quantum/molecular_isotopic_scaling.png`（sha256=`E6734BD7C2E3D4F6F0361E358D352A4EA022B0A3A6D0AC213B50646AB85ECA0A`）
  - 出力：`output/quantum/molecular_isotopic_scaling_metrics.json`（sha256=`DB96336204DF0C6DFE07408F8A6B5AE2AC568986C59186CE66A286720F84AD56`）
- NIST Atomic Weights and Isotopic Compositions（Hydrogen; relative atomic mass）：`https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=H`
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nist_isotopic_compositions.py --element H`（保存先：`data/quantum/sources/nist_isotopic_compositions_h/`）
  - 取得結果マニフェスト：`data/quantum/sources/nist_isotopic_compositions_h/manifest.json`（sha256=`1FC0A0F3151B57EC9795F6565953C72551CAA5439BBFD59A968A3F0E4DE24505`）
  - 抽出値：`data/quantum/sources/nist_isotopic_compositions_h/extracted_values.json`（sha256=`C847A9618FA3E5C5A85014CCBF07B0D38858FB242FE7BC54E3DCA613E7E16979`）
  - 取得HTML：`data/quantum/sources/nist_isotopic_compositions_h/nist_isotopic_compositions__h.html`（sha256=`90FC727E64AC51FCFA590B5515728A47B79737F308232540F37C92BECCC86615`）
- NIST Chemistry WebBook（Gas phase thermochemistry）：`https://webbook.nist.gov/cgi/cbook.cgi?Mask=1`
- 取得・キャッシュ（スクリプト）：
  - `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C12385136 --slug h_atom`（保存先：`data/quantum/sources/nist_webbook_thermo_h_atom/`）
    - 取得結果マニフェスト：`data/quantum/sources/nist_webbook_thermo_h_atom/manifest.json`（sha256=`355BA6E53395A89AD786D404EBE10B3000BE5839CDBDFF1444268D2CCE210A3E`）
    - 抽出値：`data/quantum/sources/nist_webbook_thermo_h_atom/extracted_values.json`（sha256=`E31A13673BC59CA3F13D10A6A988798A846218D7437482759FAA865AAAF0B6B6`）
    - 取得HTML：`data/quantum/sources/nist_webbook_thermo_h_atom/nist_webbook_thermo__h_atom__c12385136.html`（sha256=`C1E28E868F08B945B8A66EB964277FB6C22F02779FF0673D34A56399206E83E1`）
  - `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C16873179 --slug d_atom`（保存先：`data/quantum/sources/nist_webbook_thermo_d_atom/`）
    - 取得結果マニフェスト：`data/quantum/sources/nist_webbook_thermo_d_atom/manifest.json`（sha256=`B4A266FDEFFAF5337FE6E770E1DB955079F9E38C77E9DF2F93B5E286E6D77FE4`）
    - 抽出値：`data/quantum/sources/nist_webbook_thermo_d_atom/extracted_values.json`（sha256=`B50837E3130F046800E686D67246A0A25CF6464C5EFAED7D8353B36D96EC1452`）
    - 取得HTML：`data/quantum/sources/nist_webbook_thermo_d_atom/nist_webbook_thermo__d_atom__c16873179.html`（sha256=`17C7AD6700BFBB13BED8D8F46DF146302608E7A22C358919A6DF0F41EF58EF7D`）
  - `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C1333740 --slug h2`（保存先：`data/quantum/sources/nist_webbook_thermo_h2/`）
    - 取得結果マニフェスト：`data/quantum/sources/nist_webbook_thermo_h2/manifest.json`（sha256=`E72283CCAAB74206624A90A6BEB1A705A3EC9847AB2E5B5C814B0979F38A7362`）
    - 抽出値：`data/quantum/sources/nist_webbook_thermo_h2/extracted_values.json`（sha256=`23782B156A1BF7020F3AABB2B2EE32A7BA744966EFDBD2FD317B00168303B80F`）
    - 取得HTML：`data/quantum/sources/nist_webbook_thermo_h2/nist_webbook_thermo__h2__c1333740.html`（sha256=`E184E0EF474B3F898AFCAD2C2491FD32D5603DF1B76B0E1894A4B2D3E0AE8DF5`）
  - `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C13983205 --slug hd`（保存先：`data/quantum/sources/nist_webbook_thermo_hd/`）
    - 取得結果マニフェスト：`data/quantum/sources/nist_webbook_thermo_hd/manifest.json`（sha256=`375AAD8A69CAE45FC4847EE129B6CE0CF282ED24AF9C4468BECBAA5743C5526A`）
    - 抽出値：`data/quantum/sources/nist_webbook_thermo_hd/extracted_values.json`（sha256=`3E1DF931FAF734152F9E5D1F13657C0DE44EEE5C485DE32F679E8C0FF1B953B4`）
    - 取得HTML：`data/quantum/sources/nist_webbook_thermo_hd/nist_webbook_thermo__hd__c13983205.html`（sha256=`18EAD402E08A1483A17450F91CDE109FEB82CEB7AE9C31899280410D49B4B34C`）
  - `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C7782390 --slug d2`（保存先：`data/quantum/sources/nist_webbook_thermo_d2/`）
    - 取得結果マニフェスト：`data/quantum/sources/nist_webbook_thermo_d2/manifest.json`（sha256=`FF66CF7E82538CC3259FA3B45381204934DF7CACC2DFC673ACF06BC989E9B082`）
    - 抽出値：`data/quantum/sources/nist_webbook_thermo_d2/extracted_values.json`（sha256=`2E4EFE2642A7365800F4886F030D022279AD71CC22B1C94968B016B836C71CE7`）
    - 取得HTML：`data/quantum/sources/nist_webbook_thermo_d2/nist_webbook_thermo__d2__c7782390.html`（sha256=`4FF0B1BC06C2C9618C5A987BF0AD34A85AD8B8571A9CFB0B0AEE97631D44A5FE`）
- 固定出力（解離エンタルピー；298 K）：`python -B scripts/quantum/molecular_dissociation_thermochemistry.py`
  - 出力：`output/quantum/molecular_dissociation_thermochemistry.png`（sha256=`1871DD4047C288EA977D4BB1850A143EFF9B42A82512C2C1EE5BB1125B4BC16D`）
  - 出力：`output/quantum/molecular_dissociation_thermochemistry_metrics.json`（sha256=`12CE89D592E8A451CB499438BF8AF95F47CDC098ECB11BEE7AF608BC9088DE4E`）

- 高精度分光（ionization→D0；分光学的解離エネルギー D0（0 K））：
  - 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_molecular_dissociation_d0_spectroscopic.py`（保存先：`data/quantum/sources/molecular_dissociation_d0_spectroscopic/`）
    - 取得結果マニフェスト：`data/quantum/sources/molecular_dissociation_d0_spectroscopic/manifest.json`（sha256=`523F97320EE8BAFE352D19CFCAFACF5EE0E2B4B76E51E8124E5C6FFC2FBF67BE`）
    - 抽出値：`data/quantum/sources/molecular_dissociation_d0_spectroscopic/extracted_values.json`（sha256=`7C030F21EA2382CD3BB048C1CE1BA3EE08C82618C6279CE7458D30F087E0CA03`）
    - arXiv（H2；abs）：`data/quantum/sources/molecular_dissociation_d0_spectroscopic/arxiv_abs_1902.09471.html`（sha256=`A6659EA5D11CA3E4F1971DCF79172AD3AD7936C906C96AFFD68F61FE34265DFC`）
    - arXiv（H2；pdf）：`data/quantum/sources/molecular_dissociation_d0_spectroscopic/arxiv_pdf_1902.09471.pdf`（sha256=`F4B0F3903ED218FA084958ACAF036EF4C2939C863ED1945A3BC157FB65A3FED6`）
    - arXiv（D2；abs）：`data/quantum/sources/molecular_dissociation_d0_spectroscopic/arxiv_abs_2202.00532.html`（sha256=`443014B705A0D89A682067D6A9C223013EB3492FD33A0A996E9885291D502E59`）
    - arXiv（D2；pdf）：`data/quantum/sources/molecular_dissociation_d0_spectroscopic/arxiv_pdf_2202.00532.pdf`（sha256=`ADE6F9FA3BCCBAFCBD20DF4FFD52821E629DCC9B904640326BB1C62EDC9EC724`）
    - VU Amsterdam（HD；doi=10.1103/PhysRevA.108.022811）：`data/quantum/sources/molecular_dissociation_d0_spectroscopic/vu_publication_hd_d0.html`（sha256=`17745ED178E49FC0DCA9AAF89367EDB42E293D3573006809BC6F3E5A5C1A8ED8`）
  - 固定出力（D0；0 K）：`python -B scripts/quantum/molecular_dissociation_d0_spectroscopic.py`
    - 出力：`output/quantum/molecular_dissociation_d0_spectroscopic.png`（sha256=`E0A147F0E5E631C03D6475F37C89257028AFAF45CBA0968B30AF6CD6D39F3BBD`）
    - 出力：`output/quantum/molecular_dissociation_d0_spectroscopic_metrics.json`（sha256=`E4078BA445852FF2D84494AB87E1221F242B65C2C85168BE914AD151BF0A85BF`）

- ExoMol database（分子遷移 line list；states/trans/pf）：
  - H2（1H2; RACPPK）：`https://exomol.com/data/molecules/H2/1H2/RACPPK`
  - HD（1H-2H; ADJSAAM）：`https://exomol.com/data/molecules/H2/1H-2H/ADJSAAM`
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_exomol_diatomic_line_lists.py`
  - H2（1H2; RACPPK；保存先：`data/quantum/sources/exomol_h2_1h2_racppk/`）
    - 取得結果マニフェスト：`data/quantum/sources/exomol_h2_1h2_racppk/manifest.json`（sha256=`FAAE57550855860F038456BB424B27A4B1DC8DDB6D4A229EF88C57BBDF36282F`）
    - 抽出値：`data/quantum/sources/exomol_h2_1h2_racppk/extracted_values.json`（sha256=`B9B72DCFBBD9DCFC9B451B272B6A61975D60E38110B662BC2CEF56037CF0470B`）
    - dataset page：`data/quantum/sources/exomol_h2_1h2_racppk/exomol_dataset_page__1H2__RACPPK.html`（sha256=`A0FA316C2963B735F00CB83924622FB4D654521427637CCAC09894C738BB8C94`）
    - states：`data/quantum/sources/exomol_h2_1h2_racppk/1H2__RACPPK.states.bz2`（sha256=`276F5A36D094E7E1417C44F11C1D173417E92629C5F747B6A862BCCE5C374A81`）
    - trans：`data/quantum/sources/exomol_h2_1h2_racppk/1H2__RACPPK.trans.bz2`（sha256=`B1F180D532E58305C45E353FA80BC8049D26EE931E040CBFC62DDB539053DFC0`）
    - pf：`data/quantum/sources/exomol_h2_1h2_racppk/1H2__RACPPK.pf`（sha256=`A74DEF0D322C4D68D716F51E760F8724B194772915239AF21F3F4CC0141C2BBD`）
  - HD（1H-2H; ADJSAAM；保存先：`data/quantum/sources/exomol_h2_1h_2h_adjsaam/`）
    - 取得結果マニフェスト：`data/quantum/sources/exomol_h2_1h_2h_adjsaam/manifest.json`（sha256=`836D22698D029247CF18D988EB18362B6611709534B737EE3CE88B230452DEB0`）
    - 抽出値：`data/quantum/sources/exomol_h2_1h_2h_adjsaam/extracted_values.json`（sha256=`769FA5C9D6EF04C5EE35C3C822049273565E17F3B0ACC0E55C09738684F64873`）
    - dataset page：`data/quantum/sources/exomol_h2_1h_2h_adjsaam/exomol_dataset_page__1H-2H__ADJSAAM.html`（sha256=`347AA9528A09508556ACED9E78073E9CD517A487BFE08CBEC558C0A0A54A6D02`）
    - states：`data/quantum/sources/exomol_h2_1h_2h_adjsaam/1H-2H__ADJSAAM.states.bz2`（sha256=`B471DEA5D07F08D96CDD14AD922D8A2BD6EB5D55B4B52C0DB478DD3FDA26A74A`）
    - trans：`data/quantum/sources/exomol_h2_1h_2h_adjsaam/1H-2H__ADJSAAM.trans.bz2`（sha256=`C27ACBD56A28ABE7999A971F2615C491F3E13C9F3ADD0E33D555717E1DE8A665`）
    - pf：`data/quantum/sources/exomol_h2_1h_2h_adjsaam/1H-2H__ADJSAAM.pf`（sha256=`EC6BA391FB08A73AE14FF4CFA311032DA6F876FC1DD08658783630FF1A07EF00`）

- MOLAT（OBSPM；分子遷移 line list；D2 FUV emission；ARLSJ1999）：
  - dataset page：`https://molat.obspm.fr/index.php?page=pages/Molecules/D2/D2.php`
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_molat_d2_fuv_emission_lines.py`
  - D2（保存先：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/`）
    - 取得結果マニフェスト：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/manifest.json`（sha256=`A6FB5F38AE6BF97D7B2F986E001C52118008A868AA87D99C619B741883638455`）
    - 抽出値：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/extracted_values.json`（sha256=`F7041A3B32C256AE75D39C872DFBD25F8A9D1BABE170B8918C53E12A3C47CE5F`）
    - dataset page：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_dataset_page__D2.php.html`（sha256=`01F046907DECD5FA92357F09652F129663E44AF53EC2E62A6FFAED1EDC9D2C2F`）
    - view_data（BphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_view_data__BphysB99.html`（sha256=`A693FE473E71E7D3F5C4EBB9C207DB276F3CDC8B34D0A844119C900CCC461377`）
    - lines（BphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_lines__BphysB99.txt`（sha256=`05AA190496CDF7E88ABC8DFA7F9ABC896EECA8D1043CAEC518DC30D4344727F9`）
    - view_data（BpphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_view_data__BpphysB99.html`（sha256=`EE9E1C27F8D2BD47A17C4B4B66AC1D35390A7DC9EC73D61595332795DD20886D`）
    - lines（BpphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_lines__BpphysB99.txt`（sha256=`F72CF4AFE347D0F193E0067980EFF8CB0E87F58C619AC068A246FB5A57EAFAA1`）
    - view_data（CmphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_view_data__CmphysB99.html`（sha256=`4AC30DCB8C0E09D766EFB3967DE027715454F1933A5E4F5DF142E28471155192`）
    - lines（CmphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_lines__CmphysB99.txt`（sha256=`F64BFFCF5CDF0120C7238F0D0BA9F26D3F24CE8199068FAAA28B906D54DCB0E3`）
    - view_data（CpphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_view_data__CpphysB99.html`（sha256=`7422FA158D50014723E37DFFC8F6E7123E6812BF7B0BD26499EE67243D08A51E`）
    - lines（CpphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_lines__CpphysB99.txt`（sha256=`AC5D0425EC05A40C200ED7C8D5D0D1533A1E6CFDAA7E46E385FABD5E51DA1EB8`）
    - view_data（DmphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_view_data__DmphysB99.html`（sha256=`75A41F08BBB19C1864CD5851CF61ECE8EA9EFDBE55D635C641B3064BF8AB8F4C`）
    - lines（DmphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_lines__DmphysB99.txt`（sha256=`D2143DAF4A136CC4715F014CAAD91AF54B2C18ADAF64DD7215829CF841C49378`）
    - view_data（DpphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_view_data__DpphysB99.html`（sha256=`593C221A7008BE0F75B6BF1E026190389CDC6A8B3D6E6C796CC8D380D4FBCFAC`）
    - lines（DpphysB99）：`data/quantum/sources/molat_d2_fuv_emission_arlsj1999/molat_d2_lines__DpphysB99.txt`（sha256=`08AED2DA532D209101CC18ABACCC9BF9B15E5407FBED5ABE0A7082B70E0E98BE`）

- 固定出力（代表遷移；Einstein A top10）：`python -B scripts/quantum/molecular_transitions_exomol_baseline.py`
  - 出力：`output/quantum/molecular_transitions_exomol_baseline.png`（sha256=`D054100472441AD4BA30B60808F3AE437FE166BD66834B0EB01B99FEF4520DFB`）
  - 出力：`output/quantum/molecular_transitions_exomol_baseline_metrics.json`（sha256=`C01388454070F0DC938E2FE6B12B95BE8A21B2E16DBA9D6904F5453BB2049DFD`）
  - 出力：`output/quantum/molecular_transitions_exomol_baseline_selected.csv`（sha256=`2DC3C5652F97A6FE75EEF6DB4BB063E9B57F2E539EA9F54ADD9DC6974857FB17`）

### NIST belltestdata（Shalm et al. 2015 の time-tag 公開データ）
- バケット（S3）：`https://s3.amazonaws.com/nist-belltestdata/`（prefix: `belldata/`）
- file/folder 説明（一次）：`data/quantum/sources/nist_belltestdata/File_Folder_Descriptions.pdf`（sha256=`C1F31E5479BA813773D30F5F599039960CCC00617E40CA6039E9E1B1E4CF4BEC`）
- addendum（一次）：`data/quantum/sources/nist_belltestdata/File_Folder_Descriptions_Addendum_2017_02.pdf`（sha256=`4612B2E223F8BC6E5E88FF151C29FA45BE6B381A6C79BD8836D6D56EC1D5CF28`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nist_belltestdata.py`（保存先：`data/quantum/sources/nist_belltestdata/`）
- 固定（pipeline 用の小さめ subset；training run）：
  - Alice：`data/quantum/sources/nist_belltestdata/compressed/alice/2015_09_18/03_31_CH_pockel_100kHz.run4.afterTimingfix2_training.alice.dat.compressed.zip`（sha256=`1081263A7E2970D73053A0EFA2DB0001065F3D9D80310609AB706FA2BFB68FE9`）
  - Bob：`data/quantum/sources/nist_belltestdata/compressed/bob/2015_09_18/03_31_CH_pockel_100kHz.run4.afterTimingfix2_training.bob.dat.compressed.zip`（sha256=`0FF84C73F4F2AAF879BD1B4E10BC52762A5864214407EF3DC6D5250730228472`）
- 固定（afterfixingModeLocking；training 依存の頑健性確認）：
  - Alice：`data/quantum/sources/nist_belltestdata/compressed/alice/2015_09_18/03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking.alice.dat.compressed.zip`（sha256=`0B4EDE43343A4B0A507CBBB35BC148EA8770E7F1CBB14F5277A7718A1248979B`）
  - Bob：`data/quantum/sources/nist_belltestdata/compressed/bob/2015_09_18/03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking.bob.dat.compressed.zip`（sha256=`C74F6EEEE166E3B960B4C90CE853EFB9594CB4406E4F5CBAFFA027A7FB7E0F17`）
- processed_compressed/hdf5（trial-based；build；click定義のconfig同梱）：
  - 03_43 build：`data/quantum/sources/nist_belltestdata/processed_compressed/hdf5/2015_09_18/03_43_CH_pockel_100kHz.run4.afterTimingfix2_afterfixingModeLocking.dat.compressed.build.hdf5`（sha256=`94AF31F6F8CADDA9ED21FDF1ADB7D6154A05A33BBA551C11619E5DF70B800F06`）
- 固定（追加run；completeblind；multipart：.z01 + .zip を両方使用）：
  - Alice（z01）：`data/quantum/sources/nist_belltestdata/compressed/alice/2015_09_18/17_04_CH_pockel_100kHz.run.completeblind.alice.dat.compressed.z01`（sha256=`35435445A872199D05C3EED1781520507C7133919B85D3BE73A65CD66B89D2D9`）
  - Alice（zip）：`data/quantum/sources/nist_belltestdata/compressed/alice/2015_09_18/17_04_CH_pockel_100kHz.run.completeblind.alice.dat.compressed.zip`（sha256=`31D34791B31A6796F633B5273119EB012F565B37C9742E4FCD644255186CD680`）
  - Bob（z01）：`data/quantum/sources/nist_belltestdata/compressed/bob/2015_09_18/17_04_CH_pockel_100kHz.run.completeblind.bob.dat.compressed.z01`（sha256=`F90394A2FB7F88A598BF2D90C17FFA56D6C5F5A11B52AA8CBDC8F0D8CA65876E`）
  - Bob（zip）：`data/quantum/sources/nist_belltestdata/compressed/bob/2015_09_18/17_04_CH_pockel_100kHz.run.completeblind.bob.dat.compressed.zip`（sha256=`B3F25245C93B804202B9B10B5C7D3156D7C07E62E2B530A0A855747FC10FD750`）
  - 17_04 build：`data/quantum/sources/nist_belltestdata/processed_compressed/hdf5/2015_09_18/17_04_CH_pockel_100kHz.run.completeblind.dat.compressed.build.hdf5`（sha256=`7DF9C85F9B1167F82D943F2DE4B20C1670D973D3BE21F1105C11B119F474A931`）
- 固定（追加run；装置条件依存の切り分け：light cone shift + 200ns addition delay）：
  - Alice：`data/quantum/sources/nist_belltestdata/compressed/alice/2015_09_18/21_15_CH_pockel_100kHz.run.200nsadditiondelay_lightconeshift.alice.dat.compressed.zip`（sha256=`466555AC97AA5AF566AA6D311FB830C057EDB9E0531543370F0F84DF21C70AFD`）
  - Bob：`data/quantum/sources/nist_belltestdata/compressed/bob/2015_09_18/21_15_CH_pockel_100kHz.run.200nsadditiondelay_lightconeshift.bob.dat.compressed.zip`（sha256=`AFB39EA11749D1A2FCF8A6A5AE07B41D1C9CCB4F2D7A3498B0D44C74A905A390`）
  - 21_15 build：`data/quantum/sources/nist_belltestdata/processed_compressed/hdf5/2015_09_18/21_15_CH_pockel_100kHz.run.200nsadditiondelay_lightconeshift.dat.compressed.build.hdf5`（sha256=`67F0B3637374324957FF4594A46E0310FF369653D6C4FA7B8526027D8631B310`）
- 固定（追加run；装置条件依存の切り分け：light cone shift + 200ns reduced delay）：
  - Alice：`data/quantum/sources/nist_belltestdata/compressed/alice/2015_09_18/22_20_CH_pockel_100kHz.run.200nsreduceddelay_lightconeshift.alice.dat.compressed.zip`（sha256=`59C9FB77E92E4985EFCAD0BE79A772C377977C316FD5F6F95020F5632D3A9E5E`）
  - Bob：`data/quantum/sources/nist_belltestdata/compressed/bob/2015_09_18/22_20_CH_pockel_100kHz.run.200nsreduceddelay_lightconeshift.bob.dat.compressed.zip`（sha256=`97AA7BD82EBAECFB7FC37DB3211CCA52E67665E4D47557E208066035A46BBDF5`）
  - 22_20 build：`data/quantum/sources/nist_belltestdata/processed_compressed/hdf5/2015_09_18/22_20_CH_pockel_100kHz.run.200nsreduceddelay_lightconeshift.dat.compressed.build.hdf5`（sha256=`FB51ED6AB541561052901401827CB096EF04FBF5171861CD3C2BBEE9A9CCBC37`）
- 固定（追加run；装置条件依存の切り分け：Classical RNG XOR）：
  - Alice：`data/quantum/sources/nist_belltestdata/compressed/alice/2015_09_18/23_55_CH_pockel_100kHz.run.ClassicalRNGXOR.alice.dat.compressed.zip`（sha256=`D5BBDD1634388FE0A3046139A51DFD2F0E4B61F4765F55FB561102BA9E54B1F9`）
  - Bob：`data/quantum/sources/nist_belltestdata/compressed/bob/2015_09_18/23_55_CH_pockel_100kHz.run.ClassicalRNGXOR.bob.dat.compressed.zip`（sha256=`470ED157D53360D9BE9F4B5556AA62DF55941B57DAD6862A350B56A831EB2D95`）
  - 23_55 build：`data/quantum/sources/nist_belltestdata/processed_compressed/hdf5/2015_09_18/23_55_CH_pockel_100kHz.run.ClassicalRNGXOR.dat.compressed.build.hdf5`（sha256=`A2881BA51EE35988CC69B215C00926C7137606BBDE8D0CB87226585BEEF0DFE6`）
- 取得結果マニフェスト：`data/quantum/sources/nist_belltestdata/manifest.json`

### Delft loophole-free Bell test（Hensen et al. 2015；event-ready / CHSH の公開データ）
- データセット（4TU）：`https://data.4tu.nl/datasets/e8cf2991-3153-48ad-b67d-dfd7d7d97fd3`
  - 直接ダウンロード（data.zip）：`data/quantum/sources/delft_hensen2015/data.zip`（sha256=`C28EB0F075759AEF620A0B2E00A70FD675CE712BC8214D11C103F5A3D29492C0`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_delft_hensen2015.py`（保存先：`data/quantum/sources/delft_hensen2015/`）
- 取得結果マニフェスト：`data/quantum/sources/delft_hensen2015/manifest.json`

### Delft second loophole-free Bell test（Hensen et al. 2016；Sci Rep 6, 30289；event-ready / CHSH の公開データ）
- データセット（4TU）：`https://data.4tu.nl/articles/dataset/Second_loophole-free_Bell_test_with_electron_spins_separated_by_1_3_km/12694403`
  - DOI（dataset）：`10.4121/uuid:53644d31-d862-4f9f-9ad2-0b571874b829`
  - 直接ダウンロード（data.zip）：`data/quantum/sources/delft_hensen2016_srep30289/data.zip`（sha256=`876967E1CBE402B775D25392B773386325499B2343B18BF401D1825F2008A176`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_delft_hensen2016_srep30289.py`（保存先：`data/quantum/sources/delft_hensen2016_srep30289/`）
- 取得結果マニフェスト：`data/quantum/sources/delft_hensen2016_srep30289/manifest.json`

### Giustina et al. 2015（photon；loophole-free；PhysRevLett.115.250401）
- arXiv:1511.03190v2（paper + source + supplement）：
  - PDF：`data/quantum/sources/giustina2015_prl115_250401/arxiv_1511.03190v2.pdf`（sha256=`F2EEA302004982F418FC6AE68AD2AB4EC72D4F67E23D39CBCC61ADD2C6E13684`）
  - arXiv source：`data/quantum/sources/giustina2015_prl115_250401/arxiv_1511.03190v2_src.tar.gz`（sha256=`5FC2429B52E61EC39748BFC84BC11694375D5E37EEF68E6ADFF91C1E36715C57`）
    - 展開先：`data/quantum/sources/giustina2015_prl115_250401/arxiv_src/`
    - supplemental（一次PDF）：`data/quantum/sources/giustina2015_prl115_250401/arxiv_src/anc/supplemental_material_Vienna_20151220.pdf`（sha256=`A44AD4C28A7D4CB639E92119FEF247E5E6AFDA536D51B1A6AB41B9E2726941B4`）
- APS Harvest API（fulltext PDF + article metadata）：
  - fulltext：`data/quantum/sources/giustina2015_prl115_250401/aps_fulltext.pdf`（sha256=`627F3F5A5556969BB24AA25840C83E13B4D5B5A9B727AF75C1BEAAAEC990B0CB`）
  - metadata：`data/quantum/sources/giustina2015_prl115_250401/aps_article.json`（sha256=`762249336706CC6611FB6C477220F9459246412575DA4A8826A2B33C13C30754`）
  - BagIt zip：`data/quantum/sources/giustina2015_prl115_250401/aps_article_bag.zip`（sha256=`7524207E04F135ED0B2C65F82CE18C1766D23F02BDAAE3712154797F9B3DCE01`）
- APS supplemental（手動DL；Cloudflare challenge 回避のためローカルキャッシュ）：
  - supplemental PDF：`data/quantum/sources/giustina2015_prl115_250401/aps_supplemental/Supplemental_material_final.pdf`（sha256=`D03FC645DD1E9C9E788F501401BAB5E49E83761737B6D2EFE9DF6FAD94C50C16`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_giustina2015_prl115_250401.py`（保存先：`data/quantum/sources/giustina2015_prl115_250401/`）
  - 追加（best-effort）：`--include-aps-supplemental`（APS supplemental が取れる環境では `aps_supplemental/` へ保存し、manifest に記録する）
- 取得結果マニフェスト：`data/quantum/sources/giustina2015_prl115_250401/manifest.json`
- 注意：photon time-tag（click log）の一次公開（再解析可能な形）は未固定。APS supplemental（link.aps.org）は環境によって 403 になり得るため、見つからない場合は “保留/ブロック” として明文化して扱う。

### Giustina 2016（University of Vienna；thesis；uTheses/Phaidra）
- Phaidra（Univie）：`https://phaidra.univie.ac.at/o:1331600`（pid=`o:1331600`；Container）
- uTheses（Univie）：`https://utheses.univie.ac.at/detail/39955`（`publication_date=2016`；`fulltext_locked=1` の間は非公開）
- 参照（machine endpoint）：`https://utheses-gateway.univie.ac.at/api/proxy/solrSelect/q=id:39955&rows=10&start=0&wt=json`
- スナップショット（一次メタデータのキャッシュ）：
  - Phaidra info：`data/quantum/sources/giustina2016_utheses_39955/phaidra_o1331600_info.json`
  - Phaidra metadata（thesis_doc_pid=`o:1331601`）：`data/quantum/sources/giustina2016_utheses_39955/phaidra_o1331601_metadata.json`（filename=`39955.pdf`；rights=`InC`）
  - uTheses solr：`data/quantum/sources/giustina2016_utheses_39955/utheses_id39955_solr.json`
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_giustina2016_utheses_39955.py`
- 取得結果マニフェスト：`data/quantum/sources/giustina2016_utheses_39955/manifest.json`
- probe 出力（公開状態監視）：`output/quantum/bell/giustina2016_utheses_probe.json`
- 注意：現時点（snapshot）では `fulltext_locked=1` のため、本文PDF/添付の自動取得はできない（`thesis_doc_pid=o:1331601` は metadata=200 だが detail=404 / download=403）。解除後に `thesis_doc_pid` を追跡して取り込む。

### Weihs et al. 1998（photon；time-tag 公開データ）
- データセット（Zenodo）：`https://zenodo.org/records/7185335`（DOI: 10.5281/zenodo.7185335）
  - Alice.zip：`data/quantum/sources/zenodo_7185335/Alice.zip`（sha256=`E491063C2B366F6F3EB0D2C1A5CECC0535E36F9F0F2C1F5BDC52B6B96F25251B`）
  - Bob.zip：`data/quantum/sources/zenodo_7185335/Bob.zip`（sha256=`21B3011AA25F07F9498FCA50F9BC6E67E7FBDF37A6295CA6B0B4CBF965018B8D`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_weihs1998_zenodo_7185335.py`（保存先：`data/quantum/sources/zenodo_7185335/`）
- 取得結果マニフェスト：`data/quantum/sources/zenodo_7185335/manifest.json`

### Christensen et al. 2013（PRL 111, 130406；Kwiat group；photon；CH；time-tag+trigger）
- 配布元（Illinois QI; BellTest）：`https://research.physics.illinois.edu/QI/BellTest/index.html`
  - データ形式説明（一次）：`data/quantum/sources/kwiat2013_prl111_130406/data_organization.txt`（sha256=`4CB2F3604E68F1574BA4311090A3B03D985EF293AB0533D0938ABAB4FBE07284`）
  - データzip（一次）：`data/quantum/sources/kwiat2013_prl111_130406/CH_Bell_Data.zip`（sha256=`AE85E5EC424439638E93366CC57004E527A9CD333AE6B3B6BAB18489222DD2F4`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_kwiat2013_prl111_130406.py`（保存先：`data/quantum/sources/kwiat2013_prl111_130406/`）
- 取得結果マニフェスト：`data/quantum/sources/kwiat2013_prl111_130406/manifest.json`

### Big Bell Test Collaboration 2018（Nature；human-choice inputs；Source Data for Fig.2）
- 論文（一次）：`https://www.nature.com/articles/s41586-018-0085-3`
- Supplementary Information（一次）：`data/quantum/sources/big_bell_test_2018/supplementary_info.pdf`（sha256=`A9FB80728FB36162F5E8D3D1B68E128C311BB5353155671E0405EB8B185BC3F1`）
- Source Data（一次；Fig.2(a) sessions by country ほか）：`data/quantum/sources/big_bell_test_2018/source_data.xlsx`（sha256=`923FE9E60A2B7C2BA08067B113A69FAE40F8212C422820B58E9AAF8258D30D4A`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_big_bell_test_2018.py`（保存先：`data/quantum/sources/big_bell_test_2018/`）
- 要約（Fig.2a；sessions by country）：`python -B scripts/quantum/big_bell_test_2018_sessions_by_country.py`（出力：`output/quantum/big_bell_test_2018_sessions_by_country.png`）

### 重力×量子干渉（COW / 原子干渉計重力計）
- COW（重力誘起量子干渉）の理論整理（公開PDF）：
  - Mannheim 1996（arXiv:gr-qc/9611037）：`data/quantum/sources/arxiv_gr-qc_9611037.pdf`（sha256=`78254A6EDF26E9B1AD431D13050DE74B9C5E85E312FAECCF3E78951B3359DD60`）
  - 参考（原典；paywallの可能性）：Colella, Overhauser, Werner 1975（Phys. Rev. Lett. 34, 1472；DOI: 10.1103/PhysRevLett.34.1472）
- 原子干渉計重力計（公開PDF）：
  - Mueller et al. 2007（arXiv:0710.3768）：`data/quantum/sources/arxiv_0710.3768.pdf`（sha256=`8F21B9500E19F18CB876A7495974B83852838C29EB91B591C32F7F1895E7EFF4`）
- 量子時計（光格子時計；chronometric leveling の一次結果が要旨に数値として明示）：
  - arXiv:2309.14953v3 “Long-distance chronometric leveling with a portable optical clock”：`data/quantum/sources/arxiv_2309.14953v3.pdf`（sha256=`ACB4A20342662C6993F761FEC1AAE05730E26B692D0DCF099471748B0B6B8196`）

### 原子核（deuteron ベースライン；NIST CODATA 定数）
- NIST Cuu（CODATA）定数ページ（一次；HTMLを固定してoffline解析する）：
  - mp（proton mass；`https://physics.nist.gov/cgi-bin/cuu/Value?mp`）：`data/quantum/sources/nist_codata_2022_nuclear_baseline/nist_cuu_Value_mp.html`（sha256=`750FB1A7CFA8ED9E4267039B4C7C0F7098A32B8A45BB15F32C2D7024CF6448C4`）
  - mn（neutron mass；`https://physics.nist.gov/cgi-bin/cuu/Value?mn`）：`data/quantum/sources/nist_codata_2022_nuclear_baseline/nist_cuu_Value_mn.html`（sha256=`BC4FB2E60DF98D015C9C77230D30C15675CF65F5DC3BE88629FCF830B0544A89`）
  - md（deuteron mass；`https://physics.nist.gov/cgi-bin/cuu/Value?md`）：`data/quantum/sources/nist_codata_2022_nuclear_baseline/nist_cuu_Value_md.html`（sha256=`47CF177123224C321EDF81C8C811C93B38D334BA94DBE9624FD32B4C359A9E30`）
  - rd（deuteron rms charge radius；`https://physics.nist.gov/cgi-bin/cuu/Value?rd`）：`data/quantum/sources/nist_codata_2022_nuclear_baseline/nist_cuu_Value_rd.html`（sha256=`66304C67C79E3ACFEDD7B42FDFEA5E314253CBDDE0551C8E8C59D91898DF51A8`）
- 抽出（派生；HTMLから値を正規化してJSON化）：`data/quantum/sources/nist_codata_2022_nuclear_baseline/extracted_values.json`（sha256=`F66AEF6422CE5A6E6D9653545A4DECD10910923774B19E876132309631B9E245`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nuclear_binding_sources.py`（保存先：`data/quantum/sources/nist_codata_2022_nuclear_baseline/`）
- 取得結果マニフェスト：`data/quantum/sources/nist_codata_2022_nuclear_baseline/manifest.json`（sha256=`A4AF996C9EBA1F2DFF6C6A73923182E26754B75DDBF3EB36263359DFB43D2A2A`）
- ベースライン解析（deuteron；束縛エネルギーBとサイズ拘束を固定出力化）：`python -B scripts/quantum/nuclear_binding_deuteron.py`
  - 出力：`output/quantum/nuclear_binding_deuteron.png` / `output/quantum/nuclear_binding_deuteron_metrics.json`

### 原子核（軽核 A=2,3,4 ベースライン；NIST CODATA 定数）
- NIST Cuu（CODATA）定数ページ（一次；HTMLを固定してoffline解析する）：
  - mp（proton mass；`https://physics.nist.gov/cgi-bin/cuu/Value?mp`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_mp.html`（sha256=`C1B774E1495EB54A3A020D1CB7929F8BB2372D5AA2741EB2DA14E578DE1D0CE7`）
  - mn（neutron mass；`https://physics.nist.gov/cgi-bin/cuu/Value?mn`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_mn.html`（sha256=`ADC5DB75C5AC63D0F72BC2B3F07E9A63B4B918E4E5E0D267CB7F0F21BA83EF18`）
  - md（deuteron mass；`https://physics.nist.gov/cgi-bin/cuu/Value?md`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_md.html`（sha256=`D8A222DE94C17A2B666A1DD1212B1735649E62212B6D77FE7D44569A8F6E0025`）
  - mt（triton mass；`https://physics.nist.gov/cgi-bin/cuu/Value?mt`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_mt.html`（sha256=`10BF7988E46814E358937E2CBBCF23FC2C2464AAE3856BD018EC51DE2A32A700`）
  - mh（helion mass；`https://physics.nist.gov/cgi-bin/cuu/Value?mh`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_mh.html`（sha256=`460A0F7374C9C0D6B458A07AACC6370B665961E1F9E5398711CB9C6CBBE6A51F`）
  - mal（alpha particle mass；`https://physics.nist.gov/cgi-bin/cuu/Value?mal`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_mal.html`（sha256=`7BF7ABE432476637AA70EAEBA3A0EB4E8C00312A65CBAA45937FC4F4ADBAD508`）
  - rp（proton rms charge radius；`https://physics.nist.gov/cgi-bin/cuu/Value?rp`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_rp.html`（sha256=`96064B345D7B2DFFA79A0C5B9E2ED61D8DE01629598ACA6A38B18FDD11FE1329`）
  - rd（deuteron rms charge radius；`https://physics.nist.gov/cgi-bin/cuu/Value?rd`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_rd.html`（sha256=`84E1B520A739F1B865ED49A79E310CFB23D1DF30AC4BD27335C375B89E21FD42`）
  - ral（alpha particle rms charge radius；`https://physics.nist.gov/cgi-bin/cuu/Value?ral`）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/nist_cuu_Value_ral.html`（sha256=`1C6712B5637429EBDF8A7EF49EEA97A13938625ECA4595A117105E02BD4AF178`）
- 抽出（派生；HTMLから値を正規化してJSON化）：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/extracted_values.json`（sha256=`989AD72E5FD2B0B2DC1E9E161002B574F29B9B34B27FFD1D7E33C1DB07DCE9B0`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nuclear_binding_sources.py --out-dirname nist_codata_2022_nuclear_light_nuclei --include-light-nuclei`
- 取得結果マニフェスト：`data/quantum/sources/nist_codata_2022_nuclear_light_nuclei/manifest.json`（sha256=`0BCE791526A5B358D996491CC556AC693C57BAC4C3AD0B6F9D98BDF00CD22A6B`）
- A=3（t/h）電荷半径（専用コンパイル；一次）：
  - IAEA radii database（page）：`https://www-nds.iaea.org/radii/`（ref DOI: 10.1016/j.adt.2011.12.006）
  - charge_radii.csv（一次；固定してoffline解析する）：`data/quantum/sources/iaea_charge_radii/charge_radii.csv`（sha256=`683D028DAE93235376DDF7A345F736561EE7C0DBA070091912C034BDF740EEE6`）
  - 抽出（派生；A=3のt/hのみをJSON化）：`data/quantum/sources/iaea_charge_radii/extracted_values.json`（sha256=`4D3D15B03A520FCDD15B4D6F422C2618F5BB3F4D93D414BF1974F3E574049E84`）
  - 取得結果マニフェスト：`data/quantum/sources/iaea_charge_radii/manifest.json`（sha256=`6C2DD8CCCDD5AA524D0FEA9609CD0188EC45CD62A65B98D137F47117F06A5428`）
  - 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nuclear_charge_radii_sources.py --out-dirname iaea_charge_radii`
- ベースライン解析（軽核；A=2,3,4 の束縛エネルギーと利用可能な電荷半径を固定出力化）：`python -B scripts/quantum/nuclear_binding_light_nuclei.py`
  - 出力：`output/quantum/nuclear_binding_light_nuclei.png` / `output/quantum/nuclear_binding_light_nuclei_metrics.json` / `output/quantum/nuclear_binding_light_nuclei.csv`

### 物性（凝縮系；Si格子定数；NIST CODATA 定数）
- NIST Cuu（CODATA）定数ページ（一次；HTMLを固定してoffline解析する）：
  - asil（lattice parameter of silicon；`https://physics.nist.gov/cgi-bin/cuu/Value?asil`）：`data/quantum/sources/nist_codata_2022_silicon_lattice/nist_cuu_Value_asil.html`（sha256=`FE14057CC8A9B02596F43962F9EC36024CD5F099C60B433503F52C428F3F05ED`）
  - d220sil（lattice spacing of ideal Si (220)；`https://physics.nist.gov/cgi-bin/cuu/Value?d220sil`）：`data/quantum/sources/nist_codata_2022_silicon_lattice/nist_cuu_Value_d220sil.html`（sha256=`D3BA898215A2A2CD002D61943109CAA38F5D445425C6DA739F4862CD337E194A`）
- 抽出（派生；HTMLから値を正規化してJSON化）：`data/quantum/sources/nist_codata_2022_silicon_lattice/extracted_values.json`（sha256=`0060FBA3290185382C3E0895A8B3022B830CA6DC81F66E90C0AABCD292710FB2`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_lattice_sources.py`（保存先：`data/quantum/sources/nist_codata_2022_silicon_lattice/`）
- 取得結果マニフェスト：`data/quantum/sources/nist_codata_2022_silicon_lattice/manifest.json`（sha256=`B8F3F8FE756D2C4C4B4BA5D60BC2BAAD4C299387826E51994F02C0A37E4492BB`）
- ベースライン解析（基準値の固定出力）：`python -B scripts/quantum/condensed_silicon_lattice_baseline.py`
  - 出力：`output/quantum/condensed_silicon_lattice_baseline.png` / `output/quantum/condensed_silicon_lattice_baseline_metrics.json` / `output/quantum/condensed_silicon_lattice_baseline.csv`

### 統計力学・熱力学（黒体放射の基準量；NIST CODATA 定数）
- NIST Cuu（CODATA）定数ページ（一次；HTMLを固定してoffline解析する）：
  - c（speed of light；`https://physics.nist.gov/cgi-bin/cuu/Value?c`）：`data/quantum/sources/nist_codata_2022_blackbody_constants/nist_cuu_Value_c.html`（sha256=`0F514505759A90D8AC36F49595B2AC645AE5E808457DF254A7BE8909C11C19E6`）
  - h（Planck constant；`https://physics.nist.gov/cgi-bin/cuu/Value?h`）：`data/quantum/sources/nist_codata_2022_blackbody_constants/nist_cuu_Value_h.html`（sha256=`69F33BD71CF144CE25F09FD9BF673D5A5951D4762A692022E2F6F04401F1692B`）
  - hbar（reduced Planck constant；`https://physics.nist.gov/cgi-bin/cuu/Value?hbar`）：`data/quantum/sources/nist_codata_2022_blackbody_constants/nist_cuu_Value_hbar.html`（sha256=`3972553679A39E94EE23075221E5012EB9E60F7571679960570147DC7CEF502A`）
  - k（Boltzmann constant；`https://physics.nist.gov/cgi-bin/cuu/Value?k`）：`data/quantum/sources/nist_codata_2022_blackbody_constants/nist_cuu_Value_k.html`（sha256=`A8D0E4CDCCE8C081DACE0D73B427CB5A5F68E417CAF86DEF79C282308581C04E`）
  - sigma（Stefan–Boltzmann constant；`https://physics.nist.gov/cgi-bin/cuu/Value?sigma`）：`data/quantum/sources/nist_codata_2022_blackbody_constants/nist_cuu_Value_sigma.html`（sha256=`68C1E36FC80DCE66AB6D9A0CC7B5BEA8C28856887B533FE0D536AF1768BC98C7`）
- 抽出（派生；HTMLから値を正規化してJSON化）：`data/quantum/sources/nist_codata_2022_blackbody_constants/extracted_values.json`（sha256=`FFE4C2234B9BF48C59898745BD5688A50A9F314E5E1EEC63E172C24486032742`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_blackbody_constants_sources.py`（保存先：`data/quantum/sources/nist_codata_2022_blackbody_constants/`）
- 取得結果マニフェスト：`data/quantum/sources/nist_codata_2022_blackbody_constants/manifest.json`（sha256=`AAFA85CB8FC5F381CB7DCBB26DAA41F608F9D678092EC2B5D3C04FF687D23801`）
- ベースライン解析（黒体放射の基準量を固定出力）：`python -B scripts/quantum/thermo_blackbody_radiation_baseline.py`
  - 出力：`output/quantum/thermo_blackbody_radiation_baseline.png` / `output/quantum/thermo_blackbody_radiation_baseline_metrics.json` / `output/quantum/thermo_blackbody_radiation_baseline.csv`

### 物性（凝縮系；Si比熱 Cp(T)；NIST Chemistry WebBook condensed）
- NIST Chemistry WebBook（一次；HTMLを固定してoffline解析する）：
  - silicon（Si; ID=C7440213; Thermo-Condensed）：`https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440213&Units=SI&Mask=2#Thermo-Condensed`
  - 保存：`data/quantum/sources/nist_webbook_condensed_silicon_si/nist_webbook_condensed_si_mask2.html`（sha256=`EBADD4FA0C507C690B11F9D11C20C281BA179FC728EAB349137E09C3FFA08443`）
- 抽出（派生；solid/liquid の Shomate 係数をJSON化）：`data/quantum/sources/nist_webbook_condensed_silicon_si/extracted_values.json`（sha256=`BE06A252B49767F17EF18D4B93F23A039CC484FDCBA8456B3196013124BA7EE3`）
- 取得結果マニフェスト：`data/quantum/sources/nist_webbook_condensed_silicon_si/manifest.json`（sha256=`07CF2C41BD6F8EA9144DF05A44367FE6DCBF00D13C7960EA4AEAADDAA116F695`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_condensed_thermochemistry_sources.py`
- ベースライン解析（Cp(T)の固定出力）：`python -B scripts/quantum/condensed_silicon_heat_capacity_baseline.py`
  - 出力：`output/quantum/condensed_silicon_heat_capacity_baseline.png` / `output/quantum/condensed_silicon_heat_capacity_baseline_metrics.json` / `output/quantum/condensed_silicon_heat_capacity_baseline.csv`

### 物性（凝縮系；Si低温比熱 Cp(T)；NIST-JANAF Si-004）
- NIST-JANAF table（一次；HTML/txt を固定してoffline解析する）：
  - Si-004（HTML）：`https://janaf.nist.gov/tables/Si-004.html` → `data/quantum/sources/nist_janaf_silicon_si/nist_janaf_Si-004.html`（sha256=`AA636A0FB927B40ED40CCEEAC8D3F120A39E7119B8A6B472C978D5BC0564928F`）
  - Si-004（txt）：`https://janaf.nist.gov/tables/Si-004.txt` → `data/quantum/sources/nist_janaf_silicon_si/nist_janaf_Si-004.txt`（sha256=`E02460B88490D505E7CB433D8F0066CCFFC9CAD61119E4BD5C558BE8CA045BFB`）
- 抽出（派生；Cp表をJSON化；相ラベルは “CRYSTAL <--> LIQUID” 行から割当）：`data/quantum/sources/nist_janaf_silicon_si/extracted_values.json`（sha256=`DA4E1BC5AD80067347B68BA2F8D2097570E2A9262AC79120F7606C5F0731EA09`）
- 取得結果マニフェスト：`data/quantum/sources/nist_janaf_silicon_si/manifest.json`（sha256=`2CB4473E6029ED6989371DD9F933CC045D726F37A775942CAC82A3FB7AB0D597`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_janaf_sources.py`
- ベースライン解析（Debye（θ_D）fit で “圧縮したターゲット” を固定出力）：`python -B scripts/quantum/condensed_silicon_heat_capacity_debye_baseline.py`
  - 出力：`output/quantum/condensed_silicon_heat_capacity_debye_baseline.png` / `output/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json` / `output/quantum/condensed_silicon_heat_capacity_debye_baseline.csv`

### 物性（凝縮系；Si抵抗率 ρ(T)；NBS IR 74-496 Appendix E）
- NIST/NVL pubs（Legacy IR；一次PDFを固定してoffline解析する）：
  - NBS IR 74-496（PDF）：`https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nbsir74-496.pdf` → `data/quantum/sources/nist_nbsir74_496_silicon_resistivity/nbsir74-496.pdf`（sha256=`0CA3BA5E35489777075F4354F86BF127B3F54D7BC5E554D5AA75FB355F10EE2D`）
- 抽出（派生；Appendix E の ρ(T)（0–50°C）range table をJSON化）：`data/quantum/sources/nist_nbsir74_496_silicon_resistivity/extracted_values.json`（sha256=`B4F79B3176F3F0F305BE23F902EA6281D8A07C8F7B46BBEF12559E6E7DDD6BDF`）
- 取得結果マニフェスト：`data/quantum/sources/nist_nbsir74_496_silicon_resistivity/manifest.json`（sha256=`CAB48FC333F05D34BB7C56B46CB7B6C89C4F4532B4294B4B69385C2A511117B3`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_resistivity_nbsir74_496_sources.py`
- ベースライン解析（室温近傍の温度係数を固定出力）：`python -B scripts/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline.py`
  - 出力：`output/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline.png` / `output/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline_metrics.json` / `output/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline.csv`

### 物性（凝縮系；Si熱膨張係数 α(T)；NIST TRC cryogenics）
- NIST TRC Cryogenics（一次；HTML+画像を固定してoffline解析する）：
  - HTML：`https://trc.nist.gov/cryogenics/materials/Silicon/Silicon.htm` → `data/quantum/sources/nist_trc_silicon_thermal_expansion/Silicon.htm`（sha256=`0FA66122C3295AEA81FDEA31015C08D0C5ABA9991E91A024B8143FA3A731064B`）
  - equation：`https://trc.nist.gov/cryogenics/materials/Silicon/te.png` → `data/quantum/sources/nist_trc_silicon_thermal_expansion/te.png`（sha256=`012C53D027E2A872BC399DAB03238B692BC7715A11557BF4D6425E9AF557DAAF`）
  - plot：`https://trc.nist.gov/cryogenics/materials/Silicon/Siliconplot.png` → `data/quantum/sources/nist_trc_silicon_thermal_expansion/Siliconplot.png`（sha256=`CEDA3C21087E04E8DD677A46AFFD094C0B7D54B8813819A9ACCAF1967BD113D3`）
  - temp note：`https://trc.nist.gov/cryogenics/materials/Silicon/temp.gif` → `data/quantum/sources/nist_trc_silicon_thermal_expansion/temp.gif`（sha256=`07EB2B0B987057B5B6770A96216EB0E320F55CD4AE9C4977818F6AA571BAEC53`）
- 抽出（派生；係数 a–l と範囲/誤差行をJSON化）：`data/quantum/sources/nist_trc_silicon_thermal_expansion/extracted_values.json`（sha256=`DA3F26782DDC549B6AD881F4E690905EC53B4E1798925AFFB581108E3B40E731`）
- 取得結果マニフェスト：`data/quantum/sources/nist_trc_silicon_thermal_expansion/manifest.json`（sha256=`18CB81C3B6DA16AE94B54EFAC77F11BB56622E249C008908A67DA4F315F2D753`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_thermal_expansion_sources.py`
- ベースライン解析（α(T)の固定出力）：`python -B scripts/quantum/condensed_silicon_thermal_expansion_baseline.py`
  - 出力：`output/quantum/condensed_silicon_thermal_expansion_baseline.png` / `output/quantum/condensed_silicon_thermal_expansion_baseline_metrics.json` / `output/quantum/condensed_silicon_thermal_expansion_baseline.csv`

### 物性（凝縮系；Si弾性定数 Cij / バルク弾性率 B(T)；Ioffe semiconductor database）
- Ioffe（半導体物性DB；弾性定数と線形近似式を含む）：
  - mechanic.html：`https://www.ioffe.ru/SVA/NSM/Semicond/Si/mechanic.html` → `data/quantum/sources/ioffe_silicon_mechanical_properties/ioffe_mechanic.html`（sha256=`BA792BBB92A088E2F56813083DA87CDE9405FD0849D9F07F8B7CA6C85D2A3DA4`）
  - reference.html：`https://www.ioffe.ru/SVA/NSM/Semicond/Si/reference.html` → `data/quantum/sources/ioffe_silicon_mechanical_properties/ioffe_reference.html`（sha256=`E5D16C864C493494906C5ACF4B68914AD393EDF35BCF6035ADA126AE7B9D730F`）
- 抽出（派生；C11/C12/C44 と 400–873K の線形近似式＋代表 phonon 周波数をJSON化）：`data/quantum/sources/ioffe_silicon_mechanical_properties/extracted_values.json`（sha256=`5AA209CD7E7174A1D272BB5FA31EEB0B153BC6EF88938D3B0D44E21344EB8561`）
- 取得結果マニフェスト：`data/quantum/sources/ioffe_silicon_mechanical_properties/manifest.json`（sha256=`1776B7AFA8BB62E3E8FE5F257B0812C927B512CAF727759611164B11761143D4`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_elastic_constants_sources.py`

### 物性（凝縮系；Si phonon DOS（ω–D(ω)）proxy；public HTML table）
- public HTML（数値テーブルを含む；一次文献の明示は無いため proxy として扱う）：
  - HTML：`https://lampz.tugraz.at/~hadley/ss1/phonons/dos/si_phonon_dos.html` → `data/quantum/sources/hadley_si_phonon_dos/si_phonon_dos.html`（sha256=`7135DDE4945243B30A8E8B739E1E07A0406E5E29564FB14B9FCB505C71C90811`）
- 抽出（派生；<textarea> の ω–D(ω) をJSON化＋積分量・half-integral split を保存）：`data/quantum/sources/hadley_si_phonon_dos/extracted_values.json`（sha256=`322E42459408B8C66A2FD8515432E2E3FBB98B196DED3AA781E81E7CFAC11161`）
- 取得結果マニフェスト：`data/quantum/sources/hadley_si_phonon_dos/manifest.json`（sha256=`D62BC1D3BAEAA73C58355275F0818FC14DCBD62FA473F4C0E9C73FD042109447`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_phonon_dos_sources.py`
- 注：Step 7.14.20 では、mode-weighting（Cvの重み）を “一次（またはそれに準ずる）” で凍結する目的で使用する。正式な一次（INS/IXS の raw / supp）を取得できた段階で差し替える。

### 物性（凝縮系；Si phonon DOS の温度依存 / anharmonicity；INS一次；Kim et al. PRB 91, 014307 (2015)）
- OSTI（accepted manuscript; INSで 100–1500 K の phonon DOS を測定）：
  - landing：`https://www.osti.gov/pages/biblio/1185765-phonon-anharmonicity-silicon-from` → `data/quantum/sources/osti_kim2015_prb91_014307_si_phonon_anharmonicity/osti_biblio_1185765.html`（sha256=`A274CB3EC2B13E18400B6B1806C2B8BF81DDF3EBB143171A443044577EE89B3D`）
  - PDF：`https://www.osti.gov/servlets/purl/1185765` → `data/quantum/sources/osti_kim2015_prb91_014307_si_phonon_anharmonicity/osti_purl_1185765.pdf`（sha256=`E783E828BFAEC7933744248663C011DE883187EA0C08EAF3DBD7F477AC308442`）
  - DOI：10.1103/PhysRevB.91.014307
- CaltechAUTHORS（published PDF のオープンミラー；図の抽出用途。`download=1` は短寿命の S3 redirect になり得るため、スクリプトは cache-buster を付与する）：
  - landing：`https://authors.library.caltech.edu/records/67hvq-njk27/latest` → `data/quantum/sources/caltechauthors_kim2015_prb91_014307_si_phonon_anharmonicity/caltechauthors_record_67hvq-njk27.html`（sha256=`F00CBB0BD41FD2F0F0AF324C01B65E17B8812DD210FD4F6587A924351943EFA3`）
  - PDF：`https://authors.library.caltech.edu/records/67hvq-njk27/files/PhysRevB.91.014307.pdf?download=1` → `data/quantum/sources/caltechauthors_kim2015_prb91_014307_si_phonon_anharmonicity/PhysRevB.91.014307.pdf`（sha256=`81D08CB45C350AE1964A9F1103306B387845E8AAEF3C488C67A01F8D6E15C79C`）
  - DOI：10.1103/PhysRevB.91.014307
- 抽出（派生；PDF本文から mean shift と softening proxy を機械抽出してJSON化）：`data/quantum/sources/osti_kim2015_prb91_014307_si_phonon_anharmonicity/extracted_values.json`（sha256=`57FC7C392BE0C65BC8969E750666DD5E0E6FBBC04D1814402CDF1747B872A499`）
- 取得結果マニフェスト：`data/quantum/sources/osti_kim2015_prb91_014307_si_phonon_anharmonicity/manifest.json`（sha256=`A83EEF22CD31D357165041E528D29C99018EF6BE18027CCEF07A3D9ADB1BC03D`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_phonon_dos_sources.py --source osti_kim2015_prb91_014307`
- 抽出（softening proxy 追加）：`python -B scripts/quantum/extract_silicon_phonon_anharmonicity_kim2015_softening_proxy.py`
  - 参考：published PDF mirror を固定（派生なし）：`data/quantum/sources/caltechauthors_kim2015_prb91_014307_si_phonon_anharmonicity/extracted_values.json`（sha256=`69433D3EB09D7724F2A258F431F6E818B77D055E0E322A4DB23D5D2051C90217`）
  - 参考：published PDF mirror のマニフェスト：`data/quantum/sources/caltechauthors_kim2015_prb91_014307_si_phonon_anharmonicity/manifest.json`（sha256=`061C4E187C03A50B019259697879A1B8204B78F25DC40E4585AFDD4E1D2C5E0C`）

### 物性（凝縮系；Si Raman ω(T)（光学phonon；~520 cm⁻¹））
- arXiv（一次PDF；Ramanで 4–623 K の ω(T) を測定）：  
  - PDF：`https://arxiv.org/pdf/2001.08458.pdf` → `data/quantum/sources/arxiv_2001_08458_si_raman_phonon_shift/arxiv_2001_08458v1.pdf`（sha256=`611A79E3B68CBBA0E2162466A7E291D6C4B6A0B37DAD0C1EC4CD84D08AEC3BB4`）
- 抽出（派生；PDFのベクタ図から ω(T) マーカーをdigitizeし、softening_shape_0to1(T) を保存）：`data/quantum/sources/arxiv_2001_08458_si_raman_phonon_shift/extracted_values.json`（sha256=`8E3A69B366B2C965B2B9E432260E5FA2F6DA2B72BBAC13B0677CDB36AE5A04B2`）
- 取得結果マニフェスト：`data/quantum/sources/arxiv_2001_08458_si_raman_phonon_shift/manifest.json`（sha256=`4DBDBABC263F871F08EA3BAC20581F5EA26686AC9CCBC87A9675011BB495443B`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_silicon_raman_phonon_shift_sources.py`

### 物性（凝縮系；OFHC Copper 熱伝導率 κ(T)；NIST TRC cryogenics）
- NIST TRC Cryogenics（一次；HTMLを固定してoffline解析する）：
  - HTML：`https://trc.nist.gov/cryogenics/materials/OFHC%20Copper/OFHC_Copper_rev1.htm` → `data/quantum/sources/nist_trc_ofhc_copper_thermal_conductivity/OFHC_Copper_rev1.htm`（sha256=`9B4134273133584A4BF3FB855DE7F5E180F4A9A18BF79C2C36076E0FDB0677B2`）
- 抽出（派生；RRR別の係数 a–i と範囲/誤差行をJSON化）：`data/quantum/sources/nist_trc_ofhc_copper_thermal_conductivity/extracted_values.json`（sha256=`64C84F5035551A06B039D2555A6A649BE9CB3F93B5B775DC300DCDB026EE6D8C`）
- 取得結果マニフェスト：`data/quantum/sources/nist_trc_ofhc_copper_thermal_conductivity/manifest.json`（sha256=`D986F82A095C5EB1EBCFA70F90A5BADA38611010D5955A577B8F5E70C1987BA8`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_ofhc_copper_thermal_conductivity_sources.py`
- ベースライン解析（κ(T)の固定出力）：`python -B scripts/quantum/condensed_ofhc_copper_thermal_conductivity_baseline.py`
  - 出力：`output/quantum/condensed_ofhc_copper_thermal_conductivity_baseline.png` / `output/quantum/condensed_ofhc_copper_thermal_conductivity_baseline_metrics.json` / `output/quantum/condensed_ofhc_copper_thermal_conductivity_baseline.csv`

### 原子核（代表核種 A 依存ベースライン；AME2020 + IAEA radii）
- AME2020 atomic mass evaluation（IAEA AMDC；一次；mass tableを固定してoffline解析する）：
  - mass_1.mas20.txt：`https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt` → `data/quantum/sources/iaea_amdc_ame2020_mass_1_mas20/mass_1.mas20.txt`（sha256=`E8599C6D7F724FAC91934E59F1B9DE8FB8F63E820F4B39456B790665ED2A3307`）
  - 抽出（派生；mass_1.mas20 から B/A 等をJSON化）：`data/quantum/sources/iaea_amdc_ame2020_mass_1_mas20/extracted_values.json`（sha256=`612448CECF66277B94672A3F389B6D71EBF5FE24861B5FCE6CBB7555F6990565`）
  - 取得結果マニフェスト：`data/quantum/sources/iaea_amdc_ame2020_mass_1_mas20/manifest.json`（sha256=`82DD41EDFCA3594B76B853A44B808BC3C2A70DF19C96204842E86990DD90AA59`）
  - 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_ame2020_mass_table_sources.py --out-dirname iaea_amdc_ame2020_mass_1_mas20`
- IAEA radii database（電荷半径；一次；固定してoffline解析する）：`data/quantum/sources/iaea_charge_radii/charge_radii.csv`（sha256=`683D028DAE93235376DDF7A345F736561EE7C0DBA070091912C034BDF740EEE6`）
- ベースライン解析（代表核種；A依存の束縛エネルギー/核子あたり束縛と電荷半径を固定出力化）：`python -B scripts/quantum/nuclear_binding_representative_nuclei.py`
  - 出力：`output/quantum/nuclear_binding_representative_nuclei.png` / `output/quantum/nuclear_binding_representative_nuclei_metrics.json` / `output/quantum/nuclear_binding_representative_nuclei.csv`

### 原子核（構造入力：変形パラメータ β2；NNDC B(E2) adopted）
- NNDC B(E2)（一次配布：adopted entries JSON；β2（dimensionless）を含む）：
  - ページ：`https://www.nndc.bnl.gov/be2/`
  - adopted-entries.json：`https://www.nndc.bnl.gov/be2/data/adopted-entries.json` → `data/quantum/sources/nndc_be2_adopted_entries/adopted-entries.json`
  - 抽出（派生；nuclideごとにβ2を一意化してJSON化）：`data/quantum/sources/nndc_be2_adopted_entries/extracted_beta2.json`
  - 抽出（派生；transitionEnergy を E(2+_1) proxy（keV）としてJSON化）：`data/quantum/sources/nndc_be2_adopted_entries/extracted_e2plus.json`
  - 取得結果マニフェスト：`data/quantum/sources/nndc_be2_adopted_entries/manifest.json`
  - 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nuclear_deformation_be2_sources.py`
  - 注：NNDCページは Pritychenko et al. (ADNDT 107, 1 (2016)) と Raman et al. (ADNDT 78, 1 (2001)) を引用している。本リポジトリでは NNDC 配布の JSON を固定して offline 再現する。

### 原子核（構造入力：分光（E(2+_1), E(4+_1), E(3-_1), R4/2）；NNDC NuDat 3.0）
- NNDC NuDat 3.0（一次配布：chart data JSON；励起エネルギーを含む）：
  - ページ：`https://www.nndc.bnl.gov/nudat3/`
  - primary.json：`https://www.nndc.bnl.gov/nudat3/data/primary.json` → `data/quantum/sources/nndc_nudat3_primary_secondary/primary.json`（sha256=`8FE4B48041C42AC01E8BC5DAA2AEDD51849161D0AC8E098F37039FC388F3E796`）
  - secondary.json：`https://www.nndc.bnl.gov/nudat3/data/secondary.json` → `data/quantum/sources/nndc_nudat3_primary_secondary/secondary.json`（sha256=`EB1D29C38C2D6F510395938E856F421AD737D6E0BBCD1CC8E26C999C6FF2AE75`）
  - 抽出（派生；E(2+_1) を一意化してJSON化）：`data/quantum/sources/nndc_nudat3_primary_secondary/extracted_e2plus.json`（sha256=`AC8F70D6C3C3CE0552F476D005D7F83BD51A5E35A5BD8B8FB5314E94355E5372`）
  - 抽出（派生；E(2+_1), E(4+_1), E(3-_1), R4/2 をJSON化）：`data/quantum/sources/nndc_nudat3_primary_secondary/extracted_spectroscopy.json`（sha256=`095101891A1EC8F36FE3EB53EB3E8C05E5B9A32A0299F10993D13020A6FFAF20`）
  - 取得結果マニフェスト：`data/quantum/sources/nndc_nudat3_primary_secondary/manifest.json`（sha256=`548FEC1297446C435AA9D4F2FF8F801FEA916F9FE8A7222297D1B58A1AE41ABD`）
  - 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nuclear_spectroscopy_e2plus_sources.py`
  - 注：本リポジトリでは `secondary.json` の `excitedStateEnergies`（keV）から E(2+_1), E(4+_1), E(3-_1) と、比 R4/2 を抽出して使用する。詳細の扱い（selection/peak rule）は Phase 7 / Step 7.13.15.* の strict プロトコルに従う。

### np散乱（低エネルギー：散乱長/有効レンジ；有効レンジ展開）
- 低エネルギーパラメータ（a_t, r_t, v2t / a_s, r_s, v2s）を、同一一次ソース内の複数手法（legacy vs phase-shift analyses）で比較できる形で固定する。
- 一次（arXiv:0704.1024v1；PDF + source tarball）：
  - PDF：`data/quantum/sources/arxiv_0704.1024v1.pdf`（sha256=`1AC200A683110B39013832DA99981B5404CA009DF3A83DA56A67FE7548458D25`）
  - arXiv source：`data/quantum/sources/arxiv_0704.1024v1_src.tar.gz`（sha256=`AC499C9AC1DC00BA07457C2377426658E0F9B3829E37D6FB5EFF7E84B6C66BD1`）
- 抽出（派生；eq.(16–19) を source から機械抽出してJSON化）：`data/quantum/sources/np_scattering_low_energy_arxiv_0704_1024v1_extracted.json`（sha256=`1508CDFE8A94B8322348F6E944FE51F3D3047D3ED620BFBE2F490D4C31AA1709`）
- 取得結果マニフェスト：`data/quantum/sources/np_scattering_low_energy_arxiv_0704_1024v1_manifest.json`（sha256=`EB2AEC45CE182215D88CA7BA32C5B8BD62DE9AB9820EE66F869B13DC64F6BE80`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_nuclear_np_scattering_sources.py`
- ベースライン解析（np散乱；deuteron B との整合チェックを含めて固定出力化）：`python -B scripts/quantum/nuclear_np_scattering_baseline.py`
  - 出力：`output/quantum/nuclear_np_scattering_baseline.png`（sha256=`0CDB99BBBA3394B8D37CC39BB56CDCBAD32F2D9A52454E37C3A4419FE1D190B5`）
  - 出力：`output/quantum/nuclear_np_scattering_baseline_metrics.json`（sha256=`CF61439F8ED677782CA78A5289C79F14CCAA6D7380785C872A99B5DB9D276E69`）
- 有効方程式（square-well；`(B,a_t)` を拘束にして `r_t` を予言し、棄却条件を固定）：`python -B scripts/quantum/nuclear_effective_potential_square_well.py`
  - 出力：`output/quantum/nuclear_effective_potential_square_well.png`（sha256=`B20EAE91887A8BD6FA7804F4F2F0F2E546A9AA1F737DC215B8A60C54D0705F92`）
  - 出力：`output/quantum/nuclear_effective_potential_square_well_metrics.json`（sha256=`69017A1889295B2265B1613DCC0B9849279918A09968798B5BE34F017A576D7D`）
- 有効方程式（core+well；triplet を `(B,a_t,r_t)` で拘束し、`v2t` と singlet の `(r_s,v2s)` を予言して棄却条件を固定）：`python -B scripts/quantum/nuclear_effective_potential_core_well.py`
  - 出力：`output/quantum/nuclear_effective_potential_core_well.png`（sha256=`AD67016CB3AF9256D56EFBB39200B527936E9B2AEA76A0D74AB42CDDF4DD9631`）
  - 出力：`output/quantum/nuclear_effective_potential_core_well_metrics.json`（sha256=`6A42C68A65EF81B2468FDE024D04411C88C59658DFB6B72943169CE05B71BFFE`）
- 有効方程式（finite core+well；repulsive core の自由度を入れても最適解が `Rc→0,Vc→0` に退化し、singlet 予言が整合しないことを固定）：`python -B scripts/quantum/nuclear_effective_potential_finite_core_well.py`
  - 出力：`output/quantum/nuclear_effective_potential_finite_core_well.png`（sha256=`6DCFD97882EC70652F99EF36C2D481370DDCD1447811E08C60D2C4A40764AE34`）
  - 出力：`output/quantum/nuclear_effective_potential_finite_core_well_metrics.json`（sha256=`85D74400984464E865E11C37BDFB297F9CF1C9DB607F33FC62AF7234658811CC`）
- 有効方程式（2-range；triplet を `(B,a_t,r_t,v2t)` で拘束し、singlet は `a_s` のみで合わせて `(r_s,v2s)` を予言し、棄却条件を固定）：`python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.9.6`
  - 出力：`output/quantum/nuclear_effective_potential_two_range.png`（sha256=`8B19C1862EA36740CE82C209BC4B67560CD3AB0E88E08DF707AD2DDDB587EF0D`）
  - 出力：`output/quantum/nuclear_effective_potential_two_range_metrics.json`（sha256=`6B748DAFB7D58D36D25CA86862848085FF3823FD4C48FF1060FA6C1E00864C9D`）
- 有効方程式（2-range；singlet を `a_s,r_s` で拘束し、`v2s` を差分予測として棄却条件を固定）：`python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.9.7`
  - 出力：`output/quantum/nuclear_effective_potential_two_range_fit_as_rs.png`（sha256=`699A94C523F4EC5D6B2E331445608297D7E61A48AA7C4EDB11E2B17EA2EE0BCE`）
  - 出力：`output/quantum/nuclear_effective_potential_two_range_fit_as_rs_metrics.json`（sha256=`BD5241AED1EE3F65D0AFC3B947E1AB65C7A752279339B1E32048079DE99EAC5B`）
- 有効方程式（repulsive core + 2-range；符号構造を含む最小拡張で `v2s` の符号を救えるかを検証し、棄却条件を固定）：`python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.9.8`
  - 出力：`output/quantum/nuclear_effective_potential_repulsive_core_two_range.png`（sha256=`B2AFF28692265554E3B389D342A801B3501F85EA27D0747AE46FD40C08E0EC48`）
  - 出力：`output/quantum/nuclear_effective_potential_repulsive_core_two_range_metrics.json`（sha256=`561DB997AC01AB5AB53E6B4AEB0AC3D3F44FF521F6A508ED55A41A6C9D4AE53D`）
- 有効方程式（λ_π 制約下の 2-range；singlet の V2_s を signed で許しても `v2s` の符号が救えないことを固定）：`python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.13.3`
  - 出力：`output/quantum/nuclear_effective_potential_pion_constrained_signed_v2.png`（sha256=`4657206BAED009C636B573D78F133B898E46303AAB79BBE7FDFB0E83BE6B8EBD`）
  - 出力：`output/quantum/nuclear_effective_potential_pion_constrained_signed_v2_metrics.json`（sha256=`564360517A5370F77CE15E3C524AF141FAFD71C6C79AC51BADD205E99C9C0E01`）
  - 出力：`output/quantum/nuclear_effective_potential_pion_constrained_signed_v2.csv`（sha256=`C79F0BCA96FE58E0260B2153670226EE8137689E3B297CC50BA07CEC324DD899`）
- 有効方程式（λ_π 制約下の 3-range；Yukawa tail 粗視化（shared tail）でも `v2s` の符号が救えないことを固定）：`python -B scripts/quantum/nuclear_effective_potential_two_range.py --step 7.13.4`
  - 出力：`output/quantum/nuclear_effective_potential_pion_constrained_three_range_tail.png`（sha256=`0E9B06C47237779DB96D24F759D353C9057FF94EBD98F58A273E0EF4001EF801`）
  - 出力：`output/quantum/nuclear_effective_potential_pion_constrained_three_range_tail_metrics.json`（sha256=`187AC0456F8F6F54D1582CC55F274A2CC602655E3BB3F7AD2DA35B58B9104608`）
  - 出力：`output/quantum/nuclear_effective_potential_pion_constrained_three_range_tail.csv`（sha256=`B72943E1F9E83D3F2FF7D99E74415C881751D25110E0AD2B423B1BE591DCE840`）

### 強い相互作用（ハドロン/QCD；基準値の固定）
- Particle Data Group (PDG) RPP 2024：Monte Carlo 用の masses/widths/particle ID table（一次ソース）
  - URL：`https://pdg.lbl.gov/2024/mcdata/mass_width_2024.txt`
  - ローカル：`data/quantum/sources/pdg_rpp_2024_mass_width/mass_width_2024.txt`（sha256=`9D6024DA6705FCAB556F1136053C802FEF77AD69B28EABDC1CD1D9284B6E2F6C`）
  - マニフェスト：`data/quantum/sources/pdg_rpp_2024_mass_width/manifest.json`（sha256=`4C21C40D0D7670EF1E797FCDC4D277CD1BEBF212D93C730B1DB964A0DBB9B8E7`）
  - 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_pdg_mass_width_2024.py`
- ベースライン（p/n/π/K など）固定出力：`python -B scripts/quantum/qcd_hadron_masses_baseline.py`
  - 出力：`output/quantum/qcd_hadron_masses_baseline.png`（sha256=`528BB8A6DEF7D1E2C4D1D2333EEE4D56D7A918FD884255CEA544C3E43FAC0F5F`）
  - 出力：`output/quantum/qcd_hadron_masses_baseline_metrics.json`（sha256=`5F4EBB913B485934CD250AE17A5E2C56BCB41857D302315B7F9B95FACC2259E3`）
  - 出力：`output/quantum/qcd_hadron_masses_baseline.csv`（sha256=`8A3612CD9525FFB26CDAA2319BE4CB7C7F36F42A135EEC6F319B2113D5000055`）

### 物質波干渉／de Broglie（電子二重スリット／原子反跳）
- 電子二重スリット（実験；装置パラメータが本文に明記）：
  - Bach et al. 2012（arXiv:1210.6243v1）：`data/quantum/sources/arxiv_1210.6243v1.pdf`（sha256=`3FF2F5358B25586E10469081D9B0A6E07671CCED9D3F3A5074E55F5F6C749C74`）
- de Broglie（λ=h/p）を含む精密検証（原子反跳→α）：  
  - Bouchendira et al. 2008（arXiv:0812.3139v1）：`data/quantum/sources/arxiv_0812.3139v1.pdf`（sha256=`F763334508B9D7F06A390BCF32E38E246CD4468FEEBD5D9EB6FB57AC93782B55`）
- 独立参照（α；電子 g-2 → QED）：
  - Gabrielse et al. 2008（arXiv:0801.1134v2）：`data/quantum/sources/arxiv_0801.1134v2.pdf`（sha256=`562D23333D57C1C8D415F357C761508FDC4A5AEF512B28639D3AC0079A7C69F5`）

### 重力誘起デコヒーレンス（time dilation / clocks）
- time dilation による decoherence（主張；内部自由度↔重心の結合）：
  - Pikovski et al. 2013（arXiv:1311.1095v2）：`data/quantum/sources/arxiv_1311.1095v2.pdf`（sha256=`5507FEF706C4C460485D611506DD3843D9F49E83DC7AAA7E22701FB9EA17B0E7`）
- 議論（普遍性・実験文脈・自由落下系の扱い）：
  - Bonder et al. 2015（arXiv:1509.04363v3）：`data/quantum/sources/arxiv_1509.04363v3.pdf`（sha256=`2096AE35E84FC9850DF17CF11AA081AA1749EFB3669ECD22B32D7E72A2FF400C`）
  - Anastopoulos & Hu 2015（arXiv:1507.05828v5）：`data/quantum/sources/arxiv_1507.05828v5.pdf`（sha256=`25FB3DE318478837E4F9CEA0161CD4A965485553F89AC1390EEAFE4127F5E4C1`）
  - Pikovski et al. 2015（reply; arXiv:1509.07767v1）：`data/quantum/sources/arxiv_1509.07767v1.pdf`（sha256=`AE51D0F59C648796D1C3025522D3F58D763A22860CACC118FC11F41887AEB3BF`）
- 光格子時計（原子集団の重力赤方偏移→dephasing/contrast loss の提案）：
  - Hasegawa et al. 2021（arXiv:2107.02405v2）：`data/quantum/sources/arxiv_2107.02405v2.pdf`（sha256=`7E171DC5C3406C05028261C07471FE6687BC4DC5E8B078B9286D60BEA1617149`）
- 参考（別系統：重力による自発崩壊／collapse model；本稿の dephasing と同一視しない）：
  - Diósi 1987（Physics Letters A 120, 377–381；DOI: 10.1016/0375-9601(87)90681-5）`https://doi.org/10.1016/0375-9601(87)90681-5`（paywallの可能性；未キャッシュ）Accessed: 2026-01-26.
  - Penrose 1996（Gen. Relativ. Gravit. 28, 581–600；DOI: 10.1007/BF02105068）`https://doi.org/10.1007/BF02105068`（paywallの可能性；未キャッシュ）Accessed: 2026-01-26.
  - Donadi et al. 2020/2021（Nature Physics 17, 74–78；DOI: 10.1038/s41567-020-1008-4）`https://doi.org/10.1038/s41567-020-1008-4`（paywallの可能性；未キャッシュ）Accessed: 2026-01-26.

### 光の量子干渉（単一光子・HOM・スクイーズド光）
- 単一光子干渉（150 km；Mach–Zehnder；fringe visibility を要約に明記）：
  - Kimura et al. 2004（arXiv:quant-ph/0403104v2）：`data/quantum/sources/arxiv_quant-ph_0403104v2.pdf`（sha256=`C094FFA831C59E0916C77D3C70D26EBB44EFD17BA065BA1F53453AEB9723C072`）
- HOM（遠隔量子ドット間の2光子干渉；visibility と separation の議論が本文に明記）：
  - arXiv:2106.03871v2 “Quantum interference of identical photons from remote GaAs quantum dots”：`data/quantum/sources/arxiv_2106.03871v2.pdf`（sha256=`59729A4F7AB95D623DBB1FAD329087E50A601347024D143E91064CD19C6EB93D`）
  - 公開データ（Zenodo；図ごとの raw；一次プロダクト）：
    - DOI: 10.5281/zenodo.6371310 → `data/quantum/sources/zenodo_6371310/`（manifest: `data/quantum/sources/zenodo_6371310/manifest.json`）
    - 例（低周波ノイズ PSD）：`data/quantum/sources/zenodo_6371310/DataExfig3.zip`（sha256=`010076C53359BB07159BB1B4CF5589C6EAB82C51B2B4B18BE6BF7B5194ED3E73`）
- スクイーズド光（10 dB benchmark）：
  - Vahlbruch et al. 2007（arXiv:0706.1431v1）：`data/quantum/sources/arxiv_0706.1431v1.pdf`（sha256=`A80D9C75BBAAF9685EF5E5939E214D5000886A2C38EB7DE0FDD132972F855037`）

### 真空・QED精密（Casimir 効果 / Lamb shift / H 1S–2S）
- Casimir 力（AFM；sphere-plane；精度の代表値が要約に明記）：
  - Roy, Lin, Mohideen 2000（arXiv:quant-ph/9906062v3）：`data/quantum/sources/arxiv_quant-ph_9906062v3.pdf`（sha256=`AF4DB4DED1BE34D2A0828365B5DC63CEA19738A3736D2C14BAA41A61F99F870B`）
- Lamb shift（定義＋スケーリングと、非QED系統（核サイズ）の寄与例を含む）：
  - Ivanov & Karshenboim 2000（arXiv:physics/0009069v1）：`data/quantum/sources/arxiv_physics_0009069v1.pdf`（sha256=`87196CFA73B9C09B190C147DD028502017E1BAFA2FD83B412C7ADA0C1E132CFD`）
- H 1S–2S（絶対周波数；不確かさ予算が明記）：
  - Parthey et al. 2011（arXiv:1107.3101v1）：`data/quantum/sources/arxiv_1107.3101v1.pdf`（sha256=`1DAFA322B45A9626C65CDFBB134C6B5DC057DF5D5F4DFB06C02759E05F368A65`）
- 取得・キャッシュ（スクリプト）：`python -B scripts/quantum/fetch_qed_vacuum_precision_sources.py`（保存先：`data/quantum/sources/`）

---

## 6) 追加候補（困った時の一次データ取得）

### MAST（NASA Mikulski Archive for Space Telescopes）/ astroquery
- ドキュメント：`https://astroquery.readthedocs.io/en/latest/mast/mast.html`
- API（直接呼び出し）：
  - invoke：`https://mast.stsci.edu/api/v0/invoke`（`?request=<json>` または POST form `request=<json>`）
  - download：`https://mast.stsci.edu/api/v0.1/Download/file?uri=<dataURI>`
- 用途候補：
  - JWST/HST/GALEX/TESS などの公開観測（画像/スペクトル等）の **一次データ**取得
  - 高赤方偏移銀河候補の **スペクトル一次データ**を raw から扱いたい場合の入口（距離指標の二次産物を介さず、輝線のズレ→赤方偏移 z を直接扱える）
    - Mission/obs_collection：JWST
    - dataproduct_type：Spectrum
    - productSubGroupDescription：x1d（1次元スペクトル；ファイル名上は `_x1d` などを含むことが多い）
    - 対象候補（例）：GN-z11、JADES-GS-z14-0、JADES_10058975、RX_11027、LID-568、GLASS-z12、CEERS2-5429
      - 注意（LID-568）：MAST 側の target_name は `LID568` / `LID568MIRI`（program 1760）。座標（J2000）は Chandra の公開ページから確認し、参照を `data/cosmology/sources/` に保存した。
        - `data/cosmology/sources/chandra_photo_album_lid568_20241104.html`
        - `data/cosmology/sources/arxiv_2405.05333.pdf`
      - 例（GLASS-z12）：MIRI/LRS（P750L; program 3703）の公開 x1d を取得し、`data/cosmology/mast/jwst_spectra/glass_z12/` にキャッシュして z_confirmed まで到達。
      - 例（RX_11027）：NIRSpec/SLIT（F170LP;G235M; program 6073）の公開 x1d を取得し、`data/cosmology/mast/jwst_spectra/rx_11027/` にキャッシュして z_confirmed（z≈2.4216）まで到達。
- 導入する場合のルール：
  - `scripts/<topic>/fetch_mast_*.py` を作り、取得結果は `data/<topic>/mast/`（または `data/<topic>/sources/`）へ保存して offline 再現を確保する。
  - 取得したスペクトル（x1d）は、最小でも「取得クエリ条件」「対象名」「取得日時」「ファイル一覧（sha256推奨）」を `data/<topic>/mast/README.md`（またはJSON）に残す。
  - 新規テーマが必要なら、`data/jwst/`・`scripts/jwst/`・`output/jwst/` を追加する（フォルダ規約に従う）。

実装（Phase 4 / Step 4.6）：
- JWST x1d（スペクトル一次データ）のキャッシュ：`scripts/cosmology/fetch_mast_jwst_spectra.py`
  - 保存：`data/cosmology/mast/jwst_spectra/`
  - QC：`output/cosmology/jwst_spectra__<target_slug>__x1d_qc.png`

---

## 7) XRISM（X-ray Imaging and Spectroscopy Mission；X線高分解能スペクトル）

- 配布元（一次データ）：ISAS/JAXA DARTS（XRISM archive; FITS；rev3）【一次の正】
  - Public data list（UI）：`https://darts.isas.jaxa.jp/pub/xrism/browse/public_list/`
  - Resolve metadata（CSV；公開済みobsid一覧）：`https://data.darts.isas.jaxa.jp/pub/xrism/metadata/xrism_resolve_data.csv`
  - rev3 data root（obsid別の directory listing）：`https://data.darts.isas.jaxa.jp/pub/xrism/data/obs/rev3/`
  - 保存（offline再現）：
    - `data/xrism/sources/darts/manifest.json`（metadata/public list の sha256）
    - `data/xrism/raw/<obsid>/`（rev3 の `<obsid>/` 以下を mirror）
    - `data/xrism/sources/<obsid>/manifest.json`（obsid別の sha256）
- 配布元（ミラー）：NASA HEASARC（XRISM archive; FITS）
  - HEASARC（総合）：`https://heasarc.gsfc.nasa.gov/`
  - Browse（検索UI）：`https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3browse.pl`
  - 備考：XRISM（Resolve/Xtend）の公開データ（event/pha/rmf/arf/bkg 等）を、obsid 単位で取得して offline 再現できる形にキャッシュする。
- 用途（Phase 4 / Step 4.13）：
  - BH/AGN：吸収線（UFO/disk wind）の centroid ずれ→速度 v/c を一次データで固定（系統/統計分離）。
  - 銀河団：輝線 centroid/線幅→ z（距離指標非依存）と速度分散 σ_v（turbulence proxy）を固定し、cluster 系統（mass bias）の入口へ接続。
- 関連一次データ（Phase 4 / Step 4.13.7；Fe-Kα 相対論的 broad line/ISCO）：
  - XMM-Newton Science Archive（XSA；ESA）：`https://www.cosmos.esa.int/web/xmm-newton/xsa`
  - XMM-Newton（ミラー；PPS/ODF）：NASA HEASARC FTP：`https://heasarc.gsfc.nasa.gov/FTP/xmm/data/`（rev0/rev1）
    - 観測ログ（target→obsid 抽出）：HEASARC Browse（XMMMASTER）：`https://heasarc.gsfc.nasa.gov/W3Browse/xmm-newton/xmmmaster.html`
  - XMM-Newton EPIC canned RMF（HEASARC CALDB；PPS SRSPEC の RESPFILE が参照）：
    - 配布元：`https://heasarc.gsfc.nasa.gov/FTP/caldb/data/xmm/ccf/extras/responses/`
      - MOS：`https://heasarc.gsfc.nasa.gov/FTP/caldb/data/xmm/ccf/extras/responses/MOS/`
      - PN ：`https://heasarc.gsfc.nasa.gov/FTP/caldb/data/xmm/ccf/extras/responses/PN/`
    - 保存（offline再現）：`data/xrism/xmm_epic_responses/{MOS|PN}/` ＋ `data/xrism/sources/xmm_epic_responses_manifest.json`
  - NuSTAR archive（NASA HEASARC）：`https://heasarc.gsfc.nasa.gov/docs/nustar/nustar_archive.html`
  - NuSTAR（event/auxil）：NASA HEASARC FTP：`https://heasarc.gsfc.nasa.gov/FTP/nustar/data/obs/`
    - 参照（target→obsid 抽出）：NuSTAR SOC As-Flown Timeline（public）：`http://nustarsoc.caltech.edu/NuSTAR_Public/NuSTAROperationSite/AFT_Public.php`
  - 保存（offline再現）：`data/xrism/xmm_nustar/<obsid_or_dataset>/`（products）＋`data/xrism/sources/xmm_nustar_manifest.json`（取得条件・ファイル一覧・sha256）
- 参照（線同定；遷移エネルギー）：
  - NIST Atomic Spectra Database（ASD）：`https://physics.nist.gov/asd`
  - 初版（Step 4.13.2）では、Fe XXV Heα=6.700 keV / Fe XXVI Lyα=6.966 keV を line_id の参照値として使用（`scripts/xrism/xrism_bh_outflow_velocity.py`）。
- 取得スクリプト（実装）：
  - `scripts/xrism/fetch_xrism_heasarc.py`
  - 保存：`data/xrism/heasarc/manifest.json`（取得条件・obsid・ファイル一覧・sha256）
  - `scripts/xrism/fetch_xrism_darts_metadata.py`（DARTS metadata/public list の取得・sha256固定）
  - `scripts/xrism/build_xrism_target_catalog.py`（P-model検証ターゲットseed：`data/xrism/sources/xrism_target_catalog.json`）
  - `scripts/xrism/fetch_xrism_darts_rev3.py`（DARTS rev3 の raw 取得→`data/xrism/raw/<obsid>/`）
  - `scripts/xrism/update_xrism_targets_catalog_from_json.py`（既存catalogへ追記補完）
  - （Step 4.13.7）`scripts/xrism/fetch_xmm_nustar_heasarc.py`（XMM/NuSTAR の取得→cache→sha256 manifest）
    - seed：`data/xrism/sources/xmm_nustar_targets.json`
    - manifest：`data/xrism/sources/xmm_nustar_manifest.json`
  - （Step 4.13.7）`scripts/xrism/fek_relativistic_broadening_isco_constraints.py`（ISCO台帳I/Fの固定；stub）

---

## 8) 銀河回転曲線（SPARC：RAR/BTFR）

- 配布元（一次データ）：SPARC database（Lelli, McGaugh, Schombert 2016）
  - 公開：`https://astroweb.case.edu/SPARC/`（http は環境によって接続拒否になることがあるため、https を正とする）
  - 用途：銀河回転曲線（HI/Hα）とバリオン分布（Spitzer 3.6μm）から、RAR/BTFR を再構築して P-model の銀河スケール予測を検証する（Phase 6 / Step 6.5）。
- 派生（参照）：
  - RAR：McGaugh, Lelli, Schombert 2016（PRL 117, 201101）
  - BTFR：Lelli et al. 2019
- 一次論文（TeX/PDF；offline再現）：
  - SPARC I（Mass models）：arXiv:1606.09251
  - RAR（One Law to Rule Them All）：arXiv:1610.08981
  - 取得スクリプト：`scripts/cosmology/fetch_sparc_primary_sources.py`
  - 保存：`data/cosmology/sources/arxiv_1606.09251.pdf` / `data/cosmology/sources/arxiv_1610.08981.pdf`（+ source tarball）
  - manifest：`data/cosmology/sources/sparc_primary_sources_manifest.json`
- 保存（offline再現）：
  - raw cache：`data/cosmology/sparc/raw/`（SPARC配布の `*.mrt` / `Rotmod_LTG.zip`）
  - sha256 manifest：`data/cosmology/sparc/manifest.json`（URL/bytes/sha256/取得条件）
  - 取得スクリプト：`scripts/cosmology/fetch_sparc.py`
  - 解析（RAR再構築）：`scripts/cosmology/sparc_rar_from_rotmod.py`
  - 解析（BTFR）：`scripts/cosmology/sparc_btfr_metrics.py`
  - 固定出力（例）：
    - RAR：`output/cosmology/sparc_rar_reconstruction.csv` / `output/cosmology/sparc_rar_metrics.json` / `output/cosmology/sparc_rar_scatter.png`
    - BTFR：`output/cosmology/sparc_btfr_metrics.json` / `output/cosmology/sparc_btfr_scatter.png`
