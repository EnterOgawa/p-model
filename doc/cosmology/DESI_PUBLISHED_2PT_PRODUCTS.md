# DESI DR1：公開2pt（BAO）データの探索メモ（multipoles / cov）

目的：
- Phase 4 / Step 4.5B.21.4.4.2.2 の前提として、DESI DR1（特に BAO galaxies / BAO cosmology 系）で **公開されている一次統計（ξ multipoles / P(k) multipoles）と共分散**がどこにあるかを特定する。

結論（2026-01-19 UTC 時点）：
- DESI data portal（`data.desi.lbl.gov`）の DR1 公開領域で、**BAO の post-recon ξℓ（ℓ=0,2）や P(k) multipoles の「測定ベクトル＋共分散（fitに使う形）」として直接配布されているファイル**は、現状の探索範囲では見つからなかった。
- 一方で、BAO の圧縮出力（距離指標 `D_M/r_d`, `D_H/r_d` の mean/cov）は公開されており、`CobayaSampler/bao_data` にも同一内容（Gaussian BAO likelihood）が公開されている。
  - ただしこれは **LSS一次統計（銀河+random）からの解析パイプラインと fiducial（ΛCDM）を通した派生値**であるため、本リポジトリでは「一次ソース」とは扱わない。
 - 位置づけは **cross-check（参照）**：mean/cov から `F_AP=D_M/D_H`（r_d 相殺）→ `ε_expected` を計算し、LSS catalog-based の ε（一次統計から再構築）と照合する。

- 2026-01-19 追記（再探索）：
  - `data.desi.lbl.gov` が 504 を返しやすいため、探索スクリプトに retry/backoff を追加したうえで、DR1 VAC の `bao-cosmo-params` / `full-shape-cosmo-params`（v1.0）を深めに再クロールしたが `candidates=0`（= “fit用 dv+cov（multipoles/cov; HDF5等）” の公開配布は一次ソースとして確定できない）。
    - `output/cosmology/desi_public_2pt_discovery_dr1_vac_bao_cosmo_params_v1p0_depth6.json`
    - `output/cosmology/desi_public_2pt_discovery_dr1_vac_full_shape_v1p0_depth6.json`
  - 参考として DR2（Y3）側の `bao-cosmo-params`（論文付随の派生パック）も浅く探索したが `candidates=0`（このパックは cosmology params を主目的とし、2pt dv+cov の一次配布先とは見なしにくい）。
    - `output/cosmology/desi_public_2pt_discovery_papers_y3_bao_cosmo_params_depth4_dirs60.json`

探索方法：
- ディレクトリ listing をクロールして、ファイル名に `cov/covar/covariance/2pt/2pcf/xi/pk/multipole/bao` 等を含む候補を抽出する（巨大ファイルはダウンロードしない）。
- 使用スクリプト：`scripts/cosmology/discover_desi_public_2pt_products.py`

実行ログ（主要）：
- DR1 VAC（BAO/Full-shape パラメータ配布領域）：
  - `python -B scripts/cosmology/discover_desi_public_2pt_products.py --roots "https://data.desi.lbl.gov/public/dr1/vac/dr1/bao-cosmo-params/v1.0/" --max-depth 3 --max-dirs 120 --timeout-sec 15 --out-json output/cosmology/desi_public_2pt_discovery_bao_params.json`
  - `python -B scripts/cosmology/discover_desi_public_2pt_products.py --roots "https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-cosmo-params/v1.0/" --max-depth 3 --max-dirs 120 --timeout-sec 15 --out-json output/cosmology/desi_public_2pt_discovery_full_shape_params.json`
  - 結果：`candidates=0`

追加探索（2026-01-18 UTC）：
- スクリプトを拡張（候補キーワードに `likelihood / data_vector / covmat / poles` 等を追加、`.yaml/.yml` も許可、巨大になりやすい `mocks/` を探索対象から除外）し、より広いルートを浅く探索：
  - `python -B scripts/cosmology/discover_desi_public_2pt_products.py --roots "https://data.desi.lbl.gov/public/" --max-depth 4 --max-dirs 150 --timeout-sec 10 --out-json output/cosmology/desi_public_2pt_discovery_public_root_shallow.json`
  - 結果：`candidates=0`（少なくとも浅い範囲では、2ptデータベクトル/共分散を直接配布している場所は未特定）
- 公式 likelihood 実装（公開）：
  - GitHub: `cosmodesi/desi-kp-cosmological-likelihoods`（DESI KP の公式 cosmological likelihoods）
  - ただし `dr1/cobaya/README.md` は **`data/likelihood` ディレクトリ（HDF5）を別途用意する前提**で、当該データの配布場所は本探索範囲では未確定。
- Zenodo（DESI 2024 III: BAO galaxies & quasars）：
  - Record: `zenodo:15086114`（`desi_kp4_zenodo_v1.0.tar.gz`）
  - 内容は「metadata/ノートブック中心」で、2ptの multipoles/cov（fit用）そのものは含まれていないことを確認。

補足：
- 本探索は「公開場所が存在しない」ことを厳密に証明するものではない（探索範囲・深さの制約がある）。
- ただし Phase 4 / Step 4.5 の主方針は「**銀河カタログ＋random から ξℓ を再計算**」であり、公開 multipoles が見つからない場合でも、P-model 側で一次統計を再構築できる。

## 追加探索（2026-01-18 UTC）：DR1公開ポータル／ドキュメント／ミラー

目的：
- 「公開 multipoles/cov（fitに使う測定ベクトル＋共分散）」が *もし公開されているなら* どこにあるかを、DESI DR1 の **公式導線**からも再確認する。

確認したこと（要点）：
- DR1 の公式ドキュメント（`https://data.desi.lbl.gov/doc/releases/dr1/`）は、公開データの入口として
  - `https://data.desi.lbl.gov/public/dr1`
  - `https://webdav-hdfs.pic.es/data/public/DESI/DR1`
  を案内しているが、少なくとも当該ページ内に `likelihood` / `cobaya` / `HDF5` 等の明示的記載は見当たらない（2026-01-18確認）。
- `https://data.desi.lbl.gov/public/papers/y1/` は **PDFのみ**（Y1 key papers）で、2ptの測定ベクトル/共分散（fit用）の配布は確認できない。
- `https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/` は主に **銀河/ランダムのカタログ（FITS）**であり、`cov/xi/pk/multipole/likelihood` 等のファイル名を持つ「2ptの測定ベクトル＋共分散（fit用）」は見当たらない。
- `webdav-hdfs.pic.es` 側は WebDAV（PROPFIND）で一覧でき、トップ階層は `spectro/ survey/ target/ vac/` と一致する（追加の “likelihood配布” ディレクトリは少なくともトップレベルでは見当たらない）。
- 追加：探索スクリプト側も WebDAV（PROPFIND depth=1）に対応させ、`survey/catalogs/dr1/` を depth=6 / max_dirs=260 でクロールしたが `candidates=0`（=「名称で判別できる 2pt dv+cov」の公開配布は未特定）。
  - 実行例：`python -B scripts/cosmology/discover_desi_public_2pt_products.py --roots "https://webdav-hdfs.pic.es/data/public/DESI/DR1/survey/catalogs/dr1/" --max-depth 6 --max-dirs 260 --timeout-sec 25 --max-links-per-dir 180 --out-json output/cosmology/desi_public_2pt_discovery_webdav_survey_catalogs_dr1_depth6.json`
  - 出力：`output/cosmology/desi_public_2pt_discovery_webdav_survey_catalogs_dr1_depth6.json`
  - 参考：VAC 側では Lya correlations の共分散（例：`full-covariance-smoothed.fits`）は候補として検出できる（= WebDAVクロールが機能している確認）。

補足（VAC 側の “再現性パック” の確認）：
- DR1 VAC の `bao-cosmo-params` / `full-shape-cosmo-params` は、いずれも **パラメータ（チェーン）＋設定**の配布が主であり、2pt の測定ベクトル＋共分散（fit用 dv+cov）を同梱している形ではない（少なくともディレクトリlisting上は未確認）。
- `bao-cosmo-params` の Cobaya 設定（例：`.../cobaya/base/desi-bao-all/chain.updated.yaml`）では、BAO likelihood の `measurements_file` / `cov_file` が `bao_data/..._mean.txt` / `bao_data/..._cov.txt` を参照している（= `CobayaSampler/bao_data` 由来の mean/cov に一致し、Phase 4 / Step 4.5B.21.4.4.1.2 と整合）。
- `full-shape-cosmo-params` の Cobaya 設定（例：`.../cobaya/base/desi-v1.5-bao-lrg-z2_.../chain.input.yaml`）では、likelihood が `desi_y1_cosmo_bindings...` を参照し `python_path` が **内部環境（/global/...）** を指す（= 公開ポータル上で dv+cov（HDF5等）を同梱配布している導線は、この範囲では見当たらない）。

暫定結論（扱い）：
- DR1 公開導線（public / papers / survey catalogs / WebDAV mirror）の探索範囲では、`cosmodesi/desi-kp-cosmological-likelihoods` が要求する `data/likelihood`（HDF5; 測定ベクトル＋共分散）を **公開配布として確定できない**。
- したがって当面は、論文・スコアボード上の DESI の ε（catalog-based peakfit）は `cov=diag` を **screening** として扱い、`Y1data` の mean/cov から導いた `ε_expected` との cross-check（`z_score_combined`）を主軸にする。

## 追加調査（2026-01-18 UTC）：DESI 2024 III（BAO galaxies & quasars）の arXiv source / DR1公式ドキュメント

目的：
- BAO galaxies（LRG等）の「公開 multipoles/cov（測定ベクトル＋共分散; fit用 dv+cov）」が *DR1で公開されているなら* その導線を一次ソースとして特定する。

確認したこと（要点）：
- DESI 2024 III（BAO galaxies & quasars；arXiv:2404.03000）の arXiv source を取得して確認したが、Data Availability は「DR1と同時に公開される」という一般記述のみで、ファイル名や配布場所（URL）は明示されていない。
  - source（保存）：`data/cosmology/sources/arxiv_2404.03000_src.tar` / `data/cosmology/sources/arxiv_2404.03000_src/`
  - 該当箇所：`data/cosmology/sources/arxiv_2404.03000_src/main.tex` の “Data Availability”
- 同 source の `catalog.tex` には、内部wiki（trac）や `https://data.desi.lbl.gov/desi/survey/catalogs/Y1/...` のような **/public ではない**導線（covariances の場所を示唆するURL）がコメントとして残っているが、「outdated」とされており、DR1の公開導線とは一致しない（＝公開配布の一次ソースとして採用できない）。
- DESI DR1公式ドキュメント（`https://data.desi.lbl.gov/doc/releases/dr1/`）では、LSS catalog の公開場所（LSScats v1.2/v1.5）と “clustering” catalog の推奨は明示されるが、BAO galaxies の **測定ベクトル（ξℓ/Pℓ）＋共分散（fit用 dv+cov）** を直接配布する旨の記載は見当たらない（少なくとも当該ページの案内範囲では未確認）。

結論（現状の扱い）：
- “公開 multipoles/cov（fit用 dv+cov）” は、DR1の公開導線から一次ソースとして確定できないため、ロードマップ上は **保留**を維持する。
- ただし本リポジトリの BAO 方針（銀河＋random→ξ→ξℓ を自前再構成）自体は成立しており、公開dv+covが無い場合でも、jackknife 等の dv+cov 代替で “diag依存を解消” できる（`doc/ROADMAP.md` の 4.5B.21.4.4.4 参照）。
