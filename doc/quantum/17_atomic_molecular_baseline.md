# Step 7.12：原子・分子（化学結合）ベースライン（一次データ固定）

目的：原子・分子スケールでも、宇宙論と同じ型（一次データ→固定出力→不確かさ→系統→反証条件）で検証を進めるために、まず **一次データのキャッシュ**と **基準値（ターゲット）**を固定する。

本メモは「原子・分子を第一原理で導いた」と主張するものではなく、Part III の Table 1 に接続できる最小の土台（データ出典＋固定出力）を整備する。

---

## 1) 一次データ（NIST ASD；H I）

### 取得（online）
- `python -B scripts/quantum/fetch_nist_asd_lines.py --spectra "H I"`

保存先（offline 再現の正）：
- `data/quantum/sources/nist_asd_h_i_lines/manifest.json`
- `data/quantum/sources/nist_asd_h_i_lines/extracted_values.json`
- `data/quantum/sources/nist_asd_h_i_lines/nist_asd_lines__h_i__format3__obs_ritz_unc.tsv`

注意：
- NIST ASD の出力では `obs_nu(A)`（負の値）が現れるため、本リポジトリでは
  - λ_vac[Å] = −1 / obs_nu(A)
 で vacuum wavelength を復元して扱う（`fetch_nist_asd_lines.py` が変換して保存）。

---

## 1b) 一次データ（NIST ASD；He I）

### 取得（online）
- `python -B scripts/quantum/fetch_nist_asd_lines.py --spectra "He I"`

保存先（offline 再現の正）：
- `data/quantum/sources/nist_asd_he_i_lines/manifest.json`
- `data/quantum/sources/nist_asd_he_i_lines/extracted_values.json`
- `data/quantum/sources/nist_asd_he_i_lines/nist_asd_lines__he_i__format3__obs_ritz_unc.tsv`

---

## 1c) 一次データ（NIST Chemistry WebBook；H2）

### 取得（online）
- `python -B scripts/quantum/fetch_nist_webbook_diatomic_constants.py --id C1333740 --slug h2`

保存先（offline 再現の正）：
- `data/quantum/sources/nist_webbook_diatomic_h2/manifest.json`
- `data/quantum/sources/nist_webbook_diatomic_h2/extracted_values.json`
- `data/quantum/sources/nist_webbook_diatomic_h2/nist_webbook_diatomic_constants__h2__c1333740.html`

---

## 1d) 一次データ（NIST Chemistry WebBook；HD）

### 取得（online）
- `python -B scripts/quantum/fetch_nist_webbook_diatomic_constants.py --id C13983205 --slug hd`

保存先（offline 再現の正）：
- `data/quantum/sources/nist_webbook_diatomic_hd/manifest.json`
- `data/quantum/sources/nist_webbook_diatomic_hd/extracted_values.json`
- `data/quantum/sources/nist_webbook_diatomic_hd/nist_webbook_diatomic_constants__hd__c13983205.html`

---

## 1e) 一次データ（NIST Chemistry WebBook；D2）

### 取得（online）
- `python -B scripts/quantum/fetch_nist_webbook_diatomic_constants.py --id C7782390 --slug d2`

保存先（offline 再現の正）：
- `data/quantum/sources/nist_webbook_diatomic_d2/manifest.json`
- `data/quantum/sources/nist_webbook_diatomic_d2/extracted_values.json`
- `data/quantum/sources/nist_webbook_diatomic_d2/nist_webbook_diatomic_constants__d2__c7782390.html`

---

## 1f) 一次データ（NIST Chemistry WebBook；熱化学 ΔfH°gas）

分子の結合エネルギー（解離エネルギー）を、分子定数とは独立な一次ソースで固定するため、WebBook の Gas phase thermochemistry をキャッシュする。

### 取得（online）
- `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C12385136 --slug h_atom`（Hydrogen atom）
- `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C16873179 --slug d_atom`（Deuterium atom）
- `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C1333740 --slug h2`
- `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C13983205 --slug hd`
- `python -B scripts/quantum/fetch_nist_webbook_thermochemistry.py --id C7782390 --slug d2`

保存先（offline 再現の正）：
- `data/quantum/sources/nist_webbook_thermo_<slug>/manifest.json`
- `data/quantum/sources/nist_webbook_thermo_<slug>/extracted_values.json`

---

## 1g) 一次データ（NIST；同位体質量：relative atomic mass）

縮約質量 μ（同位体依存）の精密化のため、NIST Atomic Weights and Isotopic Compositions の relative atomic mass を一次データとしてキャッシュする。

### 取得（online）
- `python -B scripts/quantum/fetch_nist_isotopic_compositions.py --element H`

保存先（offline 再現の正）：
- `data/quantum/sources/nist_isotopic_compositions_h/manifest.json`
- `data/quantum/sources/nist_isotopic_compositions_h/extracted_values.json`
- `data/quantum/sources/nist_isotopic_compositions_h/nist_isotopic_compositions__h.html`

---

## 1h) 一次データ（分光：解離エネルギー D0（0 K））

分子の結合エネルギーについて、熱化学（298 K）とは独立な一次ソースとして、分光学的解離エネルギー D0（0 K）を固定する。

### 取得（online）
- `python -B scripts/quantum/fetch_molecular_dissociation_d0_spectroscopic.py`

保存先（offline 再現の正）：
- `data/quantum/sources/molecular_dissociation_d0_spectroscopic/manifest.json`
- `data/quantum/sources/molecular_dissociation_d0_spectroscopic/extracted_values.json`
- `data/quantum/sources/molecular_dissociation_d0_spectroscopic/arxiv_pdf_1902.09471.pdf`
- `data/quantum/sources/molecular_dissociation_d0_spectroscopic/arxiv_pdf_2202.00532.pdf`
- `data/quantum/sources/molecular_dissociation_d0_spectroscopic/vu_publication_hd_d0.html`

---

## 1i) 一次データ（分子遷移：ExoMol line list）

分子の遷移（線リスト）について、恣意性の少ない一次ソースとして ExoMol の line list（states/trans/pf）をキャッシュする。

### 取得（online）
- `python -B scripts/quantum/fetch_exomol_diatomic_line_lists.py`

保存先（offline 再現の正）：
- H2（1H2; RACPPK）
  - `data/quantum/sources/exomol_h2_1h2_racppk/manifest.json`
  - `data/quantum/sources/exomol_h2_1h2_racppk/extracted_values.json`
  - `data/quantum/sources/exomol_h2_1h2_racppk/1H2__RACPPK.states.bz2`
  - `data/quantum/sources/exomol_h2_1h2_racppk/1H2__RACPPK.trans.bz2`
  - `data/quantum/sources/exomol_h2_1h2_racppk/1H2__RACPPK.pf`
- HD（1H-2H; ADJSAAM）
  - `data/quantum/sources/exomol_h2_1h_2h_adjsaam/manifest.json`
  - `data/quantum/sources/exomol_h2_1h_2h_adjsaam/extracted_values.json`
  - `data/quantum/sources/exomol_h2_1h_2h_adjsaam/1H-2H__ADJSAAM.states.bz2`
  - `data/quantum/sources/exomol_h2_1h_2h_adjsaam/1H-2H__ADJSAAM.trans.bz2`
  - `data/quantum/sources/exomol_h2_1h_2h_adjsaam/1H-2H__ADJSAAM.pf`

---

## 1j) 一次データ（分子遷移：MOLAT D2 FUV emission line lists）

D2 の遷移（一次線リスト）については、ExoMol の line list を用意できないため、MOLAT（OBSPM）の D2 FUV emission line lists（transition energies + A）を一次ソースとしてキャッシュする。

### 取得（online）
- `python -B scripts/quantum/fetch_molat_d2_fuv_emission_lines.py`

保存先（offline 再現の正）：
- `data/quantum/sources/molat_d2_fuv_emission_arlsj1999/manifest.json`
- `data/quantum/sources/molat_d2_fuv_emission_arlsj1999/extracted_values.json`

---

## 2) 固定出力（基準値）

### 実行
- `python -B scripts/quantum/atomic_hydrogen_baseline.py`
- `python -B scripts/quantum/atomic_helium_baseline.py`
- `python -B scripts/quantum/molecular_h2_baseline.py --slug h2`
- `python -B scripts/quantum/molecular_h2_baseline.py --slug hd`
- `python -B scripts/quantum/molecular_h2_baseline.py --slug d2`
- `python -B scripts/quantum/molecular_isotopic_scaling.py`
- `python -B scripts/quantum/molecular_dissociation_thermochemistry.py`
- `python -B scripts/quantum/molecular_dissociation_d0_spectroscopic.py`
- `python -B scripts/quantum/molecular_transitions_exomol_baseline.py`

出力（固定名）：
- `output/quantum/atomic_hydrogen_baseline.png`
- `output/quantum/atomic_hydrogen_baseline_metrics.json`
- `output/quantum/atomic_helium_baseline.png`
- `output/quantum/atomic_helium_baseline_metrics.json`
- `output/quantum/molecular_h2_baseline.png`
- `output/quantum/molecular_h2_baseline_metrics.json`
- `output/quantum/molecular_hd_baseline.png`
- `output/quantum/molecular_hd_baseline_metrics.json`
- `output/quantum/molecular_d2_baseline.png`
- `output/quantum/molecular_d2_baseline_metrics.json`
- `output/quantum/molecular_isotopic_scaling.png`
- `output/quantum/molecular_isotopic_scaling_metrics.json`
- `output/quantum/molecular_dissociation_thermochemistry.png`
- `output/quantum/molecular_dissociation_thermochemistry_metrics.json`
- `output/quantum/molecular_dissociation_d0_spectroscopic.png`
- `output/quantum/molecular_dissociation_d0_spectroscopic_metrics.json`
- `output/quantum/molecular_transitions_exomol_baseline.png`
- `output/quantum/molecular_transitions_exomol_baseline_metrics.json`
- `output/quantum/molecular_transitions_exomol_baseline_selected.csv`

内容：
- H I の代表遷移（Lyα, Hα, Hβ, Hγ）について、vacuum wavelength（nm）と周波数（THz）、光子エネルギー（eV）を固定する。
  - 追加：fine structure の観測ターゲットとして、raw TSV（NIST ASD）から Hα/Hβ の multiplet（obs, E1）の範囲（min–max）と本数 N を抽出して `output/quantum/atomic_hydrogen_baseline_metrics.json` に固定する。
- H2（ground state）の分子定数（ωe, ωe x_e, B_e, α_e, D_e, r_e）を一次データから固定する（NIST WebBook）。
- 同位体（HD, D2）についても同じ分子定数を固定し、縮約質量スケーリング（ωe∝μ^{-1/2}, B_e∝μ^{-1}）のチェックを固定出力する。
- 熱化学（ΔfH°gas）から、解離エンタルピー（298 K）を独立ベースラインとして固定する（D0（0 K の分光学的解離エネルギー）とは区別する）。
- 分光（D0；0 K）を独立ベースラインとして固定する（298 K 熱化学とも、分子定数（ωe, ωe x_e）からの Morse 推定 D0 とも区別する）。
- 分子遷移（代表遷移）を、Einstein A の大きい順に top10 を抽出して固定する（選別規則を固定して恣意性を避ける）。
  - H2/HD：ExoMol line list（H2: RACPPK / HD: ADJSAAM）
  - D2：MOLAT D2 FUV emission line lists（ARLSJ1999）
- これは **将来の再導出（P-model からの束縛・準位・遷移の導出）で一致させるべき「観測ターゲット」**であり、現時点で一致を主張するものではない。

---

## 3) 主張範囲（安全宣言）

- 本Stepは、原子スペクトルを P-model の場の理論として導出したとは主張しない。
- 代わりに「一次データの固定」と「基準値（ターゲット）の固定」を行い、次段階での反証可能性を確保する。
- β（光伝播）と δ（速度項）は Part I で凍結されており、原子・分子の説明のために勝手に変更しない（変更が必要なら Part I を改訂し、生成物を一括再生成する）。

---

## 4) 次の拡張（Step 7.12 の本丸）

原子・分子を「既存QMへφを代入しただけ」と見なされないために、最低限以下をロードマップに従って追加する：

- He / 多電子系：一次データ（遷移、イオン化エネルギー、fine/hyperfine など）を固定し、自由度の増加がどこから来るかを明文化する。
- 分子：H2 を起点に、別分子（同位体・多原子）や結合エネルギー等を同じ I/F（manifest→fixed outputs）へ拡張する。
- 反証条件：原子時計・分光（1S–2S, Lamb shift 等）との整合を、安全確認（QED精密との矛盾が出ないこと）とセットで定義する（H 1S–2S は Step 7.8 の固定出力へ追加済み。H I hyperfine（21 cm）は NIST AtSpec で一次固定済み。次は追加QED精密の一次固定）。
