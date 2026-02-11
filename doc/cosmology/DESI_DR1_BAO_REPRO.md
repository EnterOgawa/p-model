# DESI DR1：raw→ξ0/ξ2+cov→BAO ε fit（再現手順）

最終更新（UTC）：2026-01-22

目的：
- Phase 4 / Step 4.5（DESI DR1）で行った「raw から multipoles + covariance（dv=[ξ0,ξ2]+cov）の一次プロダクトを自前生成し、BAO fit（ε）まで到達」を再現する。
- ユーザー提示フロー（raw→ξ(s,μ)→ξ0/ξ2→cov→BAO fit）に対応：`doc/ROADMAP.md` の 4.5B.21.4.4.6.*。

前提（重要）：
- raw は `data/cosmology/desi_dr1_lss/raw/` に配置済み（`*_clustering.{dat,ran}.fits`）。
- Corrfunc/pycorr/RascalC を使う計算（paircounts / covariance）は **WSL（Linux）** で実行する（`.venv_wsl`、`--threads 24`）。
- peakfit（ε）と cross-check は Windows（PowerShell）側の Python で実行できる。

---

## 1) raw 内の対象セットを棚卸し（Step 4.5B.21.4.4.6.1）

```powershell
python -B scripts/cosmology/desi_dr1_lss_raw_inventory.py --raw-dir data/cosmology/desi_dr1_lss/raw
```

出力：
- `output/cosmology/desi_dr1_lss_raw_inventory.csv`
- `output/cosmology/desi_dr1_lss_raw_inventory_summary.json`

---

## 2) FITS→抽出NPZ（reservoir）を作る（必要なら）

`cosmology_bao_xi_from_catalogs.py` は `data_dir/manifest.json` と `data_dir/extracted/*.npz` を入力とする。
未生成なら `fetch_desi_dr1_lss.py` で raw FITS から抽出する（例：random_index=0）。

```powershell
python -B scripts/cosmology/fetch_desi_dr1_lss.py --data-dir data/cosmology/desi_dr1_lss_reservoir_r0 --raw-dir data/cosmology/desi_dr1_lss/raw --sample lrg --caps combined --random-index 0 --random-sampling reservoir --random-max-rows 2000000 --sampling-seed 0
python -B scripts/cosmology/fetch_desi_dr1_lss.py --data-dir data/cosmology/desi_dr1_lss_reservoir_r0 --raw-dir data/cosmology/desi_dr1_lss/raw --sample elg_lopnotqso --caps combined --random-index 0 --random-sampling reservoir --random-max-rows 2000000 --sampling-seed 0
```

備考：
- `--data-dir` を `..._reservoir_r{idx}` に変えて `idx=0..17` を作ると、後述の multi-random 結合が可能。

---

## 3)（任意）multi-random 結合（reservoir）と tracer 合算（例：LRG3+ELG1 用）

### 3.1 random_index=0..17 を結合（例：LRG / ELG）

```powershell
python -B scripts/cosmology/desi_dr1_lss_build_combined_random_reservoir.py --out-data-dir data/cosmology/desi_dr1_lss_reservoir_r0to17_mix --sample lrg --random-indices 0-17 --target-rows 2000000 --take-mode random
python -B scripts/cosmology/desi_dr1_lss_build_combined_random_reservoir.py --out-data-dir data/cosmology/desi_dr1_lss_elg_lopnotqso_reservoir_r0to17_mix --sample elg_lopnotqso --random-indices 0-17 --target-rows 2000000 --take-mode random
```

### 3.2 LRG+ELG を合算した data_dir を生成（random の正規化を揃える）

```powershell
python -B scripts/cosmology/desi_dr1_lss_build_combined_tracer_sample.py --in-data-dir data/cosmology/desi_dr1_lss_reservoir_r0to17_mix --in-data-dir-map lrg=data/cosmology/desi_dr1_lss_reservoir_r0to17_mix,elg_lopnotqso=data/cosmology/desi_dr1_lss_elg_lopnotqso_reservoir_r0to17_mix --samples lrg,elg_lopnotqso --out-sample lrg_elg_lopnotqso --out-data-dir data/cosmology/desi_dr1_lss_lrg_elg_reservoir_r0to17_mix_rescale --random-rescale match_combined_total --target-random-rows 2000000
```

---

## 4) ξ(s,μ)→ξ0/ξ2 を生成（pycorr; WSL）（Step 4.5B.21.4.4.6.2）

例：LRG3+ELG1（0.8<z<1.1）を pycorr backend で計算（lcdm/pbg をそれぞれ実行）。

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && OMP_NUM_THREADS=24 .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_from_catalogs.py --data-dir data/cosmology/desi_dr1_lss_lrg_elg_reservoir_r0to17_mix_rescale --sample lrg_elg_lopnotqso --caps combined --dist lcdm --weight-scheme desi_default --random-kind random --match-sectors off --threads 24 --paircounts-backend pycorr --z-min 0.8 --z-max 1.1 --out-tag w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr"
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && OMP_NUM_THREADS=24 .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_from_catalogs.py --data-dir data/cosmology/desi_dr1_lss_lrg_elg_reservoir_r0to17_mix_rescale --sample lrg_elg_lopnotqso --caps combined --dist pbg --weight-scheme desi_default --random-kind random --match-sectors off --threads 24 --paircounts-backend pycorr --z-min 0.8 --z-max 1.1 --out-tag w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr"
```

出力（例）：
- `output/cosmology/cosmology_bao_xi_from_catalogs_lrg_elg_lopnotqso_combined_lcdm_zmin0p8_zmax1p1__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr.npz`
- `output/cosmology/cosmology_bao_xi_from_catalogs_lrg_elg_lopnotqso_combined_*_metrics.json`

---

## 5) covariance を作る（Jackknife / RascalC; WSL）（Step 4.5B.21.4.4.6.3）

### 5.1 sky-jackknife（例：n=8）

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && OMP_NUM_THREADS=24 .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_jackknife_cov_from_catalogs.py --xi-metrics-json output/cosmology/cosmology_bao_xi_from_catalogs_lrg_elg_lopnotqso_combined_lcdm_zmin0p8_zmax1p1__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr_metrics.json --jk-n 8 --jk-mode ra_quantile_per_cap_unwrapped --output-suffix jk_cov_ra_quantile_per_cap_unwrapped_n8 --threads 24"
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && OMP_NUM_THREADS=24 .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_jackknife_cov_from_catalogs.py --xi-metrics-json output/cosmology/cosmology_bao_xi_from_catalogs_lrg_elg_lopnotqso_combined_pbg_zmin0p8_zmax1p1__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr_metrics.json --jk-n 8 --jk-mode ra_quantile_per_cap_unwrapped --output-suffix jk_cov_ra_quantile_per_cap_unwrapped_n8 --threads 24"
```

### 5.2 RascalC（例：n_loops=240）

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && OMP_NUM_THREADS=24 .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_rascalc_cov_from_catalogs.py --xi-metrics-json output/cosmology/cosmology_bao_xi_from_catalogs_lrg_elg_lopnotqso_combined_lcdm_zmin0p8_zmax1p1__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr_metrics.json --output-suffix rascalc_cov_nloops240_lps24 --threads 24 --n-loops 240 --loops-per-sample 24"
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && OMP_NUM_THREADS=24 .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_rascalc_cov_from_catalogs.py --xi-metrics-json output/cosmology/cosmology_bao_xi_from_catalogs_lrg_elg_lopnotqso_combined_pbg_zmin0p8_zmax1p1__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr_metrics.json --output-suffix rascalc_cov_nloops240_lps24 --threads 24 --n-loops 240 --loops-per-sample 24"
```

---

## 6) BAO fit（ε）と Y1data cross-check（Windows）（Step 4.5B.21.4.4.6.4）

### 6.1 peakfit（ε）

```powershell
python -B scripts/cosmology/cosmology_bao_catalog_peakfit.py --sample lrg_elg_lopnotqso --caps combined --out-tag w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr --cov-source jackknife --cov-suffix jk_cov_ra_quantile_per_cap_unwrapped_n8 --output-suffix jk_cov_ra_quantile_per_cap_unwrapped_n8
python -B scripts/cosmology/cosmology_bao_catalog_peakfit.py --sample lrg_elg_lopnotqso --caps combined --out-tag w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr --cov-source rascalc --cov-suffix rascalc_cov_nloops240_lps24 --output-suffix rascalc_cov_nloops240_lps24
```

### 6.2 cross-check（DESI Y1 BAO の ε_expected と比較）

```powershell
python -B scripts/cosmology/cosmology_desi_dr1_bao_y1data_eps_crosscheck.py --peakfit-metrics-json output/cosmology/cosmology_bao_catalog_peakfit_lrg_elg_lopnotqso_combined__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr__jk_cov_ra_quantile_per_cap_unwrapped_n8_metrics.json --out-tag w_desi_default_ms_off_y1bins_lrg3elg1_pycorr_jk_n8 --tracers \"LRG3+ELG1\"
python -B scripts/cosmology/cosmology_desi_dr1_bao_y1data_eps_crosscheck.py --peakfit-metrics-json output/cosmology/cosmology_bao_catalog_peakfit_lrg_elg_lopnotqso_combined__w_desi_default_ms_off_y1bins_lrg3elg1_reservoir_r0to17_mix_rescale_pycorr__rascalc_cov_nloops240_lps24_metrics.json --out-tag w_desi_default_ms_off_y1bins_lrg3elg1_pycorr_rascalc_n240 --tracers \"LRG3+ELG1\"
```

---

## 7) 生成される「一次プロダクト」（fit可能な dv+cov）

dv=[ξ0(s), ξ2(s)] と cov を含むNPZ（例）：
- jackknife cov：`output/cosmology/cosmology_bao_xi_from_catalogs_*__jk_cov_*.npz`
- RascalC cov：`output/cosmology/cosmology_bao_xi_from_catalogs_*__rascalc_cov_*.npz`

fit（ε）の数値（metrics）：
- `output/cosmology/cosmology_bao_catalog_peakfit_*_metrics.json`

