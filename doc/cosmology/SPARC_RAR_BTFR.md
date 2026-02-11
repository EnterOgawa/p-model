# SPARC（RAR/BTFR）— Phase 6 / Step 6.5 の実装ノート（6.5.1 方針含む）

目的：
- SPARC（Lelli, McGaugh, Schombert 2016）一次データから、RAR（Radial Acceleration Relation）と BTFR を **offline再現可能**に再構築し、P-model の銀河スケール検証を「入力→固定→出力→反証条件」の形に落とす。

## 現状（固定済み）

- 一次データ取得（sha256 manifest固定）：
  - raw：`data/cosmology/sparc/raw/*`
  - manifest：`data/cosmology/sparc/manifest.json`
  - 入口：`python -B scripts/cosmology/fetch_sparc.py --download-missing`
- 一次論文（TeX/PDF；sha256 manifest固定）：
  - SPARC I / RAR（arXiv:1606.09251, 1610.08981）：`python -B scripts/cosmology/fetch_sparc_primary_sources.py`
  - manifest：`data/cosmology/sources/sparc_primary_sources_manifest.json`
- RAR再構築（観測側 g_obs/g_bar）：
  - CSV：`output/cosmology/sparc_rar_reconstruction.csv`
  - metrics：`output/cosmology/sparc_rar_metrics.json`
  - 図：`output/cosmology/sparc_rar_scatter.png`
  - 入口：`python -B scripts/cosmology/sparc_rar_from_rotmod.py`

## 6.5.1（理論的準備）— A/B の選択（現時点の結論）

このリポジトリのコア定義（Phase 1）では

- φ ≡ -c² ln(P/P0)
- a = -∇φ = c² ∇ln(P/P0)

であり、弱場（|φ|/c² ≪ 1）では φ はニュートンポテンシャルへ写像される（|φ|/c² の高次は Solar System/量子干渉側で別途扱う）。

したがって、**現時点のコア P-model は銀河スケール（低加速度領域）に MOND 型の新しい加速度スケール a0 を自動では生成しない**。このため Step 6.5 は当面、次の扱いで進める：

- **(A) P-model = GR（弱場）を既定**とする。
  - g_bar だけでは RAR/回転曲線が説明できない、という既知の事実を再構築で再確認し、以後の議論で「手続き差（データ処理）」と「理論差（予測）」が混ざらない形に固定する。
  - 追加の重力源（ダークマター相当）を導入する場合は、その分布（g_DM）を別途モデル化しない限り **予測（predict）**にはならない（fit/predict 分離）。
- **(B) P-model が弱場で修正を持つ**シナリオは、将来の拡張仮説として保留する。
  - もし (B) を主張するなら、`g_obs = f(g_bar, P)` の具体式（自由度の数、凍結手順、反証条件）を 6.5.5 の形式で固定し、RAR/BTFR で 3σ 判定できる形に落とす必要がある。

結論：
- Step 6.5 は「銀河で DM を否定する」ことを先に目的にせず、まず **観測側の再解析 I/F を固定**して、(A)/(B) のどちらに進んでも比較可能な土台を作る。

## 次の作業（Step 6.5 内）

- 6.5.4/6.5.5（BTFR cross-check と falsification pack）は固定済み（出力：`output/cosmology/sparc_btfr_metrics.json` / `output/cosmology/sparc_falsification_pack.json`）。
- 次の追加課題：もし (B) を採用主張するなら、`g_obs=f(g_bar,P)` の具体式（導出・自由度台帳・凍結手順）を閉じ、Υ の外部制約（prior）と手続き系統（sys）を含めて、複数 dataset で同一 gate（`preferred_metric=sweep_summary_galaxy` の pass_rate(|z|<3)）を満たすことを要件化する（現状は adopted_final=False）。

## 参考：RAR baseline fit（比較用）

P-model 予測ではない比較用として、McGaugh+2016 の経験式
`g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))` を `output/cosmology/sparc_rar_reconstruction.csv` に対して当て、
`a0` を 1パラメータ fit した結果を `output/cosmology/sparc_falsification_pack.json` に保存している。

- best-fit：`a0≈1.24×10^-10 m/s^2`（log10(a0)≈-9.905）
- 注意：現段階の重みは V_obs の統計誤差のみで、g_bar 側（M/L 等）の系統は別途扱う必要がある。

## 参考：M/L（Υ）sweep（g_bar 系統）

g_bar 側の主要系統として、恒星 M/L（Υ_disk, Υ_bulge）を sweep し、baryons-only ヌルの結論がどれだけ動くかを固定した。

- 出力：`output/cosmology/sparc_rar_mlr_sweep_metrics.json`
- 結果（今回の sweep 範囲内）：
  - low-accel weighted mean residual：0.57〜0.67 dex
  - low-accel z：1.06e3〜1.34e3（依然として極めて大）
- baseline a0 fit：`a0≈(0.94〜1.72)×10^-10 m/s^2`

## 参考：fit→freeze→holdout（galaxy split）

候補モデル（弱場修正を含む）の自由度を「fit→凍結→別銀河で検証」できる枠組みを固定した。
現段階は比較用として baryons-only と McGaugh+2016（a0 1パラメータ）に加えて、**a0 を外部固定**した候補も入れている（いずれも現時点では P-model 予測ではない）。

- 出力：`output/cosmology/sparc_rar_freeze_test_metrics.json`
- 実装：`scripts/cosmology/sparc_rar_freeze_test.py`

### 候補：a0 = κ c H0^(P)（P_bg 接続案）

MOND 系でしばしば議論される関係 `a0 ~ c H0/(2π)` に倣い、P_bg のメトリクス `H0^(P)` を用いて

- `a0 = κ c H0^(P)`（既定：κ=1/(2π)）

として `a0` を **SPARC から fit しない**候補（`candidate_rar_pbg_a0_fixed_kappa`）を freeze-test に追加した。

- 入力：`output/cosmology/cosmology_redshift_pbg_metrics.json`（H0^(P)）
- 例（seed=20260129, train_frac=0.7, κ=1/(2π)）：
  - `a0≈1.08×10^-10 m/s^2`
  - holdout（with σ_int）低加速度：`z≈-0.94`（residual≈-0.006 dex）

注意：
- これはコア P-model（φ≡-c² ln(P/P0)）から導出された弱場予測ではなく、**P_bg との接続案（候補）**として扱う。

追加（6.5.5.2.1）：
- `output/cosmology/sparc_falsification_pack.json` に外部固定 a0 の統計（σ_intなし）を `baselines.rar_mcgaugh2016_a0_pbg_fixed_kappa` として保存（例：低加速度 z≈33）。
- `output/cosmology/sparc_rar_mlr_sweep_metrics.json` にも候補の envelope を追加（σ_intなし：低加速度 z は約 -33〜121）。
- したがって棄却条件は「σ_int をどう扱うか」を明確にした上で freeze-test（holdout）と整合させて固定する。

追加（6.5.5.2.3）：
- freeze-test は `--seeds` / `--train-fracs` により複数 split を走査でき、`output/cosmology/sparc_rar_freeze_test_metrics.json` に `sweep_summary`（point-level）と `sweep_summary_galaxy`（galaxy-level；銀河内平均を1サンプル）を固定する。
- 同一銀河内の多数点を独立扱いすると有意度（SEM）が過小評価され得るため、採用判定は `output/cosmology/sparc_falsification_pack.json` の `preferred_metric=sweep_summary_galaxy` を正とする。

補足：κ-fit は baseline の再パラメータ化
- `candidate_rar_pbg_fit_kappa` は train で a0 を fit した baseline（McGaugh+2016 の a0 fit）を κ=a0/(c H0^(P)) として表現しただけであり、freeze-test の `sweep_summary` も baseline と一致する（新規予測にはならない）。

追加（6.5.5.2.5）：κ sweep（外部固定 prior の感度）
- 目的：κ を外部固定（prior）する前提で、κ の取り方にどれだけ結果が敏感かを固定する（＝「この型でどこまで改善し得るか」）。
- 実装：`scripts/cosmology/sparc_rar_pbg_kappa_sweep.py`
- 出力：`output/cosmology/sparc_rar_pbg_kappa_sweep_metrics.json` / `output/cosmology/sparc_rar_pbg_kappa_sweep.png`
- 結果（seeds=50, train_fracs=5；計250split）：
  - point-level：best κ≈0.15 でも pass_rate(|z|<3)≈0.728 に留まる（ref κ=1/(2π) は≈0.648）。
  - galaxy-level：κ≈0.11–0.16 で pass_rate(|z|<3)≥0.95 を満たし、ref κ=1/(2π) は pass_rate≈0.976（best は κ≈0.12 で 1.0）。

追加（6.5.5.2.6）：M/L（Υ）系統の freeze-test robustness
- 目的：Υ_disk/Υ_bulge を変えると g_bar そのものが変わるため、freeze-test の採用判定が **M/L 系統に対して頑健か**を固定する。
- 実装：`scripts/cosmology/sparc_rar_freeze_test_mlr_sweep.py`
- 出力：`output/cosmology/sparc_rar_freeze_test_mlr_sweep_metrics.json` / `output/cosmology/sparc_rar_freeze_test_mlr_sweep.png`
- 結果（Υ_disk=[0.45,0.5,0.55], Υ_bulge=[0.65,0.7,0.75]；seeds=50, train_fracs=5；計250split）：
  - 候補（κ=1/(2π)）は **adopted_fixed_ml（Υ=0.5/0.7）では pass_rate≈0.976** だが、M/L sweep の min pass_rate≈0.852（Υ_disk=0.55, Υ_bulge=0.75）となり robust_adopted_mlr=False（`output/cosmology/sparc_falsification_pack.json`）。
  - 感度はほぼ Υ_disk に支配され、Υ_disk=0.55 の列で pass_rate が 0.852–0.88 まで低下する一方、Υ_disk≤0.5 では Υ_bulge=0.65–0.75 全域で min=0.976 を維持する（pack の `scenario_b.systematics.mlr_sweep.candidate.marginal_by_upsilon_disk`）。
  - したがって「弱場銀河での差分予測」を主張するには、Υ の許容域（prior）を一次ソースで凍結するか、Υ を nuisance として train 推定→holdout 持ち越しの手続きを追加して自由度台帳を閉じる必要がある。
  - 追加（6.5.5.2.8）：Υ を nuisance（global）として train で推定→holdoutへ持ち越す（`sparc_rar_freeze_test.py --fit-upsilon-global`）を試したが、galaxy-level pass_rate(|z|<3)≈0.888（median z≈-2.06）となり adopted_fixed_ml より悪化し、単純な nuisance fit は robustness を改善しない（`output/cosmology/sparc_falsification_pack.json` の `scenario_b.systematics.upsilon_fit_on_train`）。
  - 追加（6.5.5.2.9）：手続き系統（low-accel cut / sigma-floor / galaxy集約条件 / outlier clipping）を sweep すると、candidate の pass_rate(|z|<3) が 0.933–1.0（min<0.95）まで動く。したがって手続き差は sys として台帳化し、採用判定（gate）は procedure も含めた robustness で閉じる必要がある（`output/cosmology/sparc_falsification_pack.json` の `scenario_b.systematics.procedure_sweep`）。
  - 一次根拠（Υ★ at 3.6μm）：SPARC I（arXiv:1606.09251；`data/cosmology/sources/arxiv_1606.09251/SPARC.tex`）は、[3.6] の恒星 M/L として Υ★≈0.5 を fiducial とし、低値 Υ★≈0.2（DiskMass）や高値 Υ★≈0.7（bright galaxies の mean maximum-disk limit）も議論する。したがって Υ_disk=0.55 は “fiducial近傍” に含まれ、この近傍で候補が robust を失う点は prior 凍結後も回避できない。
