# 30. 核結合エネルギー（Δω→B.E.写像）モデル：反証条件（falsification pack）の固定（Step 7.13.17.13）

目的：
- Step 7.13.17.7–7.13.17.9 で凍結した Δω→B.E. 写像 I/F に対して、
  「どの条件が満たされないと棄却か」を **3σ運用**の形で固定する。
- ここで固定するのは “運用閾値（operational thresholds）” であり、
  物理の普遍閾値ではない（Bell の falsification pack と同じ扱い）。

入力（固定出力）：
- Step 7.13.17.9 の差分CSV（global-R と local-d を同一I/Fで比較）  
  `output/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.csv`
- Step 7.13.17.11 の比較メトリクス  
  `output/quantum/nuclear_binding_energy_frequency_mapping_theory_diff_metrics.json`
- Step 7.13.17.12 の差分定量メトリクス  
  `output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification_metrics.json`
- Step 7.13.15.9 の分離エネルギー（Sn/S2n, shell-gap）診断  
  `output/quantum/nuclear_a_dependence_hf_three_body_separation_energies_metrics.json`
- Step 7.13.15.49 の電荷半径kink strict（A_min=100）  
  `output/quantum/nuclear_a_dependence_hf_three_body_radii_kink_delta2r_radius_magic_offset_even_even_center_magic_only_pairing_deformation_minimal_metrics.json`

---

## 30.1 反証条件パック（固定JSON）

固定出力：
- `output/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_falsification_table.csv`（代表核種）

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.py`

パックが提供するもの：
- 2つの候補I/F（global: x=R, local: x=d）について
  - `log10(B_pred/B_obs)` の分布統計（median, p16, p84）
  - `σ_proxy = (p84−p16)/2`（ロバストな内部散らばり指標）
  - `z_median = median/σ_proxy`（中央値の整合度）
  - A-bin 固定による `z_Δmedian`（A-trend の整合度）
  を計算し、3σ閾値で pass/fail を返す。
- 差分チャネル（P-SEMF / P-Yukawa）について
  - `median abs(ΔB)`（MeV）
  - `median σ_req,rel = median(abs(ΔB)/(3B_obs))`
  を同一JSONに統合し、3σ決着に必要な相対精度ゲートを固定する。

重要な注意：
- `σ_proxy` は **実験誤差ではない**（B_obs の統計誤差は AME2020 では極小）。
  ここでの `σ_proxy` は「このI/Fの手続き誤差・モデル不足」を含む内部散らばりの proxy であり、
  “fit逃げ” を防ぐための運用尺度として使う。

---

## 30.2 反証条件（固定）

本リポジトリでの運用閾値（JSONに固定）：
- **中央値整合**：`abs(z_median) ≤ 3`
  - 解釈：B_pred/B_obs の分布中央値が 1（log10=0）から 3σ_proxy を超えてズレるなら、
    このI/Fは “全核種を同時に説明する入口” として棄却。
- **A-trend 整合**：`abs(z_Δmedian) ≤ 3`
  - 定義：A-bin を固定（例：low=[40,60), high=[250,300)）し、
    `Δmedian = median_high − median_low` を計算して `z_Δmedian=Δmedian/σ_proxy` を評価する。
  - 解釈：A 依存が “手続き起源” で残っているなら棄却。
- **代表核種（shape-only）**：
  - 代表核種（CSVに固定）に対し、`scale_to_median_1`（中央値が1になる正規化）を報告し、
    その正規化後に `abs(z)` が多数で 3 を超えるなら、
    距離I/F（R vs d）の差分だけでは説明が足りず、追加物理が必要と結論づける。
- **差分チャネル精度ゲート**：
  - P-SEMF / P-Yukawa ごとに `σ_req,rel` を固定し、
    `σ_rel(total) <= σ_req,rel` を満たせる核種群のみを 3σ判定対象として採用する。
- **独立cross-checkの必須化（Step 7.13.17.14）**：
  - 差分チャネルの3σ判定は、分離エネルギー（Sn/S2n, shell-gap）と
    電荷半径kink（A_min=100, pairing凍結）の固定指標を同時に併記する。
  - JSON条件 `channel_gate_requires_independent_crosschecks` を満たさない判定は
    「採用」ではなく「判定保留」とする。

このStepでの結論：
- global（x=R）と local（x=d）いずれも、現状の凍結I/Fのままでは `abs(z_median) > 3` となり、
  “追加物理なしでの全核種説明” という意味では棄却される。
- ただし A-trend 指標（z_Δmedian）は両者とも 3σ内に収まるため、
  支配的な問題は “距離I/Fの解釈の違い” による **形状**よりも、
  **全体スケール（中央値）**の不整合にあることが分かる。
- 差分チャネル統合後の基準値（中央値）は、
  P-SEMF が `σ_req,rel ≈ 2.70%`、P-Yukawa が `σ_req,rel ≈ 33.43%` であり、
  反証優先度は P-SEMF 側を先行させるのが運用上合理的である。
- 7.13.17.14 で independent cross-check を統合した結果、`gate_status` として
  `radius_strict_pass_A100_pairing=true`、`separation_gap_available=true` を固定し、
  差分チャネル判定の最小ガードレールを明示できるようになった。

次（ロードマップ）：
- 7.13.18（または 7.13.17 の改訂）として、追加物理（飽和・局所近傍の bond count 上限・shell/pairing の二次項）を
  “反証条件パックの閾値を先に固定した上で” 導入する。
