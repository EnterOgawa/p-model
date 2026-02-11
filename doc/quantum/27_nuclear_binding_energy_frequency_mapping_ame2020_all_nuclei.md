# 27. 核結合エネルギー（Δω→B.E.写像）モデル：AME2020 全核種での残差分布固定（Step 7.13.17.7）

目的：
- Step 7.13.17.4–7.13.17.6 で凍結した最小I/F（anchor + range + 幾何 proxy + C候補）を、
  AME2020 全核種に機械適用し、
  **残差分布（B_pred/B_obs）**と **破綻領域（magic/pairing 等の兆候）**を
  “固定成果物”として残す。
- 本Stepでは **一致を主張しない**。一致しない箇所を
  「どの系統が支配しているか（A依存・偶奇・shell closure 等）」として可視化し、
  次の Step（7.13.17.8–7.13.17.10）で追加物理/反証条件へ接続するための台帳を作る。

入力（一次ソース）：
- AME2020（IAEA/AMDC; mass_1.mas20）：B/A（keV/A）  
  `data/quantum/sources/iaea_amdc_ame2020_mass_1_mas20/extracted_values.json`
- IAEA charge radii compilation（charge_radii.csv）：r_rms（測定値がある核種のみ）  
  `data/quantum/sources/iaea_charge_radii/charge_radii.csv`
- deuteron anchor（B_d, R_ref=1/κ_d）：`output/quantum/nuclear_binding_deuteron_metrics.json`
- range proxy（L=λπ）：`output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json`

---

## 27.1 固定I/F（このStepでの “全核種適用” の仕様）

### (a) anchor / range

- anchor（凍結）：

  $$ R_{\\rm ref} \\equiv 1/\\kappa_d,\\qquad J_{\\rm ref} \\equiv B_d/2 $$

- range（凍結）：

  $$ L \\equiv \\lambda_{\\pi} $$

### (b) 幾何 proxy（R の決め方）

測定 r_rms がある核種は

$$
R \\equiv \\sqrt{\\frac{5}{3}}\\, r_{\\rm rms}
$$

を採用する。

測定 r_rms が無い核種は “fallback” として

$$
R \\equiv r_0\\,A^{1/3}
$$

を採用する（r0 は measured radii（A>=4）からの median として凍結）。
この fallback は、全核種へ機械適用するための I/F であり、
ここでの一致/不一致は “モデル不足の検出” に使う（fit で吸収しない）。

### (c) 結合スケールと B_pred

$$
J_E(R) = J_{\\rm ref}\\exp\\left(\\frac{R_{\\rm ref}-R}{L}\\right)
$$

$$
B_{\\rm pred} = 2 C J_E(R)
$$

baseline は collective（C=A−1）で固定し、診断用に pn_only（C=Z·N）と pairwise_all（C=A(A−1)/2）も保存する。

---

## 27.2 固定出力

出力（凍結）：
- `output/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.py`

このStepで固定する “観測” は AME2020 の B/A であり、
解析の出力は「残差分布（比）」と「偶奇/魔法数フラグでの系統差」である。

次（ロードマップ）：
- Step 7.13.17.8：数式→物理語の翻訳（共鳴条件/周波数低下/準位）と、どの不足がどの追加物理に対応するかを整理する。

