# 26. 核結合エネルギー（Δω→B.E.写像）モデル：代表核種（A=2,4,12,16）の予備診断（Step 7.13.17.6）

目的：
- Step 7.13.17.4（deuteron）と Step 7.13.17.5（He-4）で凍結した入口を踏まえ、
  AME2020 全域（7.13.17.7）へ入る前に、**代表核種（A=2,4,12,16）**で
  A 依存の予備診断と、系統パターン（magic/pairing 等）の兆候を台帳化する。
- ここで固定するのは「一発で全域に行ける理論式」ではなく、
  **データI/F（AME2020 + charge radii）と、最小モデル（C の候補）で何が足りないか**である。

入力（一次ソース）：
- AME2020（IAEA/AMDC）：binding energy per nucleon（B/A）  
  `data/quantum/sources/iaea_amdc_ame2020_mass_1_mas20/extracted_values.json`
- IAEA charge radii compilation：charge rms radius r_rms  
  `data/quantum/sources/iaea_charge_radii/charge_radii.csv`
- deuteron anchor（B_d, R_ref=1/κ_d）：`output/quantum/nuclear_binding_deuteron_metrics.json`
- range proxy（L=λπ）：`output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json`

---

## 26.1 固定I/F（このStepでの比較枠）

このStepでは、Step 7.13.17.5 の枠を「代表核種」に機械的に適用し、C の候補感度を確認する。

**距離スケール**

- anchor：deuteron の tail スケールを参照距離として凍結

  $$ R_{\\rm ref} \\equiv 1/\\kappa_d $$

- 他核種：charge rms 半径から一様球等価半径を構成（アルゴリズムとして凍結）

  $$ R \\equiv \\sqrt{\\frac{5}{3}}\\, r_{\\rm rms} $$

**range**

$$ L \\equiv \\lambda_{\\pi} $$

**結合スケール（包絡）**

$$
J_E(R) = J_E(R_{\\rm ref})\\exp\\left(\\frac{R_{\\rm ref}-R}{L}\\right),\\qquad
J_E(R_{\\rm ref}) = B_d/2
$$

**多体系 reduction**

$$
B_{\\rm pred} = 2 C J_E(R)
$$

このStepでは C の候補として
- C=A−1（collective）
- C=Z·N（pn_only）
- C=A(A−1)/2（pairwise_all）
を比較し、同時に

$$
C_{\\rm required} \\equiv \\frac{B_{\\rm obs}}{2J_E(R)}
$$

を計算して A に対する傾向を固定する（「何が足りないか」の指標）。

---

## 26.2 固定出力

出力（凍結）：
- `output/quantum/nuclear_binding_energy_frequency_mapping_representative_nuclei.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_representative_nuclei_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_representative_nuclei.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_representative_nuclei.py`

読み方（このStepの結論の範囲）：
- 代表核種に対して、**C の取り方で B_pred/B_obs が大きく動く**ことを「手続き依存の系統」として固定する。
- C_required の A 依存（どれくらい C を増やさないと AME2020 の B を説明できないか）を、
  次の Step 7.13.17.7（AME2020全域）へ持ち込む “不足量台帳” とする。

次（ロードマップ）：
- Step 7.13.17.7：AME2020 全核種へ拡張し、残差分布と破綻領域（magic/pairing 等）を固定する。

