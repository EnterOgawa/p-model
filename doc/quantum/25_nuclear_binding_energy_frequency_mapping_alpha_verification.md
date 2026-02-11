# 25. 核結合エネルギー（Δω→B.E.写像）モデル：He-4（α粒子）の検証拡張（Step 7.13.17.5）

目的：
- Step 7.13.17.4（deuteron）で凍結した「束縛スケールの操作的I/F」を、
  4体系（He-4; α粒子）へ拡張し、**多体系 reduction の入口**を最小形で固定する。
- ここでは QCD の第一原理導出は行わず、**deuteron で固定したスケール**から
  α粒子の結合エネルギーを再構成するための「手続き」を出力として凍結する。

位置づけ（重要）：
- 本Stepは「P-model核力の完成」を主張しない。
- 目的は、後段（代表核種→AME2020全域）へ進むために、
  **多体系の数え上げ（C）と、距離スケールの置き方（R_ref, R_alpha）**を
  “恣意的選択ではなく、明示されたアルゴリズム”として固定することである。

関連：
- deuteron baseline（B_obs, 1/κ）：`output/quantum/nuclear_binding_deuteron_metrics.json`
- light nuclei baseline（αのB_obs, r_rms）：`output/quantum/nuclear_binding_light_nuclei_metrics.json`
- range proxy（λπ）：`output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json`

---

## 25.1 固定I/F（距離スケールと多体系 reduction）

### (a) 距離スケール（R_ref, R_alpha）の固定

- **参照距離**：deuteron の tail から直接定義されるスケールとして

  $$ R_{\\rm ref} \\equiv 1/\\kappa_d $$

  を採用する（Step 22.2 の定義）。

- **α粒子のスケール**：CODATA の α粒子 charge rms 半径 r_rms を、等価な一様球の半径へ変換して

  $$ R_{\\alpha} \\equiv \\sqrt{\\frac{5}{3}}\\,r_{\\rm rms}(\\alpha) $$

  を採用する（定義アルゴリズムとして凍結）。

### (b) 結合スケールの距離依存（near-field 包絡）

Step 7.13.16.4 の I/F に合わせ、結合スケールを指数包絡で

$$
J_E(R) = J_E(R_{\\rm ref})\\,\\exp\\left(\\frac{R_{\\rm ref}-R}{L}\\right)
$$

とする。
ここで range は

$$
L \\equiv \\lambda_{\\pi}
$$

（π± Compton proxy）を用いる（Step 7.13.17.2 の固定値）。

deuteron の約束（Step 22.4）により

$$
J_E(R_{\\rm ref}) = B_d/2
$$

が固定されるので、J_E(R_alpha) は追加フィット無しに決まる。

### (c) 多体系 reduction（C）の候補

α粒子の結合エネルギーは

$$
B_{\\rm pred}(\\alpha) = 2\\,C\\,J_E(R_{\\alpha})
$$

と置き、C の候補を比較する：

- collective：C=A−1（coherent in-phase モードの最小数え上げ）
- pn_only：C=Z·N（pn のみ結合するという極端仮定）
- pairwise_all：C=A(A−1)/2（全ペアを単純加算；過剰カウントの疑いが強い）

本Stepの baseline は **collective（C=A−1）**とし、他2つは系統 envelope として保存する。

---

## 25.2 固定出力

出力（凍結）：
- `output/quantum/nuclear_binding_energy_frequency_mapping_alpha_verification.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_alpha_verification_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_alpha_verification.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_alpha_verification.py`

結果の読み方：
- C_required（観測を満たすために必要なC）が、collective（C=3）に近いかどうかで、
  この最小 reduction が “大外し” していないかを確認する。
- pairwise_all が大きく外れる場合は、「結合を単純にペア加算するのは過剰カウント」という
  反例として利用できる（過剰自由度の導入を避けるためのガードレール）。

次（ロードマップ）：
- Step 7.13.17.6：代表核種（A=2,4,12,16）へ拡張し、A依存の系統パターン（magic/pairing など）を予備診断する。

