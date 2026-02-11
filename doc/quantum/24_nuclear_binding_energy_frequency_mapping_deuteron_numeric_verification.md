# 24. 核結合エネルギー（Δω→B.E.写像）モデル：重陽子での数値検証（Step 7.13.17.4）

目的：
- Step 7.13.17.1–7.13.17.3 で凍結した写像（B=ħΔω、Δm=B/c^2）を前提に、
  **重陽子（deuteron）**で「入力（np散乱）→予測（束縛スケール）」の整合を数値化し、
  以後の多体（He-4）・核種横断（AME2020）へ拡張する前の sanity check を固定する。

位置づけ：
- ここでの「予測」は、s波の有効レンジ展開（ERE）を 2次までで打ち切った
  **操作的I/F**による κ（束縛 pole）推定であり、核力の第一原理導出ではない。
- eq18/eq19（GWU/SAID vs Nijmegen）の差は、統計誤差ではなく
  **解析依存系統（analysis-dependent systematics）の proxy**として扱う。

---

## 入力（一次固定）

- deuteron 結合エネルギー（観測）：CODATA/NIST（質量欠損）で固定（Step 7.9）
- np 散乱（triplet）の低エネルギーパラメータ：arXiv:0704.1024v1 の eq18/eq19 を固定（Step 7.9.2）

---

## 処理（固定条件）

### 1) ERE（2次打ち切り）で束縛 pole κ を推定

s波散乱の ERE を

$$
k\\cot\\delta(k) \\approx -\\frac{1}{a_t} + \\frac{r_t}{2}k^2
$$

とし、束縛状態（k=iκ）の pole 条件

$$
k\\cot\\delta - ik = 0
$$

から

$$
\\frac{r_t}{2}\\kappa^2 - \\kappa + \\frac{1}{a_t} = 0
$$

を解いて κ を得る（正の小さい根）。

### 2) κ から B を再構成し、観測 B と比較

$$
B_{\\rm pred} = \\frac{(\\hbar c)^2\\kappa^2}{2\\mu c^2}
$$

（μは pn 縮約質量）で B_pred を得て、
観測 B_obs（CODATA）との差を残差として記録する。

### 3) 不確かさ（stat/sys）

- stat：CODATA の σ(B_obs)
- sys：eq18 と eq19 の予測差を半幅で定義（analysis-dependent systematics proxy）

---

## 結果（主要数値）

固定出力（図・CSV・JSON）に集約する：
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_numeric_verification.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_numeric_verification.csv`
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_numeric_verification_metrics.json`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_deuteron_numeric_verification.py`

解釈（短く）：
- eq18/eq19 の差は、B_pred に対して O(10^-2 MeV) 程度のばらつきを与える。
  これは deuteron の測定統計（σ_B）より支配的であり、
  「解析依存（v2 など高次項の扱い）」を系統として切り分ける必要があることを示す。

次：
- Step 7.13.17.5（He-4）で、多体取り込みI/F（pair-wise vs collective）を決め、
  同じ “sys/stat 分解” を維持したまま拡張する。

