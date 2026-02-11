# 24. 核結合エネルギー（Δω→B.E.写像）モデル：重陽子の数値検証（ERE cross-check; Step 7.13.17.4）

目的：
- Step 7.13.17.2–7.13.17.3 で凍結した写像（B=ħΔω, Δm=B/c^2）を、最小系である deuteron（pn束縛）で数値的に検証する。
- ここでの「検証」は P-model の核力導出ではなく、**既知の低エネルギー散乱パラメータ（a_t, r_t）から
  deuteron の束縛スケール（B, κ）が再構成できる**ことを、再現可能な I/F として固定する。

位置づけ（重要）：
- これは「既存核力理論の正しさ」を議論するStepではない。
- 後段（7.13.17.5 以降）で P-model 側の入口（近接場干渉→2モード近似）を入れる前に、
  **観測側の最小整合条件**（束縛スケールと散乱パラメータの整合）を機械的に凍結する。

関連：
- deuteron 一次固定（B_obs, 1/κ, r_d）：`output/quantum/nuclear_binding_deuteron_metrics.json`
- np 散乱一次固定（a_t, r_t；Eq.(18)(19)）：`output/quantum/nuclear_np_scattering_baseline_metrics.json`
- 2体境界条件I/F（例：square well）：`doc/quantum/22_nuclear_binding_energy_frequency_mapping_deuteron_two_body.md`

---

## 24.1 方法（ERE の束縛極）

低エネルギー s波の effective range expansion（ERE）を、最小（2次）で

$$
k\\cot\\delta(k) = -\\frac{1}{a_t} + \\frac{r_t}{2}k^2
$$

とする。

散乱振幅の極（束縛状態）は k=iκ（κ>0）で

$$
k\\cot\\delta(k) - ik = 0
$$

を満たす。
ERE を解析接続して代入すると

$$
\\frac{r_t}{2}\\,\\kappa^2 - \\kappa + \\frac{1}{a_t} = 0
$$

となる。解は2つあるが、deuteron は小さい方（κ≈0.23 fm^-1）を物理解として採用する。

束縛エネルギーは

$$
B_{\\rm pred} = \\frac{(\\hbar c)^2\\,\\kappa^2}{2\\,\\mu c^2}
$$

で再構成できる（μ は pn 縮約質量）。

注意：
- Eq.(18)（GWU/SAID）と Eq.(19)（Nijmegen）は同一一次ソース内の別解析であり、
  **差分は統計誤差ではなく解析依存の系統 proxy**として扱う（half-range を sys として保存）。
- shape parameter v2t（k^4項）はこのStepでは無視する（κ≈0.23 では高次）。

---

## 24.2 固定出力（数値と図）

出力（凍結）：
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_verification.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_verification_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_verification.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_deuteron_verification.py`

結果の要点：
- Eq.(18)(19) から再構成した B_pred は、B_obs（CODATA）を **envelope（min–max）で包含**する。
- したがって、deuteron の束縛スケール（B, κ）は、triplet 低エネルギー散乱パラメータと
  **最小I/F（ERE束縛極）で整合**していることを固定できる。

次（ロードマップ）：
- Step 7.13.17.5：He-4 へ拡張（多体 reduction の入口を固定し、deuteron と同じ I/F で検証する）。

