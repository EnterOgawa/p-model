# 29. 核結合エネルギー（Δω→B.E.写像）モデル：差分予測の抽出（Step 7.13.17.9）

目的：
- Step 7.13.17.7–7.13.17.8 で見えた「主不足（A依存の幾何/range proxy）」を、
  **追加fitをせず**に “差分予測” として定式化する。
- ここで言う差分予測は、標準核力（湯川/EFT）との直接対決ではなく、
  **本稿の Δω→B.E. 写像 I/F の中で「距離」をどう解釈するか**（global R vs local separation）が
  残差パターンをどう変えるか、という *操作的な予測* を固定する。

入力（固定出力）：
- Step 7.13.17.7 の凍結CSV（全核種）  
  `output/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv`

---

## 29.1 2つの「距離」解釈（同じanchor/rangeで比較）

凍結された共通部分（動かさない）：
- anchor：`R_ref=1/κ_d`、`J_ref=B_d/2`
- range：`L=λπ`
- 結合スケール：`J_E(x)=J_ref exp((R_ref−x)/L)`
- 多体系係数：`C=A−1`（baseline）
- B.E. 写像：`B_pred = 2 C J_E`

差分が入るのは **x（距離変数）** の定義だけ：

### (A) global 距離 proxy（従来；Step 7.13.17.7）

`x=R_model`（核半径 proxy）をそのまま指数減衰距離に入れる。

### (B) local separation 距離 proxy（本Step）

核半径 `R_model=r0·A^(1/3)` から一様密度を仮定して

`ρ = 3/(4π r0^3)`,
`d ≡ ρ^(−1/3) = (4π/3)^(1/3) · r0`

を **最近接距離スケール（local spacing）** の proxy として採用し、
`x=d` を指数減衰距離に入れる。

重要：この置換は “fit” ではなく、同じ凍結I/Fの中で
「R を距離とみなすのは不適切かもしれない」という仮説の監査である。

---

## 29.2 固定結果（差分予測としての出力）

凍結出力（差分比較）：
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.py`

主要数値（metrics の median）：
- global（x=R）：`median(B_pred/B_obs)≈0.064`（大域的に強い過小予測）
- local（x=d）：`median(B_pred/B_obs)≈1.43`（A-trend が大きく緩和し、過小→過大へ移る）

解釈（このStepで固定すること）：
- 全核種の mismatch の主成分は **「指数減衰距離に何を入れたか」**で説明可能である。
  すなわち、`R` を距離に入れると A とともに単調に過小になり、
  `d` を距離に入れると A依存は大きく抑えられる（残差は O(1) の帯に圧縮される）。
- parity/magic の group medians は local mapping ではほぼ同程度に揃い、
  この段階では shell/pairing は “主因ではない” ことを補強する（主因は距離解釈）。

---

## 29.3 次（反証条件へ接続するための最小化）

この差分は「どちらが正しいか」をここで断定するためではなく、
次の Step 7.13.17.10（反証条件）で、

- “距離は global R か local separation か” を区別できる観測量
- その観測量が既存データで決着するか／将来測定が必要か

を、最小セットで固定するための入口である。

