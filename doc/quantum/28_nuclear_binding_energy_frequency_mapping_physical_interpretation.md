# 28. 核結合エネルギー（Δω→B.E.写像）モデル：物理的解釈の整理（Step 7.13.17.8）

目的：
- Step 7.13.17.7 で凍結した「AME2020 全核種の残差分布」を、
  数式（I/F）から物理語（描像）へ翻訳し、**どの不足がどの追加物理に対応するか**を整理する。
- 本Stepは **追加パラメータの導入はしない**。
  代わりに、次の Step 7.13.17.9–7.13.17.10 で差分予測/反証条件へ接続できるよう、
  「支配する不足（1つ）」と「二次的な不足（複数）」を分離して台帳化する。

入力（固定出力）：
- Step 7.13.17.7 の凍結CSV/metrics  
  `output/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv`  
  `output/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei_metrics.json`

---

## 28.1 何が凍結されているか（I/Fの再掲）

本Stepで “動かさない” もの：
- anchor：`R_ref=1/κ_d`、`J_ref=B_d/2`（deuteron で固定）
- range：`L=λπ`（deuteron 2体の range proxy で固定）
- 幾何 proxy（全核種適用の仕様）：
  - radii 測定あり：`R=√(5/3) r_rms`
  - radii 測定なし：`R=r0·A^(1/3)`（r0 は measured radii 由来 median を凍結）
- 結合スケール：`J_E(R)=J_ref exp((R_ref−R)/L)`
- B.E. 写像（baseline）：`B_pred = 2 C J_E(R)`、`C=A−1`

したがって、Step 7.13.17.7 の残差（B_pred/B_obs）の A 依存は、
主に **R を指数関数の中に入れたこと**と **C の取り方**に由来する。

---

## 28.2 何が観測的に見えているか（残差パターンの要点）

Step 7.13.17.7（baseline C=A−1）から読み取れる事実（数値は metrics 由来）：
- B_pred/B_obs の全体中央値は `≈ 0.064`（大域的に強い過小予測）。
- 偶奇（ee/eo/oe/oo）で中央値がほぼ同じ（pairing 的な偶奇効果はこの coarse I/F では主成分ではない）。
- magic フラグ（Z または N が magic）では中央値が `≈ 0.13` とやや大きいが、
  依然として 1 からは大きく離れる（shell closure は二次項の可能性が高い）。
- A が増えるほど B_pred/B_obs が単調に小さくなる（A依存の主不足がある）。

この時点で “主不足” は **A依存（幾何/range の扱い）**であり、
pairing/shell は **残差の上に乗る構造**として扱うのが自然である。

---

## 28.3 数式→物理語：何が「支配不足」か

### (a) baseline `C=A−1` が意味するもの

`B_pred = 2 C J_E` の形で C を `A−1` に固定すると、
1核子あたりの “有効な結合本数” を

`ν_eff ≡ 2C/A`

と定義したとき、baseline は `ν_eff≈2` を意味する。
これは「各核子が O(1) 個の相手としか coherent に結合しない」という最小形（鎖的）に近い。

### (b) しかし frozen `J_E(R)`（Rを指数に入れる）を保ったまま AME2020 を説明するには？

AME2020 の B_obs を再現するために必要な “有効結合本数” を

`ν_eff_required ≡ 2C_required/A = B_obs/(J_E·A)`

として CSV から再構成すると、
A とともに `ν_eff_required` が急速に増える（A=100 で ≈20、A=200 で ≈50 程度が典型）。
これは **核物質の飽和（局所相互作用で近接数が O(1) に留まる）**と整合しにくい。

結論として、この mismatch の主因は
「C を A−1 にしたこと」そのものより、
**核の全半径 R を “距離” と見なして J_E を指数減衰させた幾何 proxy**にある可能性が高い。

固定出力（診断図）：
- `output/quantum/nuclear_binding_energy_frequency_mapping_physical_interpretation.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_physical_interpretation_metrics.json`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_physical_interpretation.py`

---

## 28.4 次Stepへ渡す「最小追加物理」の候補（ここでは導入しない）

ここでの目的は “追加物理の候補を分離しておく” ことであり、
採否は Step 7.13.17.9（差分予測）と Step 7.13.17.10（反証条件）で決める。

候補（優先度順）：
1) **距離の再解釈（global R ではなく local separation）**  
   核半径 R は密度飽和の結果として A^(1/3) で増える一方、
   近接場の相互作用距離（最近接核子間距離）は A に強く依存しない。
   したがって、指数減衰の距離として核半径 R を入れること自体が I/F として不適切である可能性がある。
2) **有限range + 飽和（“結合本数” の上限）**  
   range（λπ）により、結合は近接の有限個へ限定される。
   この場合 C は “全ペア” ではなく、局所近傍の bond count（O(A)）になる。
3) **shell/pairing は二次項として追加**  
   Step 7.13.17.7 の parity の差が小さいことから、
   pairing は “全体スケールを決める主因” ではなく、kink/局所残差を説明する二次補正として位置づけるのが筋が良い。

---

## 28.5 Out of Scope（このStepで扱わない）

- QCD/EFT（湯川型・有効理論）との厳密対応は、現時点では “物理的解釈の補助” に留める。
  差分予測・反証条件を先に固定し、後から参照枠として整理する。

