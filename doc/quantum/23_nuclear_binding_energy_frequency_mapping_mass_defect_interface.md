# 23. 核結合エネルギー（Δω→Δm→B.E.写像）モデル：質量欠損のI/F固定（Step 7.13.17.3）

目的：
- Step 7.13.17.1–7.13.17.2 で導入した「結合＝周波数低下 Δω」という入口について、
  **質量欠損 Δm と結合エネルギー B.E. を結ぶ写像（I/F）**を、c の冪を含めて凍結する。
- 以後の多体系（He-4、核種横断）で、議論が「単位の混乱」や「定義の取り替え」で崩れないよう、
  入力（観測）と出力（モデル）の変数関係を台帳として固定する。

関連：
- 最小仮定台帳：`doc/quantum/21_nuclear_binding_energy_frequency_mapping_min_assumptions.md`
- 2体問題（deuteron；境界条件I/F）：`doc/quantum/22_nuclear_binding_energy_frequency_mapping_deuteron_two_body.md`

---

## 23.1 凍結する写像（単位・cの冪）

本プロジェクト（Part I/III）の定義に従い、質量と周波数は

$$
m \\equiv \\frac{\\hbar\\,\\omega_0}{c^2}
$$

で結ぶ（ω0 は rest frequency）。

核の結合エネルギー B（B.E.）を「周波数低下 Δω」に写像する約束は

$$
B \\equiv \\hbar\\,\\Delta\\omega
$$

で固定する（Step 7.13.16/7.13.17.2 と整合）。

したがって質量欠損は

$$
\\Delta m \\equiv \\frac{B}{c^2}
= \\frac{\\hbar\\,\\Delta\\omega}{c^2}
$$

となる（ここで **c^2** を凍結）。

---

## 23.2 観測側の定義（一次固定）

観測（一次ソース）側で固定される結合エネルギーは、核質量（核の質量）による質量欠損として

$$
B_{\\rm obs}(A,Z) \\equiv
\\left[ Z m_p + (A-Z) m_n - M_{\\rm nuc}(A,Z) \\right] c^2
$$

で定義する（本リポジトリでは CODATA/NIST の核質量を使用）。

ここで：
- A：質量数
- Z：陽子数
- N≡A−Z：中性子数

この B_obs から、周波数側の観測量（写像）は

$$
\\Delta\\omega_{\\rm obs}(A,Z) \\equiv \\frac{B_{\\rm obs}(A,Z)}{\\hbar}
$$

で固定される。

---

## 23.3 幾何学的因子と量子数（入口の整理）

本ロードマップでは、核力の細部を第一原理で導出したとは主張しない。
代わりに、境界条件で書ける束縛条件を I/F として固定し、そこに現れる「幾何」と「量子数」の依存を
後段の系統分析（magic/pairing 等）へ分離して持ち込む。

### (a) 幾何学的因子（例：square well の x≡kR）

Step 7.13.17.2 の例（square well；s-wave）では、境界条件は

$$
k\\cot(kR) = -\\kappa
$$

と書ける。

ここで
- R：レンジ（幾何）
- k：内側の波数
- κ：外側tailの指数（κ=\\sqrt{2\\mu B}/\\hbar）

であり、無次元量

$$
x \\equiv kR,\\qquad \\kappa R
$$

が、束縛条件を決める「幾何学的因子」になる。
（l>0 の場合は spherical Bessel/modified Bessel を用いた同種の境界条件へ拡張されるが、
この段階では導出しない。）

### (b) 量子数依存（l,s,j）

deuteron は S波優勢であるため、本Stepの最小 I/F は l=0 を入口として固定する。

- スピン・テンソル（D波混合）・アイソスピンの詳細は Out of Scope（Step 7.13.17.1 で宣言）とし、
  必要になった段階で「追加仮定」として別Stepで最小自由度で導入する。

---

## 23.4 一般式（モデル側の “予測I/F” の形を固定）

モデル側は、核種（A,Z）に対して周波数欠損を予測する関数

$$
\\Delta\\omega_{\\rm pred}(A,Z;\\,\\theta)
$$

を与え、結合エネルギーは

$$
B_{\\rm pred}(A,Z;\\,\\theta) \\equiv \\hbar\\,\\Delta\\omega_{\\rm pred}
$$

で定義する（ここまでが本Stepの凍結）。

θ（凍結パラメータ）の候補は次の範囲に限定する：
- near-field 干渉→2モード近似で導入した結合包絡 J(R) の I/F（距離依存の形、range proxy）
- 幾何（R の定義）と、pn/nn/pp の選別係数（自由度の数を固定）

多体（pair-wise sum vs collective）をどう扱うかは Step 7.13.17.5 で I/F を決める。
本Stepでは「B=ħΔω」の写像と、Δω_pred の引数集合だけを固定する。

---

## 23.5 固定出力（sanity check）

light nuclei（A=2,3,4）の一次固定値 B_obs から Δω_obs と Δm を再構成し、
写像が自明に整合することを出力として凍結する。

- `output/quantum/nuclear_binding_energy_frequency_mapping_interface.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_interface_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_interface.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_interface.py`

