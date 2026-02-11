# 22. 核結合エネルギー（Δω→B.E.写像）モデル：2体問題（deuteron）の最小定式化（Step 7.13.17.2）

目的：
- Step 7.13.17.1 で凍結した写像（E=ħω, m=ħω0/c^2）を前提に、
  deuteron（pn束縛）の **2体問題**として「束縛条件＝共鳴/定在波条件」を最小限の形で固定し、
  結合エネルギー B.E. を **周波数低下 Δω**へ写像する入口を明文化する。

位置づけ（重要）：
- ここで行うのは「核力の第一原理導出」ではない。
- 2体の **境界条件（連続性）**で書ける束縛条件を固定し、
  後段で “P-model固有の入口（近接場干渉→2モード近似）” と整合させるための I/F を作る。

関連：
- 最小仮定台帳：`doc/quantum/21_nuclear_binding_energy_frequency_mapping_min_assumptions.md`
- 近接場干渉→2モード近似：`doc/quantum/20_nuclear_near_field_interference_two_mode_model.md`
- deuteron一次固定（B, 1/κ, r_d）：`scripts/quantum/nuclear_binding_deuteron.py`

---

## 22.1 2体（相対座標）の最小方程式（操作的I/F）

相対座標 r に対する（非相対論）相対運動の最小形として、次を入口として採用する：

$$
\\left[-\\frac{\\hbar^2}{2\\mu}\\nabla^2 + V(r)\\right]\\psi(r) = -B\\,\\psi(r)
$$

- μ：縮約質量（pn）
- B：束縛エネルギー（deuteron：一次固定；Step 7.9）
- V(r)：有効相互作用（形はこの段階では主張しない）

ここで固定したいのは、V(r) の細部ではなく、
「束縛が成立する条件」を **境界条件として**書ける形である。

---

## 22.2 束縛状態の tail（外側解）と κ（一次固定）

外側（V=0）の解は tail 指数 κ で特徴づけられる：

$$
\\psi(r) \\propto \\frac{e^{-\\kappa r}}{r}, \\qquad
\\kappa = \\frac{\\sqrt{2\\mu B}}{\\hbar}
$$

したがって 1/κ は「束縛波の外側スケール」を与える（Step 7.9 で一次固定）。

---

## 22.3 例：square well による「定在波条件」の最小表現

核力の形を主張しないために、ここでは “例” として square well を用いて、
境界条件から束縛条件がどう現れるかだけを固定する。

$$
V(r)=
\\begin{cases}
-V_0 \\quad (r<R)\\\\
0 \\quad (r\\ge R)
\\end{cases}
$$

このとき内側（r<R）の解は

$$
\\psi(r) \\propto \\frac{\\sin(kr)}{r},
\\qquad
k = \\frac{\\sqrt{2\\mu(V_0-B)}}{\\hbar}
$$

となり、r=R での連続条件（ψ と dψ/dr の連続）から、
s波束縛の定在波条件は

$$
k\\cot(kR) = -\\kappa
$$

（同値に：x≡kR として x cot x + κR = 0）となる。

この式は「(i) 外側tail（κ）」「(ii) 内側レンジ（R）」「(iii) 内側スケール（V0）」の
対応を固定する最小の I/F であり、以後の near-field 定義にも使える。

---

## 22.4 B.E.→Δω の写像（本ロードマップの約束）

Step 7.13.17.1 の凍結に従い、束縛エネルギーは周波数低下（角周波数差）へ

$$
\\Delta\\omega \\equiv \\frac{B}{\\hbar}
$$

で写像する。

さらに Step 7.13.16（2モード近似）の I/F に接続する場合は

$$
\\Delta\\omega = 2J(R_0)
$$

を採用し、deuteron が要求する結合スケール（J_E(R0)=B/2）を固定する。

---

## 22.5 固定出力（数値）

上の写像（B, κ, Δω）と、例としての square well 解（R→V0）を
再現可能な形で出力として凍結する：

- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body.py`

次（ロードマップ）：
- Step 7.13.17.3：Δω→Δm→B.E. の写像（c の冪を含めて）を正式に固定し、
  多体（He-4）・核種横断（AME2020）へ拡張できる I/F へ整理する。

