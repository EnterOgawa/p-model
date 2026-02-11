# Step 7.6：重力誘起デコヒーレンス（可視度低下の予言と反証）

## 目的

重力（時間構造）が量子干渉の **可視度（visibility）** や **位相雑音** に与える寄与を、一次ソースに基づき整理し、
P-model の立場でも反証可能な形（観測量・スケーリング・棄却条件）へ落とす。

この Step は「量子現象を説明した」と主張するものではなく、**重力×量子干渉の観測量が、時間構造（P）とどこで結び付くか**を明確化するための設計・最小再現である。

参照（一次）：`doc/PRIMARY_SOURCES.md`

---

## 1) 観測量（最小）

- 干渉の可視度（Ramsey contrast 等）：V(T)
- 位相雑音：σφ（run-to-run、またはサンプル内での inhomogeneous）

ガウス位相雑音の最小モデルでは、

$$
V = \exp\left(-\frac{\sigma_\phi^2}{2}\right)
$$

となる。

---

## 2) 重力による inhomogeneous dephasing（光格子時計の原子集団）

高さ z の原子は重力赤方偏移により

$$
\frac{\Delta f}{f} \simeq \frac{g z}{c^2}
$$

の周波数差を持つ。原子が高さ分布（標準偏差 σz）を持つと、同一 Ramsey でも位相がばらつき、

$$
\sigma_\phi \simeq \omega_0 \left(\frac{g\,\sigma_z}{c^2}\right) T
$$

となり、V(T) が低下する。

Hasegawa et al.（arXiv:2107.02405v2）は、この効果が将来の高安定度時計（≲10^-21）では観測可能になり得ること、
また 1 s stability に ~10^-19 近傍の制限が現れ得ることを議論している。

---

## 3) time dilation による decoherence（内部自由度↔重心の結合）

Pikovski et al.（arXiv:1311.1095v2）は、内部自由度がある複合系では、重力 time dilation により内部自由度と重心が結合し、
重心位置のコヒーレンスが低下し得ると主張した（実験文脈・普遍性については関連論争あり：arXiv:1509.04363v3、1507.05828v5、1509.07767v1）。

本リポジトリでは、まず「観測量（V, σφ）」と「どの時間差が効くか（Δτ）」を固定し、個別実験で反証可能な形へ落とすことを優先する。

---

## 4) P-model 側：時間構造（P）の揺らぎを「σy」として定量化

P-model では弱場で

$$
\frac{d\tau}{dt} = \frac{P_0}{P}
$$

であり、P に run-to-run の揺らぎ（または腕間の差）があるなら、それは「有効な時計レート雑音」として
干渉可視度を低下させ得る。

最小のパラメータ化として、RMS の分数レート雑音を σy と置くと、

$$
V = \exp\left(-\frac{(\omega_0\,\sigma_y\,T)^2}{2}\right)
$$

で「その雑音が decoherence に見える条件」を定量化できる（実験では腕間相関などに注意）。

---

## 再現（本リポジトリ）

```powershell
python -B scripts/quantum/gravity_induced_decoherence.py
```

- 出力：`output/quantum/gravity_induced_decoherence.png`
- 数値：`output/quantum/gravity_induced_decoherence_metrics.json`
