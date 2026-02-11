# Step 7.15（統計力学・熱力学）：黒体放射の基準量（baseline）

## 目的

Phase 7 の目的は「時間＝波の媒質」という仮定を量子側へ接続し、既知の現象と矛盾しないことを **反証可能な形**で固定することにある。  
本 Step 7.15 では、温度・エントロピー・平衡分布（熱統計）が P-model と矛盾しないかを整理する。

初版では、導出主張を最小にし、まず **黒体放射の基準量（スケール）**を一次ソース（SI/CODATA）で固定し、後段の拡張仮説が満たすべき制約（=棄却条件）として扱う。

---

## 1) 一次ソース（定数の固定）

黒体放射の数値スケールは、SI 基本定数（c, h, k_B）と数学定数（π, ζ(3)）で決まる。  
本リポジトリでは NIST Cuu（CODATA）ページを **HTMLで固定**し、派生 JSON（extracted_values）を作る。

- 保存先：`data/quantum/sources/nist_codata_2022_blackbody_constants/`
- 取得：`python -B scripts/quantum/fetch_blackbody_constants_sources.py`

参照（一次）：`doc/PRIMARY_SOURCES.md`（「統計力学・熱力学（黒体放射の基準量；NIST CODATA 定数）」）

---

## 2) baseline（黒体放射の基準量）

黒体放射について、最低限のスケール固定として次を扱う：

- Stefan–Boltzmann 定数（σ）
- 放射定数（a；u=aT^4）
- 光子数密度の係数（nγ∝T^3）
- 平均光子エネルギー（⟨E⟩≈2.701 k_B T）

### 再現（本リポジトリ）

```powershell
python -B scripts/quantum/thermo_blackbody_radiation_baseline.py
```

- 図：`output/quantum/thermo_blackbody_radiation_baseline.png`
- 数値：`output/quantum/thermo_blackbody_radiation_baseline_metrics.json`
- 表：`output/quantum/thermo_blackbody_radiation_baseline.csv`

---

## 3) 位置づけ（P-model 側の制約）

P-model は「時間の刻み（時計）」の機構を与えることを狙うが、平衡統計（黒体、熱力学第1・第2法則）と矛盾すれば棄却される。  
したがって本 Step の baseline は、今後の拡張（例：P の揺らぎや相互作用を熱浴として扱う等）に対する **必須の整合条件**として機能する。

例（棄却条件の形式）：

- 平衡の黒体スペクトルが、同一温度で Planck 分布から系統的にずれる（単なる校正差ではない）ことが 3σ で再現されるなら、当該の拡張仮説は棄却。
- エネルギー密度の温度スケーリングが u∝T^4 から系統的にずれるなら、拡張仮説は棄却。

（次段階）温度・エントロピーの P-model 内での操作的定義を提案する場合も、上の baseline を満たすことを前提に、  
追加自由度（仮定）と反証条件をセットで固定する必要がある。

