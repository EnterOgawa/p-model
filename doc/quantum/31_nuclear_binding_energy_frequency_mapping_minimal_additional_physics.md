# 31. 核結合エネルギー（Δω→B.E.写像）モデル：最小追加物理（飽和・局所近傍）の固定（Step 7.13.18）

目的：
- Step 7.13.17.10 で凍結した反証条件パック（3σ運用）を **変更せず**、
  Δω→B.E. 写像の入口I/Fへ「最小の追加物理」を入れて、
  pass/fail がどう動くかを固定する。
- ここでの最小追加物理は、shell/pairing のような二次構造の前に、
  全核種で支配的だった “大域スケール（中央値）” の不整合を、
  **局所近傍（saturation）**の仮定で抑えることにある。

入力（固定出力）：
- Step 7.13.17.9 の差分CSV（local spacing d を含む）  
  `output/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.csv`
- Step 7.13.17.10 の反証条件パック（閾値の読み込みに使う）  
  `output/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.json`

固定出力：
- `output/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics.py`

---

## 31.1 追加物理（凍結I/F）

距離I/F（凍結）：
- 指数減衰の距離として、A-trend を大きく除去できた local spacing proxy を採用する（Step 7.13.17.9 と同じ）。

結合数I/F（追加・凍結）：
- baseline の “coherent bond count” は `C_collective=A−1` としていた。
  これは核子あたりの結合数（指標）として
  `ν_base = 2(A−1)/A`（A→∞で 2）を意味する。
- 最小追加物理として、**局所近傍の飽和**を
  `ν_eff = min(ν_base, ν_sat)` で導入し、
  `C_eff = (ν_eff·A)/2` を用いる。
- 本Stepでは `ν_sat=1.5` を **凍結**する。
  これにより A<=4（d, 3He, 4He）は baseline と一致し、
  A>4 では過大な ν_base の成長が抑制される（= heavy-A の overbinding を抑える）。

注意：
- ここでの ν は “物理的な最近接配位数” の主張ではなく、
  Δω→B.E. 写像の I/F を閉じるための **操作的な結合数指標**である。

---

## 31.2 結果（反証条件パックに対する影響）

反証条件パックの閾値（凍結）：
- `abs(z_median) ≤ 3`（中央値整合）
- `abs(z_Δmedian) ≤ 3`（A-bin固定での A-trend 整合）

結果（要点）：
- baseline（local spacing d のみ）：
  - `z_median=5.63` で fail（中央値が 3σ_proxy を超える）
  - `z_Δmedian=2.39` は pass
- 最小追加物理（local spacing d + ν saturation）：
  - `z_median=1.31` で pass
  - `z_Δmedian=2.21` で pass

したがって本Stepの結論は、
「距離I/F（d）を保ったまま、結合数の飽和（局所近傍）を入れるだけで、
凍結した 3σ 運用の acceptance に入る」ことである。

次（ロードマップ）：
- 7.13.19 以降（または 7.13.18 改訂）として、
  pairing/shell を二次補正として “系統” へ落とし込み、
  どの自由度が必要か（過剰適合でないか）を反証条件パックの形式で固定する。

