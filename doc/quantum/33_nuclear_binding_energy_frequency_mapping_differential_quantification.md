# 33. 核結合エネルギー（Δω→B.E.写像）モデル：差分予測の定量化（Step 7.13.17.12）

目的：
- Step 7.13.17.11 の差分（`DeltaB = B_pred_Pmodel - B_pred_standard`）を
  核種ごとに固定し、3sigma判定に必要な観測精度を数値化する。

入力（固定出力）：
- `output/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.csv`

定義（固定）：
- `DeltaB = B_pred_Pmodel - B_pred_standard`
- `sigma_req_3sigma = abs(DeltaB) / 3`
- `required_relative_sigma = sigma_req_3sigma / B_obs`

固定出力：
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification.csv`
- `output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification_top20.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification.py`

---

## 33.1 主要結果（全核種）

| 指標 | P-SEMF | P-Yukawa proxy |
|---|---:|---:|
| median abs(DeltaB) [MeV] | 75.15 | 1150.06 |
| p84 abs(DeltaB) [MeV] | 230.77 | 1866.18 |
| max abs(DeltaB) [MeV] | 472.66 | 2532.92 |
| median required_relative_sigma | 0.0270 | 0.3343 |

重核（A>=121）では、P-SEMF の `median abs(DeltaB)` は 145.22 MeV、
`median required_relative_sigma` は 0.0340（約3.4%）となる。
したがって、差分決着点は軽核より重核側に強く集中する。

---

## 33.2 運用上の使い方

- `..._top20.csv` を「優先核種リスト」として固定し、
  今後の差分検証（追加自由度導入前）で同じ核種群を追跡する。
- 3sigma要件は `sigma_req_3sigma` をそのまま使い、判定閾値を後から変更しない。

---

## 33.3 注意

- `sigma_B_obs` は AME 側の表値を運用指標として使う。
  実験統計・系統の完全分離を主張するものではない。
