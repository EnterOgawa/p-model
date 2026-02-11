# 32. 核結合エネルギー（Δω→B.E.写像）モデル：既存理論との差異抽出（Step 7.13.17.11）

目的：
- Step 7.13.18 で凍結した P-model（local spacing `d` + `nu_sat`）を、
  標準理論proxyと同一データ上で比較し、差分予測の入口を固定する。
- 比較は「同じ AME2020 ターゲットに対して、どの写像がどの規模で外れるか」を
  3sigma運用指標で監査する。

入力（固定出力）：
- `output/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics.csv`
- `output/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.json`

比較モデル（固定）：
- P-model：`local spacing d + nu_sat`（Step 7.13.18）
- 湯川距離proxy：`global R` を指数減衰の距離へ入れる baseline
- 標準理論proxy：SEMF（固定係数；fitなし）

固定出力：
- `output/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.png`
- `output/quantum/nuclear_binding_energy_frequency_mapping_theory_diff_metrics.json`
- `output/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.csv`

再現：
- `python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.py`

---

## 32.1 主要結果（metrics固定値）

| モデル | median(log10(B_pred/B_obs)) | z_median | z_Δmedian | median abs residual [MeV] | 判定 |
|---|---:|---:|---:|---:|---|
| P-model（local+d, nu_sat） | 0.0345 | 1.31 | 2.21 | 76.73 | pass |
| 湯川距離proxy（global R） | -1.1933 | -3.22 | -2.61 | 1072.50 | median fail |
| SEMF（固定係数） | 0.0025 | 1.51 | 1.19 | 6.69 | pass |

観測量レベルでは、P-model と SEMF はともに 3sigma運用の監査を通るが、
湯川距離proxy（global R）は中央値で fail する。
差分（P-model - standard）は重核側で最も大きく、次段の精度要件評価（Step 7.13.17.12）へ接続する。

---

## 32.2 注意

- 本Stepの「湯川/EFT」は、同一データI/Fで比較するための運用proxyである。
- ここでの目的は理論の最終優劣判定ではなく、差分信号がどこに集中するかを固定すること。
