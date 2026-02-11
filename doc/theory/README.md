# THEORY REPRO (offline)
ASCII preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

# Theory checks：再現手順

目的：観測データに依存しない形で、P-model の基本予言（光の偏向・重力赤方偏移のオーダー）を確認する。

---

## 1) 太陽近傍の光の偏向（Solar light deflection）

`python -B scripts/theory/solar_light_deflection.py --beta 1.0`

生成物（output/theory）
- `output/theory/solar_light_deflection.png`
- `output/theory/solar_light_deflection_metrics.json`

---

## 2) GPSの時間進み（重力＋速度の内訳）

`python -B scripts/theory/gps_time_dilation.py --delta 1e-60`

生成物（output/theory）
- `output/theory/gps_time_dilation.png`
- `output/theory/gps_time_dilation_metrics.json`

---

## 3) 速度項の飽和 δ（高γ観測との整合）

`python -B scripts/theory/delta_saturation_constraints.py`

入力（data/theory）
- `data/theory/delta_saturation_examples.json`（例：加速器/宇宙線/ニュートリノのエネルギースケール）

生成物（output/theory）
- `output/theory/delta_saturation_constraints.png`
- `output/theory/delta_saturation_constraints.csv`
- `output/theory/delta_saturation_constraints.json`

---

## 4) 重力赤方偏移（地上/衛星の一次ソース比較）

`python -B scripts/theory/gravitational_redshift_experiments.py`

入力（data/theory）
- `data/theory/gravitational_redshift_experiments.json`（一次ソースの公表値：偏差 ε と不確かさ）

生成物（output/theory）
- `output/theory/gravitational_redshift_experiments.png`
- `output/theory/gravitational_redshift_experiments.csv`
- `output/theory/gravitational_redshift_experiments.json`
