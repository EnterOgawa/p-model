# MERCURY REPRO (offline)
ASCII preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

# 水星（近日点移動）検証：再現手順

目的：簡易モデルで水星軌道を数値積分し、近日点の移動を可視化する。

---

## ワンコマンド再現（PowerShell）

`python -B scripts/mercury/mercury_precession_v3.py`

---

## 生成物（output/mercury）

- `output/mercury/mercury_orbit.png`

