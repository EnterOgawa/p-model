# VIKING REPRO (cache + offline replay)
ASCII preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

Offline replay:
- First run downloads Horizons vectors and saves them to output/viking/horizons_cache/.
- Later runs auto-use cache (no network). To force offline, use --offline or set HORIZONS_OFFLINE=1.

# Viking（太陽会合）検証：再現手順

目的：Viking（地球-火星）往復通信の Shapiro 遅延（~250 μs）を、HORIZONS幾何から計算してCSV/図を生成する。

補足：この検証は現時点では「観測の完全な時系列データ」を取り込んでいない。
図中の赤点は、文献でよく引用されるピーク遅延（約250 μs）の**代表値**を示す。

---

## 入力

- 幾何（地球/火星ベクトル）：HORIZONS APIから取得（ネットワーク必要）
- スライド雛形：`doc/slides/時間波密度Pで捉える「重力・時間・光」.pptx`

---

## ワンコマンド再現（PowerShell）

`python -B scripts/viking/viking_shapiro_check.py --beta 1.0; python -B scripts/viking/update_slides.py`

補足：`viking_shapiro_check.py` は `--start/--stop/--step` で期間を変えられる。`--beta` を指定すると内部で `gamma = 2*beta - 1` を使う。

---

## 生成物（output/viking）

- `output/viking/viking_shapiro_result.csv`
- `output/viking/viking_p_model_vs_measured_no_arrow.png`
- `output/viking/P_model_Verification_Full.pptx`
