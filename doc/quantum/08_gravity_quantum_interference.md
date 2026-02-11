# Step 7.5（抜粋）：重力×量子干渉（COW / 原子干渉計 / 量子時計）

## 目的

Phase 7 では「時間＝波の媒質」という仮定を量子側へ接続する。
この Step 7.5 では、**干渉位相**（物質波・原子干渉計・量子時計）が重力（時間構造）とどう結びつくかを、
一次ソースに基づき再現可能な形へ落とす。

ここではまず、代表的な2系（COW と原子干渉計重力計）を入口として固定する。

---

## 1) COW 実験（重力誘起量子干渉）

Mannheim（arXiv:gr-qc/9611037）は、COW（Colella-Overhauser-Werner）実験の位相差を
「古典的時間遅延（重力場での運動の違い）」として整理し、典型パラメータも与える。

平方形干渉計（辺 H）、入射速度 v0 のとき、最低次で位相差は

$$
Δφ ≃ - m g H^2/(ħ v0)
$$

（重力成分の取り方により、幾何・姿勢に依存して比例因子が入る）。

### 再現（本リポジトリ）

```powershell
python -B scripts/quantum/cow_experiment_phase.py
```

- 出力：`output/quantum/cow_phase_shift.png`
- 数値：`output/quantum/cow_phase_shift_metrics.json`

---

## 2) 原子干渉計重力計（light-pulse Mach–Zehnder）

Mueller et al.（arXiv:0710.3768）は、原子干渉計での重力位相差（重力計の基本式）として

$$
φ_g ≃ k_eff g T^2
$$

（laser phase を用いた readout を含めると式全体は `k_eff g T^2 - φ_L` の形）を与える。

### 再現（本リポジトリ）

```powershell
python -B scripts/quantum/atom_interferometer_gravimeter_phase.py
```

- 出力：`output/quantum/atom_interferometer_gravimeter_phase.png`
- 数値：`output/quantum/atom_interferometer_gravimeter_phase_metrics.json`

### β（光伝播）由来の差分予測（Step 7.5.8）

Mueller 2007 は `k_eff` を「光の分散（レーザ波数）」から決めており、P-model の光伝播自由度 β（`n(P)=(P/P0)^(2β)`）は **レーザー位相項**を通じて位相へ入り得る。  
一方で、局所実験では定数項（共通モード）の相殺/吸収により差分は強く抑圧されるため、本リポジトリでは A/B（上限/局所差分）モデルとして整理し、採用モデル（局所差分）の Δφ_beta と 3σ 必要精度を固定した。

- 固定（式＋モデル＋必要精度）：`output/quantum/atom_interferometer_gravimeter_phase_metrics.json`（`results.beta_phase_dependence`）
- 固定（反証カタログ）：`output/summary/quantum_falsification.json`（criteria: `atom_beta_phase_delta_est`）

---

## 3) 量子時計（光格子時計）：chronometric leveling

光格子時計（例：87Sr）の重力赤方偏移（周波数比）は、静止時計の近似で

$$
Δf/f ≃ ΔU/c^2
$$

（ΔU はジオポテンシャル差）と書ける。公開一次ソース（arXiv）の要旨に数値が明示されている例として、arXiv:2309.14953v3 は、457 km 離れた2地点のジオポテンシャル差を時計比較から測り、独立な測地測量と整合することを報告している。

### 再現（本リポジトリ）

```powershell
python -B scripts/quantum/optical_clock_chronometric_leveling.py
```

- 出力：`output/quantum/optical_clock_chronometric_leveling.png`
- 数値：`output/quantum/optical_clock_chronometric_leveling_metrics.json`

---

## 4) 物質波干渉／二重スリット（電子）

Bach et al.（arXiv:1210.6243v1）は、電子二重スリットを「可逆に片方だけ閉じる」マスクと組み合わせ、
P1（片スリット）・P2（片スリット）・P12（両スリット）の分布を制御して観測した。

遠方場（Fraunhofer）の最小モデルでは、スリット幅 a、スリット間隔 d、de Broglie 波長 λ のとき、

$$
I(θ) ∝ \\mathrm{sinc}^2\\left(\\frac{π a\\sinθ}{λ}\\right)\\cos^2\\left(\\frac{π d\\sinθ}{λ}\\right)
$$

となり、一般に **P12 は P1+P2 と一致しない**（干渉項がある）ことが直接出る。

### 再現（本リポジトリ）

```powershell
python -B scripts/quantum/electron_double_slit_interference.py
```

- 出力：`output/quantum/electron_double_slit_interference.png`
- 数値：`output/quantum/electron_double_slit_interference_metrics.json`

---

## 5) de Broglie 波長の精密検証（原子反跳→α）

Bouchendira et al.（arXiv:0812.3139v1）は、原子へ多数の光子運動量を与える Bloch oscillations と
原子干渉計を用いて反跳速度（ħk/m）を精密測定し、微細構造定数 α を導出した。

この種の測定は「運動量 p と波数 k の対応（p=ħk）」と「α と h/m の関係」を組み合わせるため、
de Broglie（λ=h/p）を含む精密な一貫性チェックとして使える。

ここでは代表として、原子反跳（Bouchendira）と電子 g-2（Gabrielse）の α の整合を確認する。

### 再現（本リポジトリ）

```powershell
python -B scripts/quantum/de_broglie_precision_alpha_consistency.py
```

- 出力：`output/quantum/de_broglie_precision_alpha_consistency.png`
- 数値：`output/quantum/de_broglie_precision_alpha_consistency_metrics.json`

---

## 位置づけ（P-model 側の読み替え）

P-model では、弱場での重力はポテンシャル φ（またはその勾配 a=-∇φ）により記述されるため、
上記の式に入る g は「局所の重力加速度」として同定される。
この Step の目的は、まず観測量（干渉位相）の定義・スケーリングを一次ソースで固定し、
次に「P の時間構造がどこへ入るか（入らないか）」を反証可能な形で整理することにある。
