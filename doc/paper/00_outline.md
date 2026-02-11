# P-model 論文アウトライン（メモ）

このファイルは Phase 8 / Step 8.2（本論文）の “骨格”。
まずは章立てを固定し、以後は各章に図表・数式・出典を差し込んでいく。

参照（図の一覧）：
- `doc/paper/01_figures_index.md`

参照（定義・導出）：
- `doc/P_model_handoff.md`
- `doc/theory/04_postulates_and_dictionary.md`
- `doc/theory/05_minimal_derivations.md`

参照（検証結果のカタログ）：
- `output/summary/pmodel_public_report.html`

---

## 0. 要旨（Abstract）

- 目的：相対性理論（弱場テスト領域）の置換候補として、Pモデル（時間波密度P）を提示する。
- 方法：Pをスカラー場として定義し、重力・時計・光伝播を一つの枠組み（位相/作用の停留）で記述する。
- 結果：LLR / Cassini / 太陽光偏向 / Viking / Mercury / GPS / EHT に加え、強重力・放射の代表例として二重パルサー（軌道減衰）と GW150914（chirp位相）を整理し、差分予測（EHT係数差・速度飽和δ）を提示する。
- 結論：一致する範囲と差が出る範囲を明示し、反証可能性を確保する。

---

## 1. 序論（Introduction）

### 1.1 背景
- GRの成功と、直感的理解・実装の難しさ（テンソル、座標系、実装コスト）
- Pモデルの狙い：スカラーで同じ実験事実を説明できるか（まず弱場テスト）

### 1.2 本研究の貢献
- 記号・符号規約の固定（P増→重力はP勾配へ滑る）
- 一次ソース（可能な範囲）での再現パイプライン化（`data/→scripts/→output/`）
- 公開レポート（図表の統一）と、差分予測（強場・宇宙論）

---

## 2. 理論（Theory）

### 2.1 公理・定義（必ず固定）
- `P` / `P0` / `φ` / `β` / `δ` / `τ` / `t`
- 重力：`φ = -c^2 ln(P/P0)`、`a = -∇φ`
- 光：`n(P)=(P/P0)^(2β)`（係数2はβが担う）
- 速度飽和：`dτ/dt = sqrt((1 - v^2/c^2 + δ)/(1+δ))`

### 2.2 弱場近似とニュートン極限
- `∇^2 φ = 4πGρ`、点質量 `φ=-GM/r`

### 2.3 観測量への写像（最低限）
- 赤方偏移（時計）
- Shapiro遅延（光）
- 光の偏向（光）
- PPN（特にγ）との対応（β↔(1+γ)/2）

### 2.4 動的P（放射・波動）【強場・放射へ拡張】
- 静的P（重力・時計・光）の枠組みを保ったまま、放射（波動）を扱う最小拡張を導入する。
- 連星放射（四重極）と重力波chirp（位相）の観測量（`Pdot_b` / `f(t)`）への写像を定義する。

---

## 3. 方法（Methods）

### 3.1 再現性（データとコード）
- データ：`data/<topic>/`
- スクリプト：`scripts/<topic>/`
- 生成物：`output/<topic>/`

### 3.2 指標
- RMS、相関、最大残差、系統性（残差 vs 仰角など）

### 3.3 再現コマンド（例）
- `python -B scripts/summary/run_all.py --offline`
- `python -B scripts/summary/public_dashboard.py`

---

## 4. 結果（Results）

### 4.1 Table 1（検証サマリ）
- 検証一覧（一次ソース/指標/主要数値）を Table 1 として固定する。

### 4.2 LLR（月レーザー測距）
- バッチ要約：`output/llr/batch/llr_batch_summary.json`

### 4.3 Cassini（太陽会合・ドップラー）
- 一次データ処理：`output/cassini/`（メタデータ含む）

### 4.4 太陽光偏向（観測γ vs P-model）
- 代表値（観測）と P-model を比較する。

### 4.5 Viking（太陽会合・Shapiro遅延）
- `output/viking/viking_shapiro_result.csv`

### 4.6 Mercury（近日点移動）
- `output/mercury/mercury_precession_metrics.json`

### 4.7 GPS（衛星時計）
- `output/gps/gps_compare_metrics.json`

### 4.8 EHT（ブラックホール影）
- `output/eht/eht_shadow_compare.json`

### 4.9 重力赤方偏移（GP-A / Galileo）
- 一次ソース起点で観測値と P-model を比較する。

### 4.10 二重パルサー（軌道減衰：放射の制約）
- 観測が支持する四重極放射と整合するかを確認し、追加放射（双極など）の上限を見積もる。

### 4.11 重力波（GW150914：chirp位相の整合性）
- 公開 strain から抽出した `f(t)` が四重極チャープ則（Newton近似）と整合するかを確認する。

---

## 5. 差分予測（Differential Predictions / Falsifiability）

### 5.1 EHT：影直径係数の差
- `output/eht/eht_shadow_differential.png`

### 5.2 速度飽和δ：γ上限（SRとの差）
- `output/theory/delta_saturation_constraints.png`

---

## 6. 議論（Discussion）

- 一致している範囲（弱場テスト）と、差が出る範囲（Phase 4）を明確に区切る
- 既存理論（GR/PPN）との対応関係（β、δの役割）
- 制限事項（観測の定義、処理手順、系統誤差）

---

## 7. 結論（Conclusion）

- 何が成立し、何が差分予測になっているかを簡潔にまとめる

---

## 付録（Appendix）

- 記号表（確定版の再掲）
- 追加導出（必要に応じて）
