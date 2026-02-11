# Step 7.8：真空・QED精密（Casimir 効果 / Lamb shift）

## 目的

Phase 7 の量子側は本稿の結論ではないが、「時間＝媒質」「物質＝波」という仮定が量子現象へ拡張可能かを検証可能にする。  
その際に避けて通れないのが、真空の境界条件に由来する現象（Casimir）と、束縛状態の精密分光（Lamb shift）である。

この Step 7.8 では、

- 一次ソースを固定（URL＋ローカル保存＋sha256）
- 観測量（何を比較するのか）を定義
- P-model の量子解釈が **最低限満たすべき要件**（満たせなければ棄却）を明文化

する。

---

## 1) Casimir 効果（境界条件→力）

理想導体の平行平板の Casimir 圧は

$$
P(a)=-\\frac{\\pi^2\\hbar c}{240\\,a^4}
$$

で与えられる。  
球–平板（AFM）では、近接力近似（PFA）で代表的に

$$
F(a)\\simeq -\\frac{\\pi^3\\hbar c\\,R}{360\\,a^3}
$$

が使われる（R は球半径）。現実の実験では有限導電率・粗さ・温度・距離校正などが系統として効く。

Roy, Lin, Mohideen（arXiv:quant-ph/9906062v3）は AFM による sphere–plate の Casimir 測定を報告し、
最小距離付近の代表精度を要約で述べている（一次ソースは `doc/PRIMARY_SOURCES.md`）。

### P-model 側の最低要件（棄却条件）

- 境界条件（反射）を扱う「波の量子化」に相当する構造を導入しない限り、Casimir の再現はできない。  
  したがって **Casimir 力のスケーリング（a^{-4} / a^{-3}）や符号を再現できない**場合、P-model の量子解釈は棄却。
- さらに、材料依存（導電率スペクトル）や幾何依存（corrugation 等）まで説明できるかは拡張課題であり、本 Step ではまず観測量の入口を固定する。

---

## 2) Lamb shift（束縛状態のQED補正）

Lamb shift は、Dirac 方程式のクーロン束縛の準位に対する QED 補正（自己エネルギー、真空偏極など）により生じる。

Ivanov & Karshenboim（arXiv:physics/0009069v1）は、軽い水素様イオンの Lamb shift を整理し、

- Lamb shift の代表スケーリング：Z^4
- 未計算の高次2ループ項のスケーリング：Z^6

を本文で述べる（一次ソースは `doc/PRIMARY_SOURCES.md`）。

また核サイズ（陽子半径等）に由来する非QEDの寄与が系統として入ることも、表で具体例を示している。

### P-model 側の最低要件（棄却条件）

- 原子スペクトルの精密差（Lamb shift）を **同一のパラメータ体系（Phase 1 の凍結値を破らず）**で説明できない場合、
  「量子現象を P-model で再解釈できる」という主張は棄却。
- 特に「核サイズ（物質分布）寄与」と「真空場/QED寄与」の識別が必要であり、ここを区別できないモデルは観測と突き合わせて反証される。

---

## 2b) H 1S–2S（絶対周波数：精密ベンチマーク）

束縛状態の精密分光として、Hydrogen 1S–2S の絶対周波数は代表的な基準である。  
Parthey et al. 2011（arXiv:1107.3101v1）は、不確かさ予算を含めて

$$
f_{1S-2S}=2466061413187035(10)\,\\mathrm{Hz}
$$

を報告している（一次ソースは `doc/PRIMARY_SOURCES.md`）。  
本リポジトリでは、同一の凍結値（Phase 1）下で、P-model の最小結合がこの精密量を直ちに破らないこと（safety check）を固定出力に含める。

---

## 2c) α（原子反跳 vs 電子 g-2）：独立精密クロスチェック

束縛分光以外の QED 精密量として、微細構造定数 `α` の独立決定は重要なクロスチェックである。

- 原子反跳（Rb；Bloch+atom interferometry）：Bouchendira et al. 2008（arXiv:0812.3139v1）
- 電子 g-2（QED）：Gabrielse et al. 2008（arXiv:0801.1134v2）

本リポジトリでは、両論文の一次ソースPDFから `α^{-1}` を機械抽出し、差分と z-score を `output/quantum/qed_vacuum_precision_metrics.json` に固定する（再現性のために値の手入力は避ける）。

### P-model 側の最低要件（棄却条件）

- 現段階では、Coulomb 定数や `α` の P 依存は主張しない（導入する場合は一次拘束が必須）。
- 将来、Pゆらぎが短距離で追加結合（非重力的）を持つモデルを導入する場合、`α` の独立決定と整合すること（差分が統計誤差で説明できること）が必須となる。

---

## 再現（本リポジトリ）

一次ソース取得：

```powershell
python -B scripts/quantum/fetch_qed_vacuum_precision_sources.py
```

観測量の入口（図＋メトリクス）：

```powershell
python -B scripts/quantum/qed_vacuum_precision.py
```

- 出力：`output/quantum/qed_vacuum_precision.png`
- 数値：`output/quantum/qed_vacuum_precision_metrics.json`
