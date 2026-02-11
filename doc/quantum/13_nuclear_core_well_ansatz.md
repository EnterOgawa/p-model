# 13. 原子核（Step 7.9.4）：core+well ansatz（最小拡張）の棄却例を固定する

目的：Step 7.9.3（square-well）の次として、ansatz class を最小拡張し、**任意関数fit**に逃げずに核スケール `u(r)=ln(P/P0)` の拘束を強める。

本章は「核力の第一原理導出」を主張しない。目的は次の 3 点である。

1) 追加自由度を **最小**にした ansatz class を定義する  
2) 一次データ（B, a, r, v2）でどこまで拘束できるかを固定出力化する  
3) 棄却された場合でも「次に何を足すべきか（最小自由度）」が明確になる形でログを残す

---

## 13.1 ansatz の定義（core+well）

Step 7.9.4 の最小拡張として、`u(r)` の短距離プロファイルを core+well で表す（現象論的 ansatz）。

- `V(r) = -V0`（`r<R`）の 2パラメータ（square-well）から出発し、
- 「core（短距離）＋well（中距離）」の構造を導入して、`r_t` や `v2` を調整できるかを検証する

ただし本リポジトリの方針として、**追加自由度は最小**に制限し、棄却条件を先に固定する。

---

## 13.2 入力（一次データ凍結）

Step 7.9.1–7.9.2 の一次固定を前提にする：

- deuteron：CODATA/NIST（質量欠損→束縛エネルギー B）
- np散乱：arXiv:0704.1024v1（eq18–19；GWU/SAID と Nijmegen）
  - `a_t,r_t,v2t / a_s,r_s,v2s`

一次ソースは `doc/PRIMARY_SOURCES.md` に集約する。

---

## 13.3 実装と固定出力

実装：
- `python -B scripts/quantum/nuclear_effective_potential_core_well.py`

固定出力：
- `output/quantum/nuclear_effective_potential_core_well.png`
- `output/quantum/nuclear_effective_potential_core_well_metrics.json`

方針：
- eq18/eq19 をそれぞれ別に処理し、**解析依存系統（proxy）**として差分を残す。
- triplet の拘束（B, a_t, r_t）を満たした上で、`v2t` と singlet 側の `(r_s,v2s)` を予言できるかを評価する。

---

## 13.4 結果（棄却）

結論：この core+well ansatz class では、

- triplet の `r_t` を観測包絡へ入れられず
- 予言 `v2t` が観測（eq18–eq19）より大きく
- singlet の `(r_s,v2s)` も観測と整合しない

ため、核スケール `u(r)` の候補としては **棄却**される（少なくとも本 ansatz の範囲では不十分）。

これは失敗ではなく、次に足すべき自由度が明確になったことを意味する。

---

## 13.5 次（Step 7.9.5）

次段階では、以下を優先する：

- finite repulsive core / 2-range などで ansatz class を拡張し、`v2` 系まで含めた拘束を可能にする
- 追加自由度は最小化し、**「1つ足すごとに、どの観測量が新しく説明できるようになったか」**を固定出力として残す

（ロードマップ：`doc/ROADMAP.md` の Step 7.9.5）

