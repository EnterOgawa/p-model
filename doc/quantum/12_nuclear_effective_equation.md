# 12. 原子核（pn束縛）の有効方程式：P場 `u=ln(P/P0)` から核スケール拘束を固定する（Step 7.9.3）

目的：原子核（例：deuteron）が「既存の量子力学/核模型へ重力ポテンシャル φ を入れただけ」に見える致命的攻撃を避けるため、Part I の基本変数 `u=ln(P/P0)`（すなわち `φ=-c^2 u`）から、核スケールで何を“有効方程式/有効ポテンシャル”として固定するかを明文化する。

本章は **第一原理（強い相互作用の微視的導出）を主張しない**。代わりに、

- 一次ソースで観測量を凍結（B, a_t, r_t, …）
- P-model 変数 `u` と観測量の間の **最小の写像**（effective equation）を固定
- “単純な ansatz class がどこで棄却されるか” を出力として残す

ことで、以後の拡張（より現実的な `u` プロファイル、singlet 併用、核種依存など）が恣意的にならない土台を作る。

---

## 12.1 最小の有効方程式（Part I → Part III の接続）

Part I の時計写像（固有時 τ）と `φ=-c^2 ln(P/P0)` を用いると、弱場では rest phase の位相蓄積から

- `V = m φ`

が自然に現れる（詳細は Part III の理論的枠組み参照）。

原子核は 2体（p–n）の相対運動として扱えるため、最小の effective equation として

- 相対座標 `r` に対する s-wave 非相対論シュレーディンガー方程式
- 有効ポテンシャル `V(r) = μ φ(r) = - μ c^2 u(r)`

を入口として採用する。

重要なのは、ここで “既存QMを採用した” と言い切るのではなく、**P-modelの時計写像→位相→V=mφ** という写像を介して、`u` の局所構造が束縛条件（固有値）を決める、という論理を固定する点である。

---

## 12.2 `u` プロファイルの現象論的 ansatz（短距離の核スケール）

現段階では、核スケールでの `u` の支配方程式（source/非線形/結合）が未提示である。したがって本章では、`u(r)` を **短距離の局所プロファイル**として現象論的に仮定し、観測量で拘束する。

最初の ansatz class として、2パラメータの **square-well** を採用する：

- `V(r) = -V0`（`r<R`）、`V(r)=0`（`r≥R`）
- これは `u(r)≈+V0/(μc^2)`（`r<R`）に対応し、`P/P0 = exp(u) > 1` を意味する

この ansatz は「形が単純で、拘束が明確」なため、棄却された場合でも次に何を足すべきかが分かりやすい。

---

## 12.3 一次データ拘束（凍結）

本章で拘束に使う一次ソースは以下に固定する：

- deuteron ベースライン（B, r_d）：NIST Cuu（CODATA）定数（`mp,mn,md,rd`）
  - 出力：`output/quantum/nuclear_binding_deuteron_metrics.json`
- np散乱（低エネルギー）：arXiv:0704.1024v1 の eq.(18)(19)（GWU/SAID と Nijmegen）
  - 出力：`output/quantum/nuclear_np_scattering_baseline_metrics.json`

一次ソースのURL/sha256は `doc/PRIMARY_SOURCES.md` に集約する。

---

## 12.4 結果（square-well の拘束と棄却条件）

実装：`python -B scripts/quantum/nuclear_effective_potential_square_well.py`  
出力：`output/quantum/nuclear_effective_potential_square_well.png` / `output/quantum/nuclear_effective_potential_square_well_metrics.json`

方針：

- 固定拘束：deuteron 束縛エネルギー `B`（CODATA）と triplet 散乱長 `a_t`（eq18/eq19）
- 予言：triplet 有効レンジ `r_t`

結果（初版）：

- eq18（GWU/SAID）を a_t として用いると、`R≈2.087 fm, V0≈34.1 MeV` を得る。  
  予言は `r_t(pred)≈1.758 fm` で、観測 `r_t(obs)=1.7494 fm` に対し `Δr_t≈+0.0087 fm`。
- eq19（Nijmegen）を a_t として用いると、`R≈2.118 fm, V0≈33.3 MeV` を得る。  
  予言は `r_t(pred)≈1.781 fm` で、観測 `r_t(obs)=1.753 fm` に対し `Δr_t≈+0.0277 fm`。

このため、**最も単純な 2パラメータ（square-well）ansatz は、eq18–eq19 の観測包絡に対しては一致せず**、核スケールの `u(r)` としては “粗すぎる” ことが分かる（棄却条件の初期例）。  
一方で、`u0=V0/(μc^2)≈7×10^-2` 程度が必要であり、`P/P0≈1.07` の増加が核力スケールの有効 `u` 構造として要求される、という **スケール拘束**を凍結できる。

---

## 12.5 次のステップ（冗長化せず、統一性を維持する）

square-well の棄却は “失敗” ではなく、次に足すべき最小自由度を明確にするための基準である。次の段階では以下を優先する：

- 形の改善：Step 7.9.4（core+well）と Step 7.9.5（finite core+well）の試行は棄却/退化まで固定済みである。次は **2-range（2段）**などで ansatz class を拡張し、**追加自由度を最小**に保ったまま triplet/singlet の同時拘束（`v2` 系を含む）を強化する（Step 7.9.6）。
- チャネル拡張：singlet（a_s,r_s）も同時に拘束し、任意関数fitを避ける。
- 既知精密物理との整合：核スケールで新たな短距離結合を入れる場合は、QED精密（Lamb/g-2等）からの上限拘束（safety check）を必須化する（Step 7.8.3 と整合）。

本章は「核力の最終導出」を主張する場所ではなく、**P-model 変数 `u` と核観測量を結ぶ“拘束の法典”を増分的に整備する場所**として運用する。
