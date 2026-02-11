# 14. 原子核（Step 7.9.5）：finite core+well ansatz（最小拡張）の試行と退化を固定する

## 14.1 目的（なぜ必要か）

Step 7.9.3（square-well）と Step 7.9.4（core+well）では、低エネルギー np 散乱パラメータのうち特に shape（v2）と singlet の再現が破綻し、核スケールの `u(r)=ln(P/P0)` を表す ansatz class としては棄却された。

Step 7.9.5 では、**追加自由度を最小**に保ったまま（任意関数 fit を避けるため）、
- repulsive core を hard-core（∞）ではなく有限値 **Vc** として導入し、
- triplet 側の残余自由度（任意性）を `v2t` の目標で潰し、
- singlet は `a_s` のみを合わせて `(r_s, v2s)` を予言

することで、「core が入っても `v2` が救えるか」を固定出力として判定する。

## 14.2 ansatz の定義（finite core + well）

相対運動の s-wave に対し、有効ポテンシャルを
- `V(r)=+Vc`（`0<=r<Rc`）
- `V(r)=-V0`（`Rc<=r<R`）
- `V(r)=0`（`r>=R`）

の 3 領域の piecewise-constant で近似する（現象論的 ansatz）。

Part III の最小写像 `V(r)=μ φ(r)=-μ c^2 u(r)` により、`u(r)` の短距離拘束へ戻す。

## 14.3 再現コマンドと固定出力

- 実行：
  - `python -B scripts/quantum/nuclear_effective_potential_finite_core_well.py`
- 出力：
  - `output/quantum/nuclear_effective_potential_finite_core_well.png`
  - `output/quantum/nuclear_effective_potential_finite_core_well_metrics.json`

## 14.4 結果（要点）

- fit（triplet）：B（deuteron）, a_t を満たす解は得られるが、`v2t`（特に eq19）の整合が得られず、最適解は **`Rc→0, Vc→0` に潰れて square-well 相当へ退化**する。
- predict（singlet）：`a_s` のみを合わせても、`r_s` と `v2s` の予言は観測の包絡（eq18–eq19）に入らない。

結論：finite core+well ansatz class は、核スケール `u(r)` の候補としては棄却される（少なくとも本リポジトリの「最小自由度」条件下では救えない）。

## 14.5 次（Step 7.9.6）

次段階では、**2-range（2段）**などの最小拡張で、triplet/singlet を同時拘束した上で `v2` 系まで反証条件を固定する必要がある（`doc/ROADMAP.md` の Step 7.9.6）。

