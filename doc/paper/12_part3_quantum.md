# P-model Part III（量子物理編）：応用検証

本稿は P-model の応用検証（宇宙/量子）であり、記号規約最小仮定写像は [Part I（コア理論）](pmodel_paper.html) に従う。

---

## 要旨（Abstract）

本稿（Part III）は P-model の応用検証（量子物理編）である。  
Part I の写像（P→φ→重力/時計/光）と凍結パラメータ（β）を前提に、
公開一次データを起点として量子テストを「再現可能・反証可能」な形へ整備する。
本稿は量子力学の再導出（振幅則・非局所性の否定など）を主張しない。  
代わりに、どの仮定が観測手続きで破れ得るか、そしてどこで棄却されるか（反証条件）を、
固定出力として提示することを目的とする。
また、Part I の動的Pの最小仮定は、短波長・非相対論極限で Schrödinger 方程式
（有効ポテンシャル `V=mφ`）と整合することを 2.5.1 で明示し、
標準量子現象との互換性の最低条件を固定する。  
（English）We show that the P-field fluctuation equation reduces to a Schrödinger-type equation
in the non-relativistic limit (Sec. 2.5.1), providing a minimal consistency check with standard quantum phenomenology.

検証の要点は Table 1 に集約し、各テーマについて入力（一次ソース）→処理（固定条件）→
指標→図→注意（系統）の順で要約する。特に Bell では time-tag / trial log の公開一次データを用い、
selection（coincidence window / trial 定義 / event-ready window）が統計母集団を動かし得る入口を
定量化し、共分散（統計）と系統（手続き依存）を分離した上で反証条件パックへ接続する。

---

## 1. 序論（Introduction）

量子現象に対して P-model が取り得る立場は「統一の主張」ではなく「反証可能な入口の提示」である。  
P-model は時間を波として仮定し、粒子を波の境界条件（反射）として、
相関を局所P構造の共有として再解釈し得る。  
しかし本稿は、この枠組みが量子力学の全予測を再現できるとは主張しない。

入口として 2.5.1 で、Part I の動的Pの最小仮定が短波長・非相対論極限で Schrödinger 形へ還元されることを明示し、
量子側の最小整合条件を先に固定する。

本稿が扱うのは、公開一次データに基づき「どの仮定が、どの解析手続きで破れ得るか」を固定し、
差分予測（反証条件）へ落とすことである。宇宙編（Part II）と同様に、Table 1 を入口にして
各検証の入力・処理条件・指標を機械的に追跡できる体裁を採り、
将来のデータ追加（計画テスト）も同じI/Fで吸収できるようにする。

本三部作は **Part I → Part II → Part III** の順で読むことを想定する。  
Part I で写像・凍結・反証形式を確認し、Part II で古典〜宇宙スケールの応用検証を同じ枠組みで積み上げ、
最後に本稿（Part III）で量子領域を同型のI/Fで扱うことで、P-model の統一性を「主張」ではなく
「反証可能性の一貫性」で評価できるようにする。

表現規約：本文は P-model の定義写像手順のみで閉じる。  
参照枠（既存理論）の比較は節末の Reference note に隔離し、
本文中で定義や導出を参照枠で置き換えない。

Reference note: 本稿は、既存の量子力学（QM）に隠れた変数を追加して解釈し直す試みではない。  
Part I で固定した P-model の最小方程式と写像から出発し、その波動的性質がマクロな観測として
どのように量子現象（干渉・相関）を再現するかを検証する。量子力学の標準形式は、
導出された近似式・指標の妥当性を確認するための現象論的ベンチマークとして位置づける。

---

## 2. 理論的枠組み（Theoretical Framework）

本稿で用いる記号規約・最小仮定・写像は Part I に従う。  
理論式の再掲は最小限とし、ここでは「独立に読めるための最低限」だけを要約する。

### 2.1 P 場とポテンシャル φ

時間波密度を `P(x)`、遠方基準を `P0` とし、**質量近傍で `P>P0`** を採用する。  
観測量に現れるのは比 `P/P0`（無次元）のみであり、`P` の絶対値や次元の同定は本稿の範囲外とする。  
ただし `ln(P/P0)` を定義するため `P` と `P0` は同次元である。  
重力ポテンシャルは

$$ \phi \equiv -c^2 \ln\left(\frac{P}{P_0}\right) $$

で定義する（無限遠で0、質量近傍で負）。

### 2.2 重力（運動方程式）

重力加速度は

$$ \mathbf{a} = -\nabla\phi = c^2\,\nabla\ln\left(\frac{P}{P_0}\right) $$

と統一し、「P勾配へ滑り落ちる」を式に固定する。

### 2.3 時計（重力項 + 速度項）

時計の進みは

$$ \frac{d\tau}{dt} = \frac{P_0}{P}\left(\frac{d\tau}{dt}\right)_v $$

と分解し、速度項は標準回収として

$$ \left(\frac{d\tau}{dt}\right)_v = \sqrt{1-\frac{v^2}{c^2}} $$

と置く。

Reference note: 本節の速度項は参照枠における SR（特殊相対論）の時間遅れと同型である。

### 2.4 光（屈折：高P側へ曲がる）

光は屈折率を

$$ n(P) = \left(\frac{P}{P_0}\right)^{2\beta} $$

と置き、フェルマーの原理に従って **高P側へ屈折**するとする。  
`β` は Part I の凍結値（`β≈1`）として扱い、Part II/III の本文では `β` の値を個別に変更せず、更新が必要な場合は Part I の改訂に集約する。

Reference note: 参照枠として PPN を用いる場合、光の係数は `1+γ=2β` の対応で比較できる（係数2は β 側で担う）。

### 2.5 量子（局所P構造・相関・selection）

Part I の中心は重力・時計・光伝播であるが、P-model の出発点
「全ての物質は波であり、その媒質は時間である」は量子現象（干渉・相関）とも接続し得る。  
ここでは量子力学の再導出を主張せず、**最小の立場と検証入口**
（何を仮定し、どこが反証点になるか）を理論側に明記する。

#### 2.5.1 仮定列：P→位相→Schr/KG（何を仮定し何を導出したか）

Part I は動的Pの最小仮定として `u(x,t)≡ln(P/P0)` とその動力学（弱場で波動方程式、真空では `□u=0`）を採用した。  
Part III はここから量子力学（確率振幅則）を厳密導出したとは主張しないが、少なくとも「粒子＝束縛波モード」という立場が、短波長極限で Schr/KG と整合するための **仮定と帰結**を固定する。

**要約（仮定→導出；本節で固定）**：

```
仮定1: u(x,t)=ln(P/P0) と最小動力学（真空では □u=0）
仮定2: 静止時計 dτ/dt = P0/P（φ ≡ -c^2 ln(P/P0)）
仮定3: 束縛モード位相は固有時τに比例（rest phase）
仮定4: E=ħω と m≡ħω0/c^2
仮定5: 非相対論包絡近似（高速位相因子×遅い包絡）
導出: iħ ∂t ψ = [-(ħ^2/2m)∇^2 + mφ] ψ
```

**仮定（固定）**：
- 仮定1（Part I 最小動力学）：`u(x,t)=ln(P/P0)` は弱場で波動方程式を満たす（真空では `□u=0`）。
- 仮定2（時計写像の定義）：静止時計は

  $$
  \left(\frac{d\tau}{dt}\right)_{v=0}=\frac{P_0}{P}=e^{-u}=e^{\phi/c^2}
  $$

  と定義し、ポテンシャルは

  $$
  \phi\equiv -c^2\ln\left(\frac{P}{P_0}\right)=-c^2 u
  $$

  を採用する。
- 仮定3（内部位相時計）：束縛モードの位相は固有時 τ に比例して蓄積する（rest phase）：

  $$
  \psi \propto e^{-i\omega_0 \tau}
  $$

- 仮定4（ħと質量の導入）：ħ を位相周波数とエネルギーの変換係数として定義する（`E=ħω`）。さらに

  $$
  m \equiv \frac{\hbar \omega_0}{c^2}
  $$

  でモードの “質量” を定義する。
- 仮定5（非相対論包絡近似）：高速位相因子と遅い包絡を分離し、包絡が “十分に遅い” という近似を入れる（典型：KG型分散→自由 Schr）。例えば、

  $$
  \Psi(x,t)=\psi(x,t)\,e^{-i\omega_0 t}
  $$

  および

  $$
  \left|\partial_t^2\psi\right|\ll \omega_0\left|\partial_t\psi\right|
  $$

  を仮定する。

**導出（2.5.1で固定）**：
- 導出1（ポテンシャル項）：仮定2より、弱場で

  $$
  \frac{d\tau}{dt}=\frac{P_0}{P}=e^{\phi/c^2}\approx 1+\frac{\phi}{c^2}
  $$

  となる。したがって（φ が十分ゆっくり変化する近似の下で）rest phase は

  $$
  e^{-i\omega_0 \tau}
  \approx e^{-i\omega_0 t}\,e^{-i\omega_0 \phi t/c^2}
  $$

  と分解できる。仮定4（`m=ħω0/c^2`）により、第2因子は `V=mφ` に対応する位相である。
- 導出2（Schr 形）：したがって非相対論極限の有効方程式として

  $$
  i\hbar\frac{\partial \psi}{\partial t}
  =
  \left(-\frac{\hbar^2}{2m}\nabla^2+m\phi\right)\psi
  $$

  が自然に現れる。
- 導出3（KG 形の採用）：相対論領域の最小候補として KG 型

  $$
  \left(
  -\frac{1}{c^2}\frac{\partial^2}{\partial t^2}
  +\nabla^2
  -\frac{m^2 c^2}{\hbar^2}
  \right)\Psi=0
  $$

  を置き、短波長極限（WKB）で位相がハミルトン・ヤコビ方程式に従うことを要請する（“一致する最低条件”）。

**未導出（本稿では主張しない）**：
- 未導出1：ψ（複素振幅）が P のどの自由度（位相/統計量）に一意に対応するか。
- 未導出2：Born則（確率密度 |ψ|^2）の第一原理導出。
- 未導出3：測定（状態更新）の第一原理導出（装置の不可逆性・増幅の微視的記述）。
- 未導出4：スピン/電荷/ゲージ場/相互作用項（電磁気以降は別節で最小導入）。

**P-model固有補正の観測可能性（入口）**：
- Schr 方程式の “形” は標準と同型だが、ポテンシャル φ の定義が P-model 固有であるため、P が強場で標準重力ポテンシャルとズレる場合に差分予測が生じ得る（弱場では同値）。
- 実験としては、物質位相だけでなく **光（レーザ位相・読み出し）**が入る原子干渉計で、光伝播（β）由来の位相項が差分予測の入口になる。

#### 2.5.2 Born則：採用範囲と逸脱条件（P-modelの立場）

本稿は Born則を第一原理から導出したとは主張せず、Phase 7 の現段階では
**操作的（operational）な採用**として固定する。  
その代わり、採用範囲と、どの仮定が崩れると “Born則（として使う近似）が破れるように見えるか” を
明文化する。

**採用（固定）**：
- 位置測定などの検出確率密度は

  $$
  p(x)\propto |\psi(x)|^2
  $$

  として扱う。
- 連続的な検出（クリック率）の最小モデルとして、局所検出率を

  $$
  \lambda(x,t)=\kappa(x,t)\,|\psi(x,t)|^2
  $$

  と近似する（κ は装置側の応答・閾値・利得を粗視化した係数）。

**採用範囲（成立を期待する条件）**：
- 装置が線形応答（κ が ψ の強度・位相や過去履歴に強く依存しない）。
- マルコフ近似（装置・環境の相関時間が系の時間スケールより十分短い）。
- backreaction 無視（1試行中に、検出に伴うエネルギー流が u（P）場そのものを有意に変えない）。
- 弱場（|φ|/c^2 ≪ 1）であり、時計写像（指数非線形）の高次項が実験精度に比べて十分小さい。

**逸脱し得るレジーム（追加仕様が必要な領域）**：
- マクロ重ね合わせ：系が巨大化し、(i) backreaction 無視、(ii) マルコフ近似、(iii) 線形応答、のいずれかが破れ得る（Born則の “採用形” を変える必要が出る）。
- 強重力場：|φ|/c^2 が無視できない領域では、u の非線形・時空依存・揺らぎが顕在化し、検出率の係数 κ が u に依存する可能性がある（現段階では未定義）。
- 自己重力（平均場）を導入する拡張：もし **量子質量密度を u のソースへ入れる**（例：ρ→m|ψ|^2）なら、Schr は Schrödinger–Newton 型（非線形）になり、干渉の可視度・位相・重ね合わせの安定性が変化する。これは Part I のコアからの追加仮定であり、採用する場合は “別仮説” として反証条件を固定すべきである（宇宙ベースのマクロ干渉計計画（例：MAQRO のようなコンセプト）が決着点になり得る）。

**観測可能性（定量の形）**：
- Born則の “採用形” の微小逸脱を p→p(1+δ) と表すと、単純なカウント統計では

  $$
  |\delta|\gtrsim \frac{3}{\sqrt{N}}
  $$

 （N は有効試行数）で 3σ 検出になる。
- κ が u（または ∇u）に一次で応答すると仮定して δ≈ε_u|u| または δ≈ε_g ℓ|∇u| と置けば、

  $$
  u=-\phi/c^2,\qquad |\nabla u|=|\nabla\phi|/c^2
  $$

  なので、実験室の重力場（|∇φ|≈g）では |u|~10^-9、|∇u|~10^-16 m^-1 と極小であり、Born則が破れるほど大きい効果を想定するには **別の増幅機構（装置非線形・自己重力・非マルコフ性など）**が必要になる。
- マクロ干渉で自己重力（平均場）が効く目安は、波束サイズ R と干渉時間 T で

  $$
  \frac{E_G}{\hbar}T \sim \frac{Gm^2}{\hbar R}\,T \gtrsim 1
  $$

 （E_G∼Gm^2/R を重力自己エネルギーのスケールとした粗い指標）であり、どの質量・サイズ・時間で “追加効果が見えるか” を決めるのはこの種の無次元量である。例えば

  $$
  m_{\rm crit} \sim \sqrt{\frac{\hbar R}{G T}}
  $$

  は “重力自己効果が位相へ 1 rad 程度入る” 目安であり、R=100 nm、T=1 s とすれば m_crit は 4×10^-16 kg（約 2×10^11 amu）程度になる。

**実験接続（計画実験への予測）**：
m_crit の具体例（m_crit≡sqrt(ħ R/(G T))；概算）：

| R（波束サイズ） | T（干渉時間） | m_crit | 相当する質量スケール |
|---:|---:|---:|---|
| 1 nm | 1 ms | 1.3×10^-15 kg（7.6×10^11 amu） | サブμm〜μm級ナノ粒子干渉の上限スケール |
| 100 nm | 1 s | 4.0×10^-16 kg（2.4×10^11 amu） | μm級ナノ粒子（10^11 amu級） |
| 1 μm | 10 s | 4.0×10^-16 kg（2.4×10^11 amu） | （R/T 同一） |
| 10 μm | 100 s | 4.0×10^-16 kg（2.4×10^11 amu） | （R/T 同一） |

MAQRO のような宇宙干渉計コンセプトは、長い T と大きい R を同時に取り得るため、
上のような m_crit スケールに相対的に近づきやすい（地上より環境デコヒーレンスを抑えられる）。
もし量子質量密度が u のソースへ入る（Schrödinger–Newton 型の平均場）という追加仮説を採用するなら、
m∼m_crit 近傍で位相のずれや可視度低下（見かけ上の Born則逸脱として現れる）を予測し得る。

一方、OTIMA などの分子干渉（〜10^6–10^8 amu級）では m ≪ m_crit であり、
本稿の立場では Born則逸脱は予測されない。
光格子時計の高度差比較も弱場で ε≈0 を予測し、現状の観測と整合している（4.2.2）。

Born則逸脱の直接検証は、現在の地上実験では困難であり、
将来の宇宙実験（MAQROのようなコンセプト）が決着点になり得る。

#### 2.5.3 測定（状態更新）の位置づけ：Lüders採用・デコヒーレンス・Penrose–Diósi

本稿は「波束の収縮」を物理的事実として断言せず、測定結果 `m` をマクロ記録とみなし、記録で条件付けた有効状態として状態更新（conditioning）を扱う。その最小形として、射影測定（Lüders型）と一般測定（POVM/Kraus）を **採用**し、採用理由と未解決点を明記する。

**採用（固定）**：
- 射影測定（Lüders）：

  $$
  p(m)=\mathrm{Tr}(\Pi_m\rho),\qquad
  \rho\to\rho_m=\frac{\Pi_m\rho\Pi_m}{\mathrm{Tr}(\Pi_m\rho)}
  $$

- 一般測定（Kraus）：

  $$
  p(m)=\mathrm{Tr}(E_m\rho),\qquad
  \rho\to\rho_m=\frac{M_m\rho M_m^\dagger}{\mathrm{Tr}(E_m\rho)},\qquad
  E_m=M_m^\dagger M_m,\qquad \sum_m E_m=I
  $$

**採用理由（なぜ Lüders を置くか）**：
- 最小の更新則であり、(i) 記録による条件付け、(ii) 直後の再測定の再現性（同じ記録が安定に繰り返される）、(iii) オープン系の標準形式（量子軌道・Lindbladの unraveling）と整合する。
- P-model の語彙では `Π_m` や `M_m` は “恣意的公理” ではなく、装置・環境の多数自由度を粗視化したときに残る **マクロ記録の同値類（pointer states）**として解釈する（更新は collapse 断言ではなく条件付き更新）。

**デコヒーレンス機構（P-modelでの言語化）**：
- 測定は「対象の局所P構造（モード）×装置の巨大P構造（境界条件＋散逸＋増幅）」の干渉であり、粗視化後の有効記述として pointer 基底の非対角成分が減衰する（decoherence）。
- ただし、`κ_m` や `Π_m` を P の微視モデルから導出したわけではない（現段階では有効モデルとして採用）。  
  図は次で固定する（詳細な議論は補足資料へ分離する）。

`output/quantum/quantum_measurement_born_rule_flow.png`  
Born則（|ψ|^2）と状態更新（条件付き更新）の位置づけを示す。

**Penrose–Diósi 型との関係**：
- 本稿の立場は objective collapse（重力が自発的に波束を潰す）を採用しない。したがって Penrose–Diósi 型は **別仮説**であり、同一視しない。
- ただし将来、P の揺らぎや backreaction を起点に objective collapse を導入する場合は、Penrose–Diósi/CSL 等の既存制約と同等に一次データで拘束し、採用範囲・反証条件を別途固定する必要がある（本稿では未定義）。

**反証入口：selection（採用規則）と統計母集団**：
- Bell では、観測手続き（time-tag の coincidence window、trial 定義、event-ready window）が **採用される事象集合（統計母集団）**を変え得る。これを選別重み `w_ab(λ)`（設定 `a,b` と潜在変数 `λ` に依存）として表すと、

  $$
  P_{\rm obs}(x,y\mid a,b)
  =\frac{\int d\lambda\,\rho(\lambda)\,w_{ab}(\lambda)\,P(x\mid a,\lambda)P(y\mid b,\lambda)}
  {\int d\lambda\,\rho(\lambda)\,w_{ab}(\lambda)}
  $$

  となる。ここでの `w_ab(λ)` は解析手続きが採用する事象集合を表す記法であり、P-model の力学として新自由度を導入する主張ではない。`w_ab` が設定に依存しない（同一母集団）という前提が崩れる場合、標準的な CHSH 導出の適用条件が満たされない。  
  この「どの仮定が、どの解析手続きで破れ得るか」を一次公開データで検証し、反証可能性へ接続する。


### 2.6 電磁気の位置づけ

電磁気は、原子・分子・物性へ進むための必須構成要素である。本稿は Maxwell/QED を P-model（P場）から第一原理導出したとは主張しないが、攻撃耐性のため **「採用／未導出／整合性チェック」**を明示的に固定する。

#### 2.6.1 採用（公理として）

- Maxwell 方程式（U(1) ゲージ場 `A_μ`、電場 `E`、磁場 `B`）を、P-model の枠組みとは独立に採用する（局所固有時での標準形を保持）。
- Coulomb 定数と微細構造定数 α は、現段階では **P 非依存**と仮定する（導出を主張しない）。
- 重力場中での “座標時間で見た光の伝播” は、Part I の写像（屈折率）`n(P)=(P/P0)^(2β)` で扱う（Maxwell の背景場上の導出は本稿では未実施）。

#### 2.6.2 未導出（将来課題）

- U(1) ゲージ対称性の P-model 内での起源（位相・内部対称性・トポロジー等）。
- 電荷の量子化の機構（なぜ離散化するか）。
- 電磁場のエネルギー運動量テンソルと P の backreaction の整合的な導入（現段階では phenomenological）。

#### 2.6.3 整合性チェック（本稿の射程）

- α の P 依存性の観測上限：仮に

  α(P/P0) = α0 (1 + κ ln(P/P0) + …)

  の形で導入するなら、原子時計と宇宙論的制約から **|κ| の上限**が与えられる。制約は台帳として固定し、現段階では κ=0（P 非依存）を採用する。
- 異なる φ 環境でのスペクトル整合性：H 1S–2S などの基準値に対して、仮想的な κ≠0 が与える寄与が観測精度以下となる範囲も同じ台帳に併記する。

`output/quantum/electromagnetism_minimal.png`  
Coulomb と重力のスケール差を並べ、最小位置づけを確認する。  
本節は導出ではなく、採用した近似の射程（何を固定するか）を示す。

`output/quantum/alpha_p_dependence_constraint.png`  
α(P) の κ 制約（上限）と、κ=0 採用の根拠（許容領域）を示す。  
以降の原子・分子ターゲットが満たすべき整合条件の入口として使う。


---

## 3. データと方法（Data / Methods）

本稿は一次ソース（公開一次データ／公式配布物）を起点に、取得データの保存と処理条件（窓・外れ値規則など）と出力ファイル名を固定する。一次ソース（URL/参照日）は付録「データ出典（一次ソース）」に集約する。  
これにより、解析対象の更新や選択が恣意的にならないよう、入力と処理条件（窓、閾値、外れ値規則、pairing等）を明示し、同じ入力から同じ出力が再現できる状態を優先する。

指標は Table 1 に集約し、統計不確かさ（共分散）と系統（手続き依存）を分離して扱う。特に Bell では selection ノブ（coincidence window / trial 定義 / event-ready offset 等）に対する感度を系統として固定し、bootstrap 等で統計を評価して反証条件パックに落とす。  
重要なのは、Part I の凍結値（β）を後から動かして整合させるのではなく、固定した写像の下で「どこで決着するか」を出力として残すことである。

### 3.0 共通棄却手順（Rejection Protocol）

本稿の各検証は、以下の共通書式で棄却可能性を固定する。  
- **Input**：使用データ（一次/推定済み）、依存前提、取得元（参照日）。  
- **Frozen**：凍結パラメータと凍結根拠（どの一次拘束でいつ固定したか）。  
- **Statistic**：評価統計（例：RMS, χ², logL, ΔAIC, 傾き）。  
- **Reject**：棄却閾値（例：3σ超、ΔAIC>10、系統誤差込みの超過）。  
以後、各テストは必ずこの書式に従って記述し、議論ではなくプロトコルとして再現棄却可能にする。

Frozen の一覧（凍結値＋参照元sha256）は、8章「データ/コードの入手性」に示す凍結パラメータ一覧（JSON）を正とする。  

---

## 4. 結果（Results）

### 4.1 Table 1（検証サマリ）

Table 1 は「固定した処理条件で得た結果」を集計した要約であり、
本文中では各節の理解の入口として用いる。

**目的**：本稿で凍結した検証（入力・処理・指標・棄却条件）の集約結果を俯瞰し、
各節（4.2–4.4）の入口とする。

**入力**：Table 1（検証サマリ）と、全行の状態を俯瞰するスコアボード図。

**処理**：各テーマの採用/棄却に必要な統計（z/σ/Δ 等）と、手続き系統（sweep）を
同一I/Fで固定し、Table 1 に集約する。

**指標**：Table 1 の「差/指標」列（統計量＋不確かさ、または手続き感度）を正とする。

**結果**：Table 1 に集約（各節で数値の根拠を提示）。

`output/summary/quantum_scoreboard.png`  
Table 1 全行を “OK/要改善/不一致/参考” の色で俯瞰する。  
ここでの色は結論ではなく、次に詰めるべき系統（sys）と精度ギャップの可視化を目的とする。

**注意**：スコアボードは単純化のため、カテゴリ間で指標の種類が異なる。
採用/棄却の判断は各節の「指標/棄却条件」を正とする。

---

### 4.2 量子（前提検証）

本稿の結論は量子現象の再導出ではないが、公開一次データに基づき「観測手続き（selection）が統計量を動かし得る入口」を定量化し、反証へ接続できる状態にした。以下では、量子テストを項目別に要約する。

#### 4.2.1 ベルテスト（time-tag / trial-based）

**目的**：公開一次データに対して、selection（coincidence window / trial 定義 / offset 等）の感度を定量化し、  
反証条件（どの手続きで棄却されるか）へ接続できる形で固定する。

**入力**：NIST belltestdata（photon time-tag）、Delft（event-ready; trial table）、Weihs 1998（photon time-tag）、Christensen et al. 2013（Kwiat group; photon time-tag; CH）。  
一次ソース一覧は付録「データ出典（一次ソース）」にまとめる。[NISTBelltestdata][DelftHensen2015][DelftHensen2016Srep30289][Zenodo7185335][IllinoisQIBellTestdata]


**処理**：
  - NIST：GPS PPS 整列＋click delay（sync基準）＋coincidence window sweep（greedy pairing）。併せて build（hdf5）を用いた trial-based 集計（sync×slot）を行う。
  - Delft：event-ready window start（offset）を掃引し、有効trial数と CHSH を評価する。
  - Weihs：coincidence window sweep により CHSH |S| の感度を評価する。

**指標**：NIST/Weihs は click delay の setting 依存を **遅延量（Δmedian; ns）＋残差（z）**として凍結し（KSはproxyとして併記）、  
併せて NIST は CH の `J_prob`、Weihs/Delft は CHSH `S`（固定variant）を評価する。

**結果**：

| Dataset | selectionノブ | timing bias proxy（A/B） | 統計量の変動幅（sweep） | baseline |
|---|---|---|---|---|
| NIST | window / trial 定義 | `Δmedian`=158 ns (z≈34.8) / 217 ns (z≈43.9)；KS=0.0287 / 0.0411 | CH `J_prob`: -1.41e-3→2.10e-3；trial best: 6.52e-6 | — |
| Weihs 1998 | window | `Δmedian`=0.083 ns (z≈5.54) / 0.095 ns (z≈5.53) | CHSH `|S|`: 2.48→2.90 | — |
| Delft 2015 | event-ready offset | — | CHSH `S`: 1.71→2.50 | 2.422±0.204 |
| Delft 2016（Sci Rep 30289；combined） | event-ready offset | — | CHSH `S`: 1.10→2.54 | 2.346±0.184 |

さらに、selection 系統を「誤差要因のカタログ」として 15 項目へ分解し、統計誤差 `σ_stat`（bootstrap 等）に対する比  
`abs(Δstat)/σ_stat` を **運用上の系統誤差バジェット**として固定する。ここで `Δstat` は「手続き（窓/offset/補正等）の選択」により統計量（S, `J_prob`, Δmedian 等）が動き得る幅であり、  
`abs(Δstat)/σ_stat > 1` は「統計よりも手続き依存（sys）が大きい」ことを意味する。

| 誤差要因（15項目） | median abs(Δstat)/σ_stat | max abs(Δstat)/σ_stat |
|---|---:|---:|
| offset | 3.88 | 104.09 |
| coincidence window（窓幅/ドリフト） | 3.13 | 385.34 |
| detector efficiency drift | 0.35 | 0.62 |
| polarization correction | 0.30 | 1.35 |
| threshold | 0.25 | 3.11 |
| transmission loss | 0.25 | 0.25 |
| setting switch timing | 0.24 | 3.00 |
| dark count | 0.20 | 1.00 |
| accidental correction | 0.20 | 77.91 |
| event-ready definition | 0.20 | 0.20 |
| dead time | 0.19 | 0.20 |
| environmental variation | 0.18 | 0.86 |
| trial definition | 0.15 | 77.54 |
| clock synchronization | 0.15 | 0.18 |
| electronic noise | 0.079 | 0.485 |

（読み方）上表で支配的な 2 つ（offset / coincidence window）は、装置誤差というよりも  
「どのペアを同一事象と見なすか」を決める selection ノブである。sweep 範囲を広く取ると、統計誤差 `σ_stat` を大きく上回る振れ幅を示し得るため、  
結果（S, `J_prob`）を見て window/offset を最適化すると、S>2 の“破れ”そのものが手続きに起因して見える危険がある（p-hacking の自由度）。

一方で、freeze（自然窓）を事前規則で固定した後の「近傍 drift」として見積もると、寄与は 0.0xxx のオーダーまで落ちる。  
例えば Weihs 1998 では、自然窓の近傍（window=25 ns 付近）で `d|S|/dwindow≈−5.18×10^-3 /ns` であり、  
推奨窓の drift_half≈0.251 ns から見積もると `Δ|S|≈0.0013` 程度に留まる（sweep 全体の `Δ|S|≈0.42` は「選択自由度の大きさ」を示す）。

CHSH 系列では、window/offset 等の selection 項目を除いた「装置/補正」側（dead time、検出効率drift、dark count、閾値、時計同期…）の寄与は、  
運用 proxy を含めても L2 合成で `Δ|S|≈0.05–0.14` 程度に留まる（典型的な `|S|−2≈0.3–0.6` より小さい）。  
ただし L1（線形和）の上限は相関を無視した保守側であり、Delft では 0.4 程度になり得る（同時に同符号で寄与する最悪ケース）。

| dataset | Δ|S|（装置/補正；L2） | Δ|S|（装置/補正；L1） | 備考 |
|---|---:|---:|---|
| Weihs 1998 | 0.049 | 0.081 | time-tag 由来の指標が多く、proxy 依存は相対的に小さい |
| Delft 2015 | 0.143 | 0.421 | event-ready では直接 sweep が無い項目があり、保守 proxy を含む |
| Delft 2016 | 0.133 | 0.398 | 同上（combined run） |

加えて、同一データに対して **pairing/推定器の実装差**（規約の違い）が統計量に与える影響を、
凍結した窓（または凍結点）で `Δ_impl/σ_boot` として固定した（運用ゲート：`Δ_impl/σ_boot<1`）。
このクロスチェックは「最適化」ではなく、実装差が結論を左右しないことを系統として閉じる目的である。

| dataset | 実装差クロスチェック（凍結点） | Δ_impl/σ_boot |
|---|---|---:|
| Weihs 1998 | greedy pairing vs mutual nearest-neighbor | 0.086 |
| NIST 2013 | greedy pairing vs mutual nearest-neighbor | 0.021 |
| Kwiat/Christensen 2013 | trial-based 基準点の整合（内部一致） | 0.000 |

CHSH `S` の具体例として、Weihs 1998 では coincidence window のみで `Δ|S|≈0.423`（`abs(ΔS)/σ_stat≈16.7`）動き得る。  
Delft は event-ready offset の sweep により `ΔS≈0.788`（2015；`abs(ΔS)/σ_stat≈3.88`）、`ΔS≈1.44`（2016；`abs(ΔS)/σ_stat≈7.84`）が観測される。  
これらは `S-2` のマージン（例：2.422±0.204 なら 0.422）と同程度以上であり、S の「破れ」を論じる際に selection 系統の扱いが支配的になることを定量的に示す。

上の例を、凍結点（natural window）での `S_frozen±σ_stat` と sweep の min-max でまとめると次の通りである。

| Dataset | selection凍結 | S_frozen±σ_stat | S_sweep（min→max） | S_frozen−2 | min−2 | ΔS | abs(ΔS)/σ_stat |
|---|---|---:|---:|---:|---:|---:|---:|
| Weihs 1998 | 窓幅（25 ns） | 2.553±0.025 | 2.477→2.900 | 0.553 | 0.477 | 0.423 | 16.7 |
| Delft 2015 | event-ready offset（−30 ps） | 2.431±0.203 | 1.711→2.499 | 0.431 | -0.289 | 0.788 | 3.88 |
| Delft 2016 | event-ready offset（−30 ps） | 2.321±0.184 | 1.100→2.542 | 0.321 | -0.900 | 1.44 | 7.84 |

同様に CH（`J_prob`）では、NIST は window sweep だけで `J_prob` が負→正へ跨ぎ、Kwiat/Christensen 2013 も window 依存で `J_prob` が動く（ただし本sweep範囲では 0 を跨がない）。

| Dataset | selection凍結 | J_frozen±σ_stat | J_sweep（min→max） | max(J) | ΔJ | abs(ΔJ)/σ_stat |
|---|---|---:|---:|---:|---:|---:|
| NIST | 窓幅（1000 ns） | -6.99e-4±9.10e-6 | -1.41e-3→2.10e-3 | 2.10e-3 | 3.51e-3 | 385 |
| Kwiat/Christensen 2013 | 窓幅（2812.5 ns） | -8.12e-5±3.94e-5 | -1.26e-4→-2.88e-6 | -2.88e-6 | 1.23e-4 | 3.13 |

したがって、window sweep のみで見ても、Weihs は（このwindow gridでは）`min(|S|)>2` を保つ一方、Delft は offset sweep で `S` が 2 を跨ぎ、NIST は `J_prob` が 0 を跨ぐ。  
本稿は「違反の有無」を断言するのではなく、自然な窓（freeze）と系統幅 Δ_sys をセットで凍結し、後段の棄却条件（5.1）へ接続する。

補足：dead time や detector efficiency drift などの「装置側」項目は、CHSH に対して ΔS が 0.001〜0.07 程度（proxy）に留まり、単独では `S-2` のマージンを埋めないことが多い。  
一方で coincidence window（Weihs）や event-ready offset（Delft）のような selection 定義そのものは、上の例のように ΔS が 0.4〜1.4 程度動き得るため、`S>2` の議論は window/offset をどの手続きで凍結したか（natural window）と、sweep を最適化ではなく **系統幅 Δ_sys の評価**として扱ったかが支配的になる。


**既存批判との関係**：
Adenier（2007）は、EPR/Bell 実験で暗黙に置かれがちな fair sampling（検出・採用が測定設定と独立という仮定）が、実験データから自明に支持されるとは限らず、  
採用規則が相関推定を歪め得ることを指摘する。De Raedt らの event-by-event シミュレーション群は、局所的なルールでも time-tag と coincidence 判定を組み合わせると、  
窓幅や遅延処理への依存だけで CHSH 等が量子的違反に見える値を取り得る（coincidence loophole）ことを示す。Larsson（2014）は、detection・locality・memory・coincidence-time などの loophole を整理し、  
実験が何を閉じ、何が残り得るかを評価する枠組みを与える。これらは共通して「選別/同時計数の手続き」が統計量に入り得る点を強調する。[AdenierKhrennikov2007FairSampling][MichielsenDeRaedt2013EventByEventBell][DeRaedtMichielsenHess2016IrrelevanceBell][Larsson2014LoopholesBellTests]

本稿の目的は loophole の存在を概念的に主張することではなく、公開一次データ上で selection ノブ（window/offset/threshold 等）が統計量（S, `J_prob`, Δmedian など）をどれだけ動かすかを、  
同一I/Fで横断比較できる形に固定することにある。手法として event-by-event の生成モデルや理論解析を立てる代わりに、公開 time-tag/trial を直接再解析し、  
共分散（bootstrap）と系統（sweep幅）を分離して提示する。したがって結論は「Bell 違反が人工物か否か」ではなく「手続き依存性がどの程度あるか」を数値として凍結し、  
後段の Reject（棄却閾値）に接続できる形で残す点にある。

したがって本稿は loophole の存在を新たに主張するものではなく、selection の定量的感度を公開一次データで固定し、反証条件（どの手続きで棄却されるか）への接続を目的とする。

`output/quantum/bell_selection_sensitivity_summary.png`  
selectionノブ（window/offset/threshold 等）に対する統計量の変動幅を横断比較する。  
本節の「手続き依存性（sys）」を俯瞰する主図として読む。

`output/quantum/bell/falsification_pack.png`  
selection 感度（Δ(stat)/σ_stat）と、delay の setting 依存（Δmedian; z）を dataset 横断で同一軸にまとめる。  
本稿では ratio=1 と z=3 を運用上の閾値として固定し（手続きの凍結点と reject ルールに接続）、  
「selection 入口が十分に大きいか／fast switching で遅延シグネチャが一貫して現れるか」を一目で判定できる形にした。

`output/quantum/nist_belltest_time_tag_bias__03_43_afterfixingModeLocking_s3600.png`  
NIST の time-tag で click delay 分布（setting依存）を可視化する。  
Δmedian（ns）と残差（z）を読むための診断図である。

`output/quantum/nist_belltest_trial_based__03_43_afterfixingModeLocking_s3600.png`  
NIST の trial-based と coincidence-based の差（同一データでの集計差）を比較する。  
trial 定義が母集団を変え得る入口を確認する。

`output/quantum/delft_hensen2015_chsh.png`  
Delft 2015（event-ready）の offset sweep に対する CHSH `S` の感度を示す。  
「event-ready を覆す」のではなく、手続き依存の系統幅を固定するために用いる。

`output/quantum/delft_hensen2016_srep30289_chsh.png`  
Delft 2016（Sci Rep 30289）の offset sweep に対する `S` を示す。  
複数run/統合条件での頑健性（系統幅）を比較する。

`output/quantum/weihs1998_chsh_sweep_summary__multi_subdirs.png`  
Weihs 1998 の coincidence window sweep を複数条件で横断比較する。  
window 依存の系統幅（ΔS）を、統計誤差と分離して読む。

**注意**：  
（位置づけ）ここで固定するのは「selection が入る位置」とその感度であり、量子力学の全予測の再導出を主張しない（反証への入口）。

Vienna 2015（Giustina et al. 2015）は raw time-tag 未公開のため本稿の解析対象外とし、  
公開された場合は同一I/Fで追加可能である（現時点では上記5データセットで横断統合しており十分）。[Giustina2015PRL250401]

（event-ready / natural window / p-hacking）Delft（Hensen 2015）の event-ready は detection / locality loophole を同時に閉じるためのプロトコルであり、本稿はこれを覆す主張をしない。  
offset/window の sweep は S を恣意的に下げるため（p-hacking）ではなく、**採用規則（selection）が統計母集団をどれだけ動かすか**を系統として可視化するために行う。  
併せて、p-hacking に見えないように「自然な窓（natural window）」を **データから客観的に推奨**する。NIST では trial-based の coincidence 総数に一致する time-tag half-window を推奨し（独立パイプラインとの照合）、  
Weihs/Kwiat では Δt ヒストグラムのピーク外（遠方ビン）で一定背景を推定し、accidental ≈ rate_A×rate_B×window という統計的定義に基づいて「信号＝ピーク積分−背景×窓幅」が飽和する最小 half-window を推奨値とする（drift も加味）。  
推奨窓からの拡大/縮小で統計量が動く分は、sweep 幅として系統誤差に計上する。推奨窓（natural window）とその基準点（freeze）、  
統計共分散（bootstrap; N≥1000）、accidental subtraction を含む sweep の台帳は dataset ごとに固定出力し、基準点からの逸脱を判定する reject ルール（3σ）も同時に凍結した。

**凍結**：natural window（推奨窓→gridスナップ）と基準点（freeze）、sweep 幅 Δ_sys、統計誤差 σ_stat（bootstrap）、delay シグネチャ（Δmedian; z）を dataset 別に固定した。  
**棄却**：selection 起源仮説は、(i) window/offset/trial sweep の `abs(Δstat)/σ_stat < 1`（手続き依存が統計以下）または (ii) delay シグネチャ `z<3` が成立し、かつ同一手続きで `|S|>2`（または `J_prob>0`）が 3σで再現される場合に棄却される（運用条件；5.1）。  
**次**：公開待ちでない範囲では、既存5 dataset の同一I/Fを維持したまま、threshold 等のノブも含めた `σ_total` と `Δ_sys` の比較（selection感度）を更新し、棄却判定の頑健性を上げる。

#### 4.2.2 重力×量子干渉（COW／原子干渉計／量子時計）

**目的**：地上弱場での干渉・量子時計が、P-model固有の写像差分（`exp(-x)` と `sqrt(1-2x)` の差）を
実験精度として検出できるかを定量化し、「現精度では差分検出不可」および将来の検証条件を固定する。

**入力**：COW（中性子干渉）、原子干渉計重力計、光格子時計（chronometric leveling）の代表条件と公表精度（一次論文）。

**処理**：代表条件（COW: `H,v0`、原子干渉計: `T`、時計: `Δh`）を固定し、位相/赤方偏移/周波数比を計算する。  
差分は静止時計写像 `exp(-x)` と参照写像 `sqrt(1-2x)` の差として評価し、3σ検出に必要な 1σ 精度へ換算する。

**式**：本稿の定義 `φ≡-c^2 ln(P/P0)` と、点質量 `M` の `x≡GM/(c^2 r)` により、静止時計写像は

$$
\ln\left(\frac{P}{P_0}\right) = x,\qquad
\left(\frac{d\tau}{dt}\right)_{\rm P} = \frac{P_0}{P}=e^{-x},\qquad
\left(\frac{d\tau}{dt}\right)_{\rm GR} = \sqrt{1-2x}.
$$

2つの半径 `r_low<r_high` 間の周波数比（赤方偏移）は

$$
1+z = \frac{(d\tau/dt)_{\rm high}}{(d\tau/dt)_{\rm low}}
$$

で定義し、P-model と GR で `z` の差 `δz=z_P−z_GR` を比較する。  
弱場（x ≪ 1 かつ |z| ≪ 1）では、exp と sqrt の違いは `O(x^2)` に現れ、近似として

$$
\frac{\delta z}{z_{\rm GR}} \approx -(x_{\rm low}+x_{\rm high})
$$

となる（符号は P-model の方が GR より `z` が僅かに小さいことを表す）。

**指標**：差分 `δφ`/`δ(Δf/f)`/`δz` と、3σ検出に必要な 1σ 精度（統計+系統）。

**結果**：

| 実験 | 代表条件 | 主要数値 | 一次ソース |
|---|---|---:|---|
| COW（中性子干渉） | H=3 cm, v0=2000 m/s | 位相振幅 ≈ 11.16 cycles | [Mannheim1996COW][ColellaOverhauserWerner1975] |
| 原子干渉計重力計 | T=0.4 s | 重力項位相 ≈ 2.31e7 rad | [Mueller2007AtomInterferometer] |
| 光格子時計 leveling | Δh≈399 m | z=0.85σ；ε=5.67e-4±6.68e-4 | [ArXiv2309_14953] |

P-model vs GR の差分スケール（代表値）は次の通り：

| 系（代表値） | 観測量 | P−GR差分 | 3σ検出に必要な1σ精度 |
|---|---:|---:|---:|
| COW（H=3 cm） | 位相 | δφ≈-9.75e-8 rad | 相対≲4.64e-10 |
| 原子干渉計（T=0.4 s） | 位相 | δφ≈-3.22e-2 rad | 相対≲4.64e-10 |
| 光格子時計 leveling（Δh≈399 m） | Δf/f | δ(Δf/f)≈-6.05e-23 | 絶対≲2.02e-23 |
| 地上↔ISS（400 km；参考） | Δf/f | δ(Δf/f)≈-5.54e-20 | 絶対≲1.85e-20 |
| 太陽ポテンシャル（0.3 AU↔1 AU；例） | Δf/f | δ(Δf/f)≈-9.85e-16 | 絶対≲3.28e-16 |
| 強場例（中性子星：表面↔∞） | z | δz≈-4.72e-2 | 絶対≲1.57e-2 |

`output/quantum/cow_phase_shift.png`  
COW（中性子干渉）の代表位相（重力項）を示す。  
位相の絶対値スケール（cycles）を確認する。

`output/quantum/atom_interferometer_gravimeter_phase.png`  
原子干渉計重力計の代表位相（重力項）を示す。  
差分の議論に入る前に、位相スケール（rad）を固定する。

`output/quantum/optical_clock_chronometric_leveling.png`  
光格子時計の高度差比較（chronometric leveling）の代表点を示す。  
弱場で ε=0（P-modelの最小）と整合することを確認する。

`output/quantum/gravity_quantum_interference_delta_predictions.png`  
P-model vs GR の差分（必要精度）を、代表実験精度と並べて示す。  
「現在精度では差分検出不可」を定量的に可視化する。

**考察**：P-model のポテンシャルは `φ_P≡-c^2 ln(P/P0)` で定義される。弱場で `P/P0 = 1+ε`（|ε|≪1）と書くと、

$$
\phi_P
=-c^2\ln(1+\epsilon)
=-c^2\left[\epsilon-\frac{\epsilon^2}{2}+\frac{\epsilon^3}{3}-\cdots\right].
$$

  ニュートンの点質量ポテンシャルは

$$
\phi_N=-\frac{GM}{r},\qquad x\equiv\frac{GM}{c^2 r}=\frac{|\phi_N|}{c^2}.
$$

先頭次数で `ε≈x` と同定すれば（弱場の同一化）、高次の差分は

$$
\Delta\phi\equiv\phi_P-\phi_N
 =c^2\left[\frac{x^2}{2}-\frac{x^3}{3}+\cdots\right],\qquad
\left|\frac{\Delta\phi}{\phi_N}\right|=\frac{1}{2}x+O(x^2).
$$

ただし本リポジトリの統一規約では、点質量の Poisson 解として
**`ln(P/P0)=x`（すなわち `P/P0=exp(x)`）を採用**しており、
この場合は `φ_P=−c^2 ln(P/P0)=−c^2 x=φ_N` で **保守的セクターの差分は 0** になる。  
したがって上式は「`P/P0−1` を線形化（`ln(1+ε)≈ε`）した場合に見落とす非線形高次（≃x/2）スケール」  
の見積もりとして用いる。弱場での識別可能性は、実際には本節が用いた **時計写像** `exp(-x)` と `sqrt(1-2x)` の差  
（`O(x^2)`；相対≃`x_low+x_high`）として現れる。

弱場（代表値）における実験精度と補正スケールの比較は次の通り：

| 実験 | 典型 x (=abs(φ_N)/c^2) | 現在精度（1σ；代表） | P-model補正（代表） | 検出可能性 |
|---|---:|---|---|---|
| COW（中性子；H=3 cm） | 6.95e-10 | 位相相対≈1e-2（想定） | δφ≈−9.75e-8 rad（相対≈1.39e-9） | No（必要：相対≲4.64e-10 for 3σ） |
| 原子干渉計（T=0.4 s） | 6.95e-10 | 位相相対≈1e-9（想定） | δφ≈−3.22e-2 rad（相対≈1.39e-9） | No（必要：相対≲4.64e-10 for 3σ） |
| 光格子時計（Δh≈399 m） | 6.95e-10 | σ(Δf/f)≈2.89e-17 | δ(Δf/f)≈−6.05e-23 | No（必要：≲2.02e-23 for 3σ） |
| ACES/PHARAO（地上↔ISS 400 km；参考） | ~6–7e-10 | TBD（ミッション仕様） | δ(Δf/f)≈−5.54e-20 | 将来検討（必要：≲1.85e-20 for 3σ） |

地上弱場（x≈10^-9）では、P-model の写像差分は `O(x^2)` で極小になり、既存の干渉実験・時計比較が直ちに差分検出器になるとは言いにくい（本節の代表値でも要求精度は非常に厳しい）。  
一方で、太陽ポテンシャルの長基線（深宇宙時計）や、強場天体の重力赤方偏移は差分が大きくなり得るため、本モデルの差分予測を観測へ接続する候補となる（ただし後者は天体物理の系統が支配的になりやすい）。

β（光伝播）由来の差分について：Mueller 2007 の導出では `k_eff` が光の分散（レーザ波数）から決まるため、本モデルの `n(P)=(P/P0)^(2β)` はレーザー位相項を通じて位相へ入り得る。  
ただし局所実験では「定数項（共通モード）」が相殺/吸収されるため、差分は干渉計内のポテンシャル差で強く抑圧される。凍結β（Cassini）と代表値（T=0.4 s）では、採用モデル（局所差分）で **δφ_beta≈4.16e-14 rad**、3σ検出に必要な 1σ 精度は **≲1.39e-14 rad** と見積もられる。

**棄却条件**：各代表条件で、観測された赤方偏移/位相が P-model の静止時計写像（`exp(-x)`）から総合不確かさ（統計+系統）の 3σ を超えてずれるなら、この最小写像は棄却する。  

**凍結**：COW/原子干渉計/量子時計の代表条件を固定し、P−GR の差分スケール（δφ, δ(Δf/f), δz）と「3σ検出に必要な1σ精度」を表として凍結した。  
**棄却**：上記の 3σゲート（赤方偏移/位相が `exp(-x)` から逸脱）を運用棄却条件として凍結し、これを満たさない限り差分検出の主張はしない。  
**次**：差分は弱場で `O(x^2)` となるため短基線では決着しにくい。太陽ポテンシャルの長基線（深宇宙時計/周回）や強場赤方偏移で、必要精度（表）を満たす測定が現れた時点で差分予測として更新する。

#### 4.2.3 物質波干渉（電子二重スリット／de Broglie 精密）

**目的**：物質波干渉のスケール（de Broglie 波長と干渉縞）を代表実験で確認し、  
独立定数（α）の整合性を z-score として固定して、棄却条件へ接続する。

**入力**：電子二重スリット（代表条件）と、原子反跳（干渉）由来の α（独立測定との比較）。

**処理**：電子は運動量から de Broglie 波長を算出し、干渉縞の角スケールを評価する。  
α は独立測定との差を z-score に変換して固定する。

**式**：

$$
\lambda=\frac{h}{p},\qquad p\simeq \sqrt{2mE},\qquad
z=\frac{\alpha_{\rm obs}-\alpha_{\rm ref}}{\sigma}.
$$

**指標**：de Broglie 波長と縞スケール、ならびに α 整合の z-score（棄却条件：|z|>3）。

**結果**：

| テーマ | 代表条件 | 主要数値 | 一次ソース |
|---|---|---:|---|
| 電子二重スリット | 600 eV | de Broglie 波長 50.05 pm；縞間隔 0.179 mrad | [Bach2012ElectronDoubleSlit] |
| de Broglie 精密（α） | 原子反跳→α | z=0.59σ；有効ε=-5.33e-9±9.08e-9 | [Bouchendira2008AlphaRecoil][Gabrielse2008AlphaG2] |
| 横断I/F（電子/原子/分子） | 統合 | channels_n=4；atom α 整合 z=0.59；molecular scaling z_max=2.64；atom precision gap median=7.18e5 | 統合固定出力（一次ソース統合） |

`output/quantum/electron_double_slit_interference.png`  
電子二重スリットの縞間隔と、de Broglie 波長の代表値を示す。  
物質波干渉のスケール確認（入口）として用いる。

`output/quantum/de_broglie_precision_alpha_consistency.png`  
原子反跳（干渉）由来の α と、独立測定（g-2）との整合性を示す。  
有効εの z-score を固定し、棄却条件（z>3）へ接続する。

`output/quantum/matter_wave_interference_precision_audit.png`  
電子（縞スケール）、原子（α整合と重力干渉の精度ギャップ）、分子（同位体縮約質量スケーリング）を
同一I/Fで横断表示した統合監査図である。  
本節で固定した横断指標（channels_n=4）を、後続の光干渉（HOM/スクイーズド光）へ接続する入口として扱う。

**凍結**：電子干渉の代表縞スケールと、原子反跳由来の α の独立クロスチェック（z-score）を固定し、電子/原子/分子を横断する監査指標（channels_n, z_max 等）の入口を作った。  
**棄却**：独立測定間で α が `|z|>3` を示す、または干渉縞スケールが de Broglie 予測から総合不確かさの 3σ を超えて外れるなら、この最小整合は棄却される。  
**次**：分子干渉（大型分子/温度依存）と同位体系統を追加し、横断指標（channels_n, z_max）を更新して光干渉（4.2.5）との対比を強める。

#### 4.2.4 重力誘起デコヒーレンス（可視度低下）

**目的**：重力赤方偏移に起因する ensemble dephasing（可視度低下）のスケールを、代表パラメータで固定し、  
弱場の実験室条件では効果が極小であること（差分検出が主張点でないこと）を明示する。

**入力**：Pikovski 型の dephasing モデルと、代表パラメータ（σ_z, t, σ_y）の評価。

**処理**：高さ分布の幅 `σ_z` を仮定し、`V=0.5` に到達する時間スケール `t_half` と、等価な `σ_y` を算出して固定する。

**指標**：可視度 `V` と、追加の時間構造ノイズ（fractional rate noise）`σ_y` の必要条件。

**結果**：高さ分布の幅 `σ_z` を 0.1/1/10 mm としたとき、`V=0.5` に対応する時間スケールはそれぞれ **4.0e4 s / 4.0e3 s / 4.0e2 s**（同等に `σ_y` は 10倍刻みで増える）。[Pikovski2013TimeDilationDecoherence][Bonder2015Decoherence][AnastopoulosHu2015Decoherence][Pikovski2015Reply][Hasegawa2021LatticeClockDephasing]

**棄却条件**：弱場実験室条件で、観測される可視度低下が本節の重力起因dephasing見積もりを総合不確かさの 3σ を超えて上回るなら、重力起因dephasing単独仮説は棄却する（支配因は別系統）。  

**注意**：  
（collapse との区別）本節で扱う「デコヒーレンス」は、時間遅れ／位相拡散としての可視度低下（dephasing）であり、Penrose–Diósi 型の「重力による自発崩壊（objective collapse）」とは別の仮説である。[Diosi1987][Penrose1996StateReduction]  
  したがって、collapse model に対する地下実験等の制約は本節のモデルへ自動的に適用できない（ただし将来、collapse 型の追加仮定を導入する場合は別途の制約整理が必須）。代表的に、Diósi–Penrose 型の検証として地下環境での実験制約が議論されている。[Donadi2020UndergroundCollapse]  
  本稿の数値（`V=0.5` が `10^2–10^4 s` 級）から、弱場の実験室条件では効果は極小であり、既存の干渉実験を直ちに破る補正は要求されない（安全宣言）。

`output/quantum/gravity_induced_decoherence.png`  
高さ分布（σ_z）に対する可視度低下のスケールと、必要な `σ_y` を示す。  
弱場の実験室条件では効果が極小であることを確認する。

**凍結**：Pikovski 型の重力起因 dephasing の代表スケール（σ_z=0.1/1/10 mm に対する `t_half` と等価 `σ_y`）を固定した。  
**棄却**：観測される可視度低下が上の見積もりを総合不確かさの 3σ を超えて上回るなら、重力起因 dephasing 単独仮説は棄却される。  
**次**：将来の高精度干渉計/時計で `σ_y` を独立に拘束し、dephasing/collapse/環境ノイズの分離監査（どの項が支配か）へ接続する。

#### 4.2.5 光の量子干渉（単一光子／HOM／スクイーズド光）

**目的**：光学側の位相ノイズ・損失が支配するスケールを、代表実験の数値で固定し、  
以降の「差分予測」ではなく「必要条件（整合）」として位置づける。

**入力**：単一光子干渉、HOM、スクイーズド光の代表実験（一次ソース）。

**処理**：可視度や損失（η）を、等価な光路長雑音などのスケールに変換して比較する。

**指標**：等価光路長雑音 `σ_L`、HOM 可視度 `V`、損失下限 `η`。

**結果**：

| テーマ | 代表条件 | 主要数値 | 一次ソース |
|---|---|---:|---|
| 単一光子干渉 | V≥0.8, λ=1550 nm | 等価光路長雑音 σ_L≈165 nm | [Kimura2004SinglePhotonInterference] |
| HOM | 例：D=13 ns / 1 μs | visibility=0.982±0.013 / 0.987^{+0.013}_{-0.020} | [ArXiv2106_03871][Zenodo6371310] |
| スクイーズド光 | 10 dB | loss-only 下限 η≥0.900 | [Vahlbruch2007Squeezed10dB] |
| HOM+スクイーズド光統合I/F | 統合 | channels_n=5；HOM z=37.1/24.35；遅延差分 z=0.21；squeezing variance ratio=0.10；PSD(10k/100k)=0.84 | [ArXiv2106_03871][Zenodo6371310][Vahlbruch2007Squeezed10dB] |

**棄却条件**：HOM の可視度が古典限界（V<=0.5）に留まる、またはスクイーズド光の 10 dB 条件（loss-only 下限 η>=0.90）を満たさない場合、本節の量子干渉I/Fは成立しない（棄却）。  

`output/quantum/photon_quantum_interference.png`  
単一光子干渉/HOM/スクイーズド光の代表数値をまとめて示す。  
光学側の位相ノイズ・損失が支配するスケールを固定する。

`output/quantum/hom_squeezed_light_unified_audit.png`  
HOM可視度の古典閾値（50%）に対する有意性、13 ns↔1 μs の遅延依存、  
スクイーズド光の分散比（10 dB→0.10）と PSD 比（10 kHz/100 kHz）を同一I/Fで監査した図である。  
本節の固定出力として、次段の独立cross-checkへ接続する。

**凍結**：単一光子干渉/HOM/スクイーズド光の代表値（`V`, `η`, `σ_L`）と、古典限界（HOM: `V<=0.5`）からの有意性を統合指標として固定した。  
**棄却**：上記の棄却条件（HOM が古典限界に留まる、または 10 dB 条件を満たさない）が成立する場合、本節の量子干渉I/Fは棄却される。  
**次**：遅延 sweep の全カーブ（`V(τ)`）と損失/位相雑音の誤差予算を拡張し、Bell と同型の「手続き依存監査」へ寄せる。

#### 4.2.6 真空・QED精密（Casimir／Lamb shift）

**目的**：既知の精密物理（Casimir/Lamb shift/H 1S–2S/α）をベースラインとして固定し、  
P-model の最小結合が直ちに矛盾を生まないこと（安全確認）を定量で示す。

**入力**：Casimir、Lamb shift、H 1S–2S、α（原子反跳 vs 電子 g-2）の一次ソース。

**処理**：代表値を台帳化し、原子スケールでの重力ポテンシャル起源の寄与をオーダー推定して比較する。

**指標**：代表精度（相対/絶対）、α 整合の z-score、オーダー推定 `|φ_nuc|/c^2` と `ΔE`。

**結果**：

| プローブ | 主要数値（代表） | 精度/注 | 一次ソース |
|---|---:|---|---|
| Casimir | sphere-plane；球直径 201.7 μm | 相対精度 ~1%（最短距離付近） | [RoyLinMohideen2000Casimir] |
| Lamb shift | スケーリング整理（Z^4, Z^6） | 核サイズ項など非QED系統の寄与例 | [IvanovKarshenboim2000LambShift] |
| H 1S–2S | 2466061413187035(10) Hz | fractional 4.2×10^-15（不確かさ予算含む） | [Parthey2011Hydrogen1S2S] |
| α（独立クロスチェック） | z≈0.59σ | 原子反跳 vs 電子 g-2 | [Bouchendira2008AlphaRecoil][Gabrielse2008AlphaG2] |

安全確認（原子スケール）：P-model の最小結合（`φ=-c^2 ln(P/P0)` を重力ポテンシャルとして扱う）では、核の重力ポテンシャルは原子スケールで極小であり、電子の準位に与える追加項は現在の QED 精密測定より桁違いに小さい（既知の精密物理と矛盾しない）。オーダーは

$$
\left|\frac{\phi_{\rm nuc}}{c^2}\right|
\sim \frac{GM_{\rm nuc}}{c^2 r},
\qquad
  \Delta E \sim m_e\,|\phi_{\rm nuc}|
$$

  で与えられ、代表的には `|φ_nuc|/c^2≈10^-44（H）〜10^-40（重元素）`、`ΔE≈10^-38〜10^-34 eV` 程度である。したがって本稿が扱う範囲では、Lamb shift や g-2 などの QED 精密テストを直ちに破る補正は要求されない。  
  ただし将来、Pゆらぎが短距離で追加結合（非重力的）を持つモデルを導入する場合は、QED 精密量を使った上限拘束が必須である。

**棄却条件**：将来の追加結合を含むモデルが、H 1S–2S などの精密量で不確かさ予算を 3σ 以上超える予測差を出すなら、その追加結合は棄却する（本節は監査入口）。  

`output/quantum/qed_vacuum_precision.png`  
Casimir/Lamb shift/H 1S–2S/α の入口をまとめ、既知の精密物理と矛盾しないことを示す。  
原子スケールでの追加補正が極小である（安全確認）を読む。

**凍結**：Casimir/Lamb shift/H 1S–2S/α の基準値と、原子スケールでの追加補正のオーダー上限（`|φ_nuc|/c^2`, `ΔE`）を「安全確認」として固定した。  
**棄却**：追加結合を含む拡張が、これら精密量の不確かさ予算を 3σ 以上系統的に超える差を予測するなら、その拡張は棄却される。  
**次**：短距離での追加結合（非重力）を導入する場合は、QED精密を用いた上限拘束を最初から誤差予算へ組み込み、Pゆらぎ仮説のパラメータ空間を先に閉じる。

#### 4.2.7 原子核（総論）

**目的**：核スケールの検証を、  
（A）操作的I/Fとしての有効モデル検証（4.2.7.1–4.2.7.3）と、  
（B）第一原理的解釈としての波動干渉理論（4.2.7.4）に分離して固定する。  
この節の総論では、まず有効モデルの棄却/非棄却を固定し、続く 4.2.7.4 で  
Part I 2.7 の `Δω→B.E.` 写像を観測I/Fへ接続する。

接続規則は次の通りに固定する。  
有効モデルは「観測量を再現可能な操作的I/F」として扱い、閾値判定を先に固定する。  
波動干渉理論は「そのI/Fを生む第一原理的解釈」として扱い、同じ観測I/Fへ投影して比較する。  
両者が一致する領域は、deuteron/He-4 近傍と `A=2–16` の入口診断（4.2.7.4–4.2.7.6）で確認し、  
乖離が増える領域は 5.4 の差分予測（`ΔB`, `σ_req`）で判定する。  
したがって本節の導線は「4.2.7.2（操作的I/Fの最小固定）→4.2.7.4–4.2.7.6（解釈層の接続）→5.4（反証条件）」である。

**入力**：deuteron 質量欠損（CODATA/NIST）と、np散乱の低エネルギーパラメータ（`a_t,r_t` 等；eq18–19）。  
一次ソース一覧は付録「データ出典（一次ソース）」にまとめる。

**処理**：deuteron の束縛エネルギー B と、np散乱の低エネルギーパラメータ（散乱長 a、有効レンジ r、shape `v2`）を  
「拘束（fit）」と「予言（pred）」に分けて固定する。解析依存の系統として、eq18（GWU/SAID）と eq19（Nijmegen）の差は  
“包絡（min–max）” として扱い、予言が包絡へ入るかで棄却/非棄却を判定する。

**式**：低エネルギー散乱は有効レンジ展開（ERE）で表す：

$$
k\cot\delta_{t,s}(k)= -\frac{1}{a_{t,s}} + \frac{1}{2} r_{t,s} k^2 + v_{2,t,s} k^4 + \cdots
$$

**指標**：triplet/singlet の予言値（`r_t,v2t,r_s,v2s`）が、eq18–eq19 の包絡に入るか。  
特に `v2s` の符号は差分予測として強く、符号が救えない class は棄却する。

**結果**：

deuteron 束縛エネルギー（入口）：**B≈2.224566 MeV**（CODATA 由来）。

| ansatz class | target（拘束/予言） | key result（要約） | 判定 |
|---|---|---|---|
| square-well（2 param） | fit: `B,a_t`；pred: `r_t` | `r_t` が +0.009〜+0.028 fm ずれる（eq18/eq19） | 棄却（粗すぎ） |
| core+well | fit: `B,a_t,r_t`；pred: `v2t` と singlet `r_s,v2s` | `r_t` が包絡に入らず、`v2t` も大きすぎ／singlet も不整合 | 棄却 |
| repulsive core + well | fit: triplet `v2t` まで；pred: singlet | 最適解が `Rc→0,Vc→0` に潰れて square-well へ退化 | 棄却 |
| 2-range（2段；幾何共有） | fit: triplet `B,a_t,r_t,v2t`；pred: singlet `r_s,v2s` | singlet `r_s≈1.88–2.00 fm`（obs 2.626–2.678）かつ `v2s` の符号不一致 | 棄却 |
| 2-range（singlet自由度追加） | fit: singlet `a_s,r_s`；pred: `v2s` | `v2s>0`（≈+0.52,+1.12）で obs 包絡（-0.48〜-0.005）外 | 棄却（決定的） |
| repulsive core + 2-range | pred: `v2s` | core を足しても `v2s>0` のまま | 棄却 |
| λπ 制約（2/3-range） | pred: `v2s` | tail を足しても `v2s>0` で符号救済不可 | 棄却強化 |
| barrier+tail（global k,q） | pred: `v2`（両チャネル） | `v2s<0` は達成するが、同時に `v2t` が包絡外へ出る | 自由度不足 |
| barrier+tail（範囲拡張） | pred: triplet `v2t` と singlet `r_s,v2s` | `v2t,v2s` は包絡内でも `r_s` が崩壊（~1.07 fm） | no-go |
| barrier+tail（(k,q)分離） | pred: triplet `v2t` と singlet `r_s,v2s` | `r_s` が常に崩壊（~1.17 fm） | no-go |
| singlet 幾何（`R1_s/λπ`） | trade-off | `r_s` と `v2s` が両立せず | no-go |
| singlet 幾何（`R2_s/λπ`） | fit: singlet `r_s,v2s` | `R2_s/λπ=1.5` で `r_s,v2s` が同時に包絡内 | 解除（singlet） |
| triplet split（barrier比/k_t） | fit: triplet `a_t,r_t,v2t` | singlet を凍結（`R1_s/λπ=0.45`, `R2_s/λπ=1.5`）したまま、triplet の最小追加幾何として `barrier_len_fraction_t` と `k_t` を走査し、`barrier_len_fraction_t=0.1`, `k_t=0.0` で両datasetが包絡内（例：`v2t≈0.052–0.055`、`v2s<0`） | 解除（暫定） |

canonical 解（triplet split）の主要数値（ERE; k→0）：

| dataset | a_t (fm) | r_t (fm) | v2t (fm^3) | a_s (fm) | r_s (fm) | v2s (fm^3) |
|---|---:|---:|---:|---:|---:|---:|
| eq18（GWU/SAID） | 5.410 | 1.750 | 0.0520 | -23.719 | 2.626 | -0.1028 |
| eq19（Nijmegen） | 5.412 | 1.752 | 0.0547 | -23.739 | 2.676 | -0.0584 |

`output/quantum/nuclear_binding_deuteron.png`  
deuteron の束縛エネルギー B と、サイズ拘束の位置づけを示す。  
以降の `u(r)` ansatz が満たすべき最低条件を確認する。

`output/quantum/nuclear_np_scattering_baseline.png`  
np散乱の低エネルギーパラメータ（`a_t,r_t,a_s,r_s`）を一次固定した基準図である。  
eq18/eq19 の dataset 系統を、同一I/Fで比較する。

`output/quantum/nuclear_effective_potential_two_range_fit_as_rs.png`  
2-range ansatz で singlet を `a_s,r_s` で拘束し、`v2s` を差分予測として評価する。  
`v2s` の符号不一致が棄却点になることを可視化する。

`output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan.png`  
λ_π 制約下の barrier+tail（global (k,q)）で `v2s` 包絡へ入る例を示す。  
「最小拡張で棄却されない ansatz class」を固定する入口として使う。

`output/quantum/nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan.png`  
singlet の幾何 split（`R1_s/λπ=0.45`, `R2_s/λπ=1.5`）を凍結したまま、triplet の barrier/tail 分割比を走査し、  
canonical 解（`barrier_len_fraction_t=0.1`, `k_t=0.0`）で両datasetが包絡内へ入ることを示す。

**考察**：現時点では、2-range 系の最小拡張では singlet の `v2s` の符号を救えず棄却される。  
一方、λ_π 制約下の barrier+tail（global (k,q)）は singlet 側の `v2s` を包絡へ入れられるが、  
triplet 側と同時に満たすには追加の自由度が必要であり、強い相互作用側（有効ポテンシャルの拡張）の継続課題として残る。


**注意**：本節は「核力の第一原理導出」を主張しない。P-model 変数 `u=ln(P/P0)` と核観測量（B, a, r…）を結ぶ **有効拘束（effective model）**を凍結し、棄却された ansatz class（2-range 系）と、最小拡張で棄却されない ansatz class（λ_π 制約下の barrier+tail；global (k,q)）を区別して固定することを目的とする。

##### 4.2.7.1 電荷半径の shell-closure kink（Δ²r）の no-go整理

**目的**：電荷半径の shell-closure kink を観測量 Δ²r で直接比較し、  
(i) base（mean-field＋半径写像）での no-go、(ii) 系列共分散（統計処理側）の頑健性、  
(iii) 核構造側の最小拡張（pairing）で救済できるか、を固定する。

**入力**：IAEA charge radii（採用値＋σ）、AME2020（OES/pairing のための束縛エネルギー）、  
NNDC（β2；変形補正の候補）、および本稿の核モジュール（HF+3-body＋半径写像）の固定出力。

**処理**：Δ²r を Sn/Sp の両軸で評価し、strict 判定 `max abs(Δ²r_pred−Δ²r_obs)/σ ≤ 3` を固定する。  
頑健性として、AR(1) 相関 ρ を導入して `σ^2=a^TΣa`（`a=[1,-2,1]`）で σ を再計算し、  
「統計処理の仮定だけで救えるか」を判定する。救済案として pairing 補正 `r -> r + k_n Δ_n + k_p Δ_p` を導入し、  
`k_n,k_p` は非magic中心（A>=40）で weighted LS により凍結した上で kink set を再評価する。

**式**：

$$
\Delta^2 r(x) = r(x+2) - 2r(x) + r(x-2)
$$

AR(1) 共分散（最小モデル）：

$$
\sigma_{\Delta^2 r}^2 = a^T\Sigma a,\quad a=[1,-2,1],\quad
\Sigma_{ij}=\rho^{|i-j|}\sigma_i\sigma_j.
$$

**指標**：A_min=100 の strict set（even-even only, center-magic only）に対する  
`max abs(Δ²r_pred−Δ²r_obs)/σ_{Δ²r}`（Sn/Sp の両軸）。

**結果**：

A_min=100（独立 σ, ρ=0）では、base の worst が **Sn=3.075σ, Sp=3.723σ**となり no-go が継続する。  
AR(1) 共分散 sweep の結論は「救済には反相関が必要」であり、ρ>=0 は救済にならない：

| ρ（AR(1)） | max Sn (σ) | max Sp (σ) | 判定 |
|---:|---:|---:|---|
| 0.00（独立） | 3.075 | 3.723 | no-go |
| -0.40（closest pass） | 2.432 | 2.964 | pass |
| pass 範囲 | ρ∈[-0.90, -0.40] | — | 反相関が必要 |

核構造側の最小拡張として pairing を半径へ明示的に入れると（β2補正なし）、strict pass に到達する：

| モデル（A_min=100） | max Sn (σ) | max Sp (σ) | 判定 |
|---|---:|---:|---|
| base（def0; pairing無し） | 3.075 | 3.723 | no-go |
| deformation only（def1; β2） | 5.534 | 9.296 | no-go（悪化） |
| pairing（def0_pair_fit; `k_n≈0.0116`, `k_p≈0.00872`） | 0.688 | 2.014 | strict pass |
| pairing + deformation（def1_pair_fit） | 3.185 | 7.171 | no-go |

**考察**：統計処理（共分散）の仮定だけで救済するには、隣接点が反相関（ρ<=-0.40）である必要がある。  
ρ>=0（共通モードの系統に近い）では σ が縮んで残差が増えるため、救済にはならない。  
したがって、A_min=100 の no-go は「データ処理の揺らぎ」ではなく、核構造側の追加物理（pairing）を要請する条件付き制限として整理する。


**注意**：本節の “救済” は、kink 点での再フィットではなく、非magic中心で凍結した `k_n,k_p` を用いる。  
したがって pairing は「追加fitで逃げる」のではなく、核構造の追加自由度を明示して固定した拡張である。

**pairing補正の物理的根拠**：
核の pairing 相互作用により、偶数核は奇数核より束縛が強く、質量差・分離エネルギー・半径などに奇偶の揺らぎ
（odd-even staggering; OES）が現れる。Δ_n, Δ_p を AME2020 の束縛エネルギーから 3点公式で作るのは、
局所的な pairing gap をデータから推定する核物理の標準手法であり、ここで導入している補正は「P-model固有の仮定」ではなく
核構造の既知の自由度を明示化したものである。[BohrMottelson1969]

半径 `r` は核密度 ρ と概ね `A≈∫ρ d^3r ≈ (4π/3) r^3 ρ` で結びつくため、A を固定して微小変化をとれば
`dr/r ≈ -(dρ)/(3ρ)` が成り立つ。pairing は占有数の平滑化や表面の polarization を通じて密度分布（したがって半径）に
影響し得るので、局所的な gap 指標 Δ_n, Δ_p の変化が半径へ一次（線形）に寄与する形
`r -> r + k_n Δ_n + k_p Δ_p` は、最小の応答モデルとして物理的に自然である。

k_n, k_p は「pairing gap 1 MeV あたりの半径変化」を表す応答係数であり、ここでは
`k_n≈0.0116 fm/MeV`, `k_p≈0.00872 fm/MeV` を **非magic中心（A>=40）のデータ**から weighted LS で凍結し、
kink 点（shell closure 近傍）では再フィットしない。追加自由度は 2つ（k_n,k_p）に限定し、さらに同じ凍結手順で β2 幾何補正は
失敗するため基準結果として採用しないことで、「追加fitで逃げる」恣意性を避けている。

したがって pairing補正は Part I のコア写像を修正するものではなく、核構造の追加物理を明示的に入れた最小拡張として位置づける。

採用（この稿の核モジュール；凍結）：
- Δ²r（shell-closure kink）の評価では、pairing補正（def0）を採用する（`r -> r + k_n Δ_n + k_p Δ_p`）。`k_n,k_p` は `fit_protocol`（nonmagic中心・A>=40 の weighted LS）で凍結し、kink点では再フィットしない（過剰適合を避ける）。
- deformation（β2）の幾何補正（def1）は、同一の凍結手順では fail するため、本稿の基準結果としては **採用しない**（採用する場合は別仮説として反証条件を独立に凍結する）。

反証条件（固定；Δ²r shell-closure kink）：
- 採用モデル（pairingのみ；def0）で A_min=100 の strict set に対し、Sn/Sp の両軸で `max abs(Δ²r_pred−Δ²r_obs)/σ_obs ≤ 3` を満たさない場合、この核モジュール（追加自由度と凍結手順を含む）は棄却される。

**出力**：半径 kink（Δ²r）の strict 判定 metrics（A_min=100; pairing最小）。
（再現に用いるファイル一覧は 8章にまとめる。）


##### 4.2.7.2 Δω→B.E.写像の最小追加物理（ν飽和）の固定

**目的**：前段で凍結した反証条件（3σ運用）を変更せず、  
距離I/F（local spacing `d`）に対して最小追加物理を入れたときに、  
判定（pass/fail）がどう変わるかを固定する。

**入力**：距離I/F（local spacing `d`）の差分予測（核種ごとの予測/観測/残差）と、棄却条件（3σ運用; 閾値定義）。

**出力（監査用）**：
- 核種別の予測/観測/残差テーブル（差分予測の台帳）。
- 3σ運用の棄却条件パック（閾値定義＋適用条件）。
（再現に用いるファイル一覧は 8章にまとめる。）

**処理**：距離I/Fは `d` を維持したまま、結合数指標を  
`ν_base = 2(A−1)/A` から `ν_eff = min(ν_base, ν_sat)` へ置換する。  
本稿では `ν_sat=1.5` を凍結し、`C_eff=(ν_eff·A)/2` を用いる。

**式**：

$$
\nu_{\rm base}=\frac{2(A-1)}{A},\quad
\nu_{\rm eff}=\min(\nu_{\rm base},\nu_{\rm sat}),\quad
C_{\rm eff}=\frac{\nu_{\rm eff}A}{2}.
$$

**指標**：`abs(z_median) <= 3`、`abs(z_Δmedian) <= 3`（3σ運用の凍結値）。

**結果**：

| モデル | median(log10(B_pred/B_obs)) | z_median | z_Δmedian | 判定 |
|---|---:|---:|---:|---|
| local spacing `d` | 0.1557 | 5.63 | 2.39 | median fail |
| `d` + `ν`飽和 | 0.0345 | 1.31 | 2.21 | pass |

`output/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics.png`  
左は baseline（`d` のみ）と `d+ν`飽和の残差比較で、中央値ズレが最も大きく改善する。  
右は閾値を固定したままの z 指標比較で、`z_median` が 3σ内へ入ることを示す。  
閾値自体は変更していないため、判定改善は I/F 側の最小追加物理に起因する。

**考察**：この結果は、heavy-A で支配的だった大域オフセットが  
「距離I/Fの変更」ではなく「結合数指標の非飽和」に強く依存していた可能性を示す。  
したがって本節の ν飽和は、反証条件を維持したままの構造的改善として扱う。

**注意**：本節の `ν_eff` は、核子配位数を第一原理で主張する量ではなく、  
Δω→B.E.写像の操作的I/Fを閉じるための最小指標である。  
pairing/shell は引き続き二次構造として別途の反証条件で監査する。

##### 4.2.7.3 既存理論との差分予測と精度要件

**目的**：前節で固定した P-model 写像を、
標準理論proxyと同一の AME2020 核種集合で比較し、
どの核種群で差分予測が決着しやすいかを数値で固定する。

**入力**：前節の ν飽和を含む P-model 写像の予測/残差テーブルと、3σ運用の棄却条件（閾値固定）。

**出力（監査用）**：
- 全核種の予測/観測/残差テーブル（AME2020）。
- 反証条件パック（3σ運用閾値＋精度ゲート＋適用条件）。
（再現に用いるファイル一覧は 8章にまとめる。）

**処理**：
P-model（local spacing `d` + `ν_sat`）を基準とし、
湯川距離proxy（global `R`）と SEMF（固定係数）を同一I/Fで比較する。
つづいて `ΔB = B_pred_Pmodel - B_pred_standard` を核種ごとに計算し、
`σ_req = abs(ΔB)/3` を 3σ 決着に必要な精度として固定する。

**式**：

$$
\Delta B = B_{\rm pred}^{\rm P} - B_{\rm pred}^{\rm std},\qquad
\sigma_{\rm req,3\sigma}=\frac{\mathrm{abs}(\Delta B)}{3},\qquad
\sigma_{\rm req,rel}=\frac{\sigma_{\rm req,3\sigma}}{B_{\rm obs}}.
$$

**指標**：
`z_median`、`z_Δmedian`（3σ運用閾値を流用）と、
`median abs(ΔB)`、`median σ_req,rel`。

**結果**：

| モデル | median(log10(B_pred/B_obs)) | z_median | z_Δmedian | median abs residual (MeV) | 判定 |
|---|---:|---:|---:|---:|---|
| P-model（local+d, ν飽和） | 0.0345 | 1.31 | 2.21 | 76.73 | pass |
| 湯川距離proxy（global R） | -1.1933 | -3.22 | -2.61 | 1072.50 | median fail |
| SEMF（固定係数） | 0.0025 | 1.51 | 1.19 | 6.69 | pass |

| 差分チャネル | median abs(ΔB) (MeV) | p84 (MeV) | max (MeV) | median σ_req,rel |
|---|---:|---:|---:|---:|
| P-SEMF | 75.15 | 230.77 | 472.66 | 0.0270 |
| P-Yukawa proxy | 1150.06 | 1866.18 | 2532.92 | 0.3343 |

`output/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.png`  
上段で `B_pred/B_obs` の分布と 3σ運用指標を比較し、
P-model と SEMF は運用閾値内、湯川距離proxyは中央値で閾値外となる。
下段は差分 `ΔB` の A依存を示し、重核側で差分が増幅することを固定する。

`output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification.png`  
`abs(ΔB)` の核種分布、上位核種、必要相対精度分布を同時表示し、
差分検証の優先核種を `..._top20.csv` と一対で固定する。
重核群では P-SEMF の `median σ_req,rel` が約 3.4% となる。

Z–N平面上での結合エネルギー残差マップでは、各核種を（Z,N）に配置し、`log10(B_pred/B_obs)` を色で示す。  
魔法数（N,Z=2,8,20,28,50,82,126）の縦横線付近で残差が相対的に大きく、核種別残差指標（`log10(B_pred/B_obs)` の robust z）で `abs(z)>3` となる外れの割合は、  
magic 群で 21/298=7.0% と、非magic 群 59/3256=1.8% より高い。これは殻閉殻（shell closure）が「滑らかな距離写像」だけでは吸収できず、殻/相関項の扱いが支配的になることを示す。  
一方で、変形核領域（観測 β2 が利用できるサブセットで `abs(β2)≥0.20` かつ非magic）では、同じ指標の中央値 `median abs(z)≈0.52`（全非magic 0.66）に下がり、外れ率も 5/237=2.1% に留まる。  
したがって本稿の操作的 P-model 写像は、変形核では bulk の束縛を相対的によく捉える一方、魔法数近傍では殻閉殻効果が残差の主因になる、という対比を文章として固定できる。

要約（z は `log10(B_pred/B_obs)` の robust z；外れは `abs(z)>3`）：

| 群 | n | 外れ | 外れ率 | median abs(z) |
|---|---:|---:|---:|---:|
| magic（N または Z が 2,8,20,28,50,82,126） | 298 | 21 | 7.0% | 0.896 |
| 非magic | 3256 | 59 | 1.8% | 0.661 |
| 変形核（β2 利用可能 subset；`abs(β2)≥0.20` かつ非magic） | 237 | 5 | 2.1% | 0.52 |

本稿の差分チャネル統合では、上記の差分チャネル
（P-SEMF / P-Yukawa）を反証条件パックへ統合した。
運用上は `σ_rel(total) <= σ_req,rel` を満たす核種群のみを
3σ判定対象として採用する。

**棄却条件**：差分チャネルの採否判定は、反証条件パックに定義された conditions/thresholds（8章の一覧）を正とし、precision gate を満たす核種群で 3σ 閾値を超える場合は棄却する。  

**考察**：
本節での差分は、軽核より重核で大きく、
「どの核種を優先観測すれば P-model と標準proxyの差を最短で切れるか」
という運用判断を定量化する。

**注意**：
本節の湯川/EFT比較は運用proxyであり、第一原理EFT計算そのものではない。
ここで固定した `σ_req` は追加自由度導入前の基準として扱う。

##### 4.2.7.4 波動干渉理論の検証（重陽子）

**目的**：Part I 2.7 で固定した `Δω→B.E.` 写像を、最小2核子系（重陽子）で  
観測I/Fに接続し、次段の He-4・軽核系統へ拡張する入口を固定する。

**入力**：観測 `B.E._obs=2.224566342±0.000000792 MeV`（CODATA/NIST）と、  
triplet np散乱パラメータ（eq18/eq19: `a_t,r_t`）を用いる。

**処理**：ERE（2次打ち切り）で束縛極 `k=iκ` を解き、`B.E._pred` を再構成する。  
eq18（GWU/SAID）と eq19（Nijmegen）の差は解析依存の系統proxyとして扱い、  
`B.E._obs` が予測包絡（min-max）に入るかで判定する。

**式**：

$$
k\cot\delta_t = -\frac{1}{a_t}+\frac{r_t}{2}k^2,\quad
\frac{r_t}{2}\kappa^2-\kappa+\frac{1}{a_t}=0,\quad
B_{\rm pred}=\frac{(\hbar c)^2\kappa^2}{2\mu c^2}.
$$

**指標**：`abs(B.E._pred-B.E._obs)/B.E._obs`、`ΔB`（keV）、  
および `B.E._obs` の包絡内判定（envelope pass）。

**結果**：

| dataset | a_t (fm) | r_t (fm) | κ (fm^-1) | B.E._pred (MeV) | ΔB (keV) | 相対誤差 |
|---|---:|---:|---:|---:|---:|---:|
| eq18（GWU/SAID） | 5.403 | 1.7494 | 0.23227 | 2.23740 | +12.831 | 0.577% |
| eq19（Nijmegen） | 5.420 | 1.7530 | 0.23146 | 2.22173 | -2.832 | 0.127% |

| 要約指標 | 値 |
|---|---:|
| `B.E._pred` 包絡 | [2.22173, 2.23740] MeV |
| `B.E._obs` | 2.224566 MeV（包絡内） |
| 系統proxy（half-range） | 7.83 keV |
| `z_total_proxy` | 0.64 |
| 判定 | envelope pass |

`output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_verification.png`  
左図は eq18/eq19 から得た `κ` と `B.E._pred` の差を示し、  
`B.E._obs` が予測包絡内に入ることを可視化する。  
ここでの差は測定統計より解析手順（eq18/eq19）の違いが支配的である。

`output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body.png`  
2体境界条件 `k cot(kR)=-κ` のI/Fを半径スケール別に示す。  
`R=λ_π` と `R=r_d` を含む代表ケースで、必要井戸深さのスケール感を固定する。  
核力の第一原理導出ではなく、観測I/Fの運用境界条件として用いる。

**考察**：重陽子では `Δω→B.E.` 写像の入口が観測量と整合し、  
eq18/eq19 の解析差を系統proxyとして分離できる。  
このため次段は、同一I/Fを保ったまま He-4 と軽核系統へ拡張する。

**注意**：本節は ERE 2次打ち切りによる操作的検証であり、  
QCD 第一原理の同値性は主張しない。`v2_t` など高次項は次段の系統評価で扱う。

**棄却条件**：`B.E._obs` が予測包絡に入らない（envelope fail）場合、この写像（このERE打ち切りI/F）は棄却する。  

##### 4.2.7.5 波動干渉理論の検証（He-4）

**目的**：重陽子で固定した `Δω→B.E.` 写像を He-4 へ拡張し、  
多体取り込み（collective / pair-wise）の違いを同一I/Fで比較固定する。

**入力**：観測 `B.E._obs=28.295610855±0.000001624 MeV`、`r_rms=1.6785±0.0021 fm`、  
重陽子側で凍結した `R_ref=1/κ_d=4.3177 fm`、`J_ref=B_d/2=1.1123 MeV`、`L=λ_π=1.4138 fm`。

**処理**：`R_alpha=√(5/3) r_rms` を用いて `J_E(R_alpha)` を評価し、  
`B.E._pred=2 C J_E(R_alpha)` で He-4 の結合を再構成する。  
`C` は `collective(A-1)=3`、`pn_only(ZN)=4`、`pairwise_all(A(A-1)/2)=6` を比較する。

**式**：

$$
R_\alpha=\sqrt{\frac{5}{3}}\,r_{\rm rms},\quad
J_E(R)=J_{\rm ref}\exp\!\left(\frac{R_{\rm ref}-R}{L}\right),\quad
B_{\rm pred}=2CJ_E(R_\alpha),\quad
\left(\frac{B.E.}{A}\right)_{\rm pred}=\frac{B_{\rm pred}}{4}.
$$

**指標**：`abs(B.E._pred-B.E._obs)/B.E._obs`、`B.E./A`、  
および半径不確かさ由来の `z_from_r_alpha_only`。

**結果**：

| method | C | B.E._pred (MeV) | B.E._pred/A (MeV) | ΔB (MeV) | 相対差 |
|---|---:|---:|---:|---:|---:|
| collective | 3 | 30.5512 | 7.6378 | +2.2556 | +7.97% |
| pn_only | 4 | 40.7349 | 10.1837 | +12.4393 | +43.96% |
| pairwise_all | 6 | 61.1024 | 15.2756 | +32.8068 | +115.94% |

| 要約指標 | 値 |
|---|---:|
| `B.E._obs/A` | 7.0739 MeV |
| `C_required`（`B.E._obs/(2J_E)`） | 2.7785 |
| 最小差分method | collective（`C=3`） |
| 判定 | pair-wise 加算は過大; collective が最も近い |

`output/quantum/nuclear_binding_energy_frequency_mapping_alpha_verification.png`  
左図で `B.E._obs` と各 reduction rule の `B.E._pred` を比較し、  
pair-wise 系が系統的に過大評価することを示す。  
右図は `C_required` と候補 `C` の関係を示し、collective 近傍が最小残差となる。

**考察**：He-4 では `C_required≈2.78` となり、`A-1=3` に近い。  
この結果は、単純 pair-wise 足し上げよりも、  
collective な縮約が有効I/Fとして自然であることを示す。

**注意**：この結論は「第一原理で collective を導出した」ことを意味しない。  
本節は重陽子で凍結した写像の外挿診断であり、  
次段で軽核 `A=2–16` の系統性を追加監査する。

**棄却条件**：半径不確かさのみの proxy で `abs(z_from_r_alpha_only)>3` となる場合、この写像外挿は棄却する。  

##### 4.2.7.6 波動干渉理論の検証（軽核系統性; A=2–16）

**目的**：重陽子/He-4 の点検を、軽核側の系統（`B.E./A vs A` と残差A依存）へ拡張し、  
どの質量数帯で同一I/Fが崩れ始めるかを固定する。

**入力**：軽核（d/t/h/α）の束縛エネルギー観測値と、代表核種での予測比較（collective/pn_only/pairwise）。

**出力（監査用）**：
- 軽核（A=2,3,4）の観測値テーブルと、代表核種の予測比較（集計メトリクス）。
（再現に用いるファイル一覧は 8章にまとめる。）

**処理**：観測側は `A=2,3,4` の `B.E./A` を固定し、  
予測側は collective（`C=A-1`）で `A=2,4,12,16` の  
`ratio=B_pred/B_obs` と `log10(ratio)` を比較する。

**指標**：`B.E./A`、`ratio=B_pred/B_obs`、`log10(ratio)`、`ΔB`。

**結果**：

| 核種 | A | B.E._obs (MeV) | B.E._obs/A (MeV) |
|---|---:|---:|---:|
| d | 2 | 2.2246 | 1.1123 |
| t | 3 | 8.4818 | 2.8273 |
| h | 3 | 7.7180 | 2.5727 |
| alpha | 4 | 28.2956 | 7.0739 |

| A（collective） | ratio `B_pred/B_obs` | log10(ratio) | ΔB (MeV) |
|---:|---:|---:|---:|
| 2 | 1.0000 | +0.0000 | +0.0000 |
| 4 | 1.0827 | +0.0345 | +2.3393 |
| 12 | 0.5899 | -0.2292 | -37.7944 |
| 16 | 0.4714 | -0.3267 | -67.4655 |

`output/quantum/nuclear_binding_light_nuclei.png`  
`A=2–4` の観測 `B.E./A` が単調増加する基準線を固定する。  
He-4 への急増は、単一2体スケールだけでは説明が難しい領域を示す。  
次段の軽核系統比較で、どのI/F不足が支配的かを切り分ける基準になる。

`output/quantum/nuclear_binding_energy_frequency_mapping_representative_nuclei.png`  
同一 collective I/F で `A=2,4,12,16` を並べると、  
`A=4` で軽い過大、`A>=12` で過小へ転じるA-trendが現れる。  
この符号反転は、距離proxyや配位数写像のA依存不足を示す運用指標となる。

**考察**：軽核側の観測 `B.E./A` は急増し、  
collective I/F は He-4 近傍までは近いが、`A=12,16` で過小へ反転する。  
したがって「同一写像で全Aを説明する」主張は現段で採用せず、  
A依存補正（距離proxy/配位数/殻効果）の分離監査を継続する。

**注意**：本節は `A=2–16` のうち、現時点の固定出力で再現できる  
代表点（2,3,4,12,16）での運用監査である。  
全核種評価は 5章の差分予測・反証条件で扱う。

**棄却条件**：代表点での外挿診断も含め、反証条件パック（8章の一覧）で固定した 3σ運用条件を満たさない場合、この最小写像は棄却する（追加物理が必要）。  

##### 4.2.7.7 図表パック（A–E）

**目的**：4.2.7（理論入口）と 5.4（差分予測）を結ぶ最小図表セットを  
図A–Eとして固定し、概念→準位→全核種曲線→残差→差分判定の導線を明示する。

**入力**：図A〜E（概念・準位条件・全核種曲線・残差監査・差分予測）の固定図。

**処理**：  
上記5図を再生成し、4.2.7 と 5.4 で参照する図の役割を  
図A（概念）→図B（準位/境界）→図C（全核種曲線）→図D（残差監査）→図E（差分予測）へ固定した。

**指標**：  
図役割の重複有無（A–Eの一意性）と、5.4 の反証条件（`σ_req`）まで  
読者が単一路線で到達できるかを確認する。

**結果**：  
図A：`output/quantum/nuclear_near_field_interference_two_mode_model.png`  
近接場干渉の最小I/F（`Δω=2J(R0)`, `B.E.=ħΔω`）を固定する。  
左は `J_E(R)` の包絡、右は分裂エネルギー `E_split=2J_E` を示し、  
「遠距離場ではなく近接場重なりで核結合を扱う」入口を明確化する。

図B：`output/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body.png`  
2体境界条件 `k cot(kR)=-κ` を満たす準位条件を、代表半径ごとに可視化する。  
`R=λ_π`, `R=r_d`, `R=1/κ` で必要井戸深さのスケールがどう変わるかを示し、  
重陽子（2核子系）で使う操作的I/Fを固定する。

図C：`output/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.png`  
AME2020全核種で `B_pred/B_obs` のA依存を俯瞰し、基底写像の大域傾向を示す。  
同図の分布パネルで、magic/parity群を含む残差の初期状態を固定し、  
追加物理を入れる前の基準線（baseline）として使う。

図D：`output/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.png`  
距離proxyを global `R` と local spacing `d` で差し替えたときの残差を比較する。  
局所proxyでA-trendがどこまで軽減されるかを直接監査し、  
残差の主因が「距離I/F」か「二次構造（pairing/shell）」かを分離する。

図E：`output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification.png`  
標準proxyとの差分 `ΔB`、`σ_req=abs(ΔB)/3`、`σ_req,rel` を核種ごとに固定する。  
5.4 の top核種選別とカテゴリ閾値（light/mid/heavy）をこの図に接続し、  
反証条件パックへの入力図として運用する。

**考察**：図A–Eの固定により、4.2.7 の理論写像と 5.4 の反証条件が  
同一I/Fの連続線として読める状態になった。  
これにより、追加自由度の導入有無を、図C/D/Eのどこで改善が出るかで  
段階的に判定できる。

**注意**：本パックは「導線固定」のための最小セットであり、  
第一原理QCD計算の代替を主張するものではない。  
採否判定は 5.4 の 3σゲートと独立cross-check束で行う。

**棄却条件**：判定は 5.4 の 3σゲート（`σ_rel(total)<=σ_req,rel`）と、反証条件パック（8章の一覧）を正とする。  

**凍結**：距離I/F（local spacing `d`＋`ν`飽和）による `Δω→B.E.` 写像、全核種の残差面（Z–N残差マップ）と差分予測（`ΔB`,`σ_req`）を、図A–Eの導線として固定した。  
**棄却**：同一写像が 3σ運用指標（z）と精度ゲート（`σ_rel(total)<=σ_req,rel`）を満たさない場合、この最小写像は棄却され、殻閉殻/相関（magic）や pairing を追加自由度として分離監査する必要がある。  
**次**：公開待ちに依らず、magic 近傍で支配的に残る残差（外れ率）を、pairing/shell の最小項でどこまで落とせるかを同一I/Fで追加し、非magic（変形核）との対比を維持したまま更新する。

#### 4.2.8 原子・分子（原子スペクトル基準値：H I, He I；分子定数：H2/HD/D2）

**目的**：原子・分子の束縛・準位を P-model から第一原理導出したとは主張しない。  
代わりに、分光の **基準値（ターゲット）**を一次固定し、将来の再導出と棄却判定へ接続する。

**入力**：原子・分子の分光基準値（一次ソース）。
  - 原子：NIST ASD（H I / He I）と NIST AtSpec handbook（H I 21 cm hyperfine）。[NISTASDLines][NISTAtSpecHandbook]
  - 分子定数：NIST WebBook（Huber & Herzberg compilation; H2/HD/D2）。[NISTWebBookDiatomic]
  - 解離エネルギー D0（0 K）：高精度分光（ionization→D0）。[ArXiv1902_09471][ArXiv2202_00532][Holsch2023HD]
  - 分子遷移（一次線リスト）：ExoMol（H2/HD）および MOLAT（D2）。[ExoMolRACPPK][ExoMolADJSAAM][MOLATD2ARLSJ1999][ARLSJ1999]
  - 同位体質量：NIST Atomic Weights and Isotopic Compositions（Hydrogen）。[NISTIsotopicCompositions]


**指標**：本節で固定したターゲット（λ, ν̃, A, ωe, B_e, D0, ΔfH°gas 等）を正とし、将来の再導出ではターゲットからの偏差（z-score など）で棄却判定へ接続する。  

**結果**：

原子基準線（vacuum λ）：

| 種 | ターゲット | 値 |
|---|---|---:|
| H I | Lyα | 121.567 nm |
| H I | Hγ | 434.169 nm |
| H I | Hβ | 486.266 nm |
| H I | Hα | 656.466 nm |
| H I | Hα multiplet（obs, E1） | 656.452–656.466 nm（N=5） |
| H I | Hβ multiplet（obs, E1） | 486.264–486.270 nm（N=3） |
| H I | 21 cm hyperfine（f） | 1420.4057517667(10) MHz（λ≈21.106114 cm） |
| He I | representative lines | 447.273 / 501.708 / 587.725 / 668.000 nm |

分子定数（ground state; NIST WebBook）：

| 分子 | ωe (cm⁻¹) | ωe x_e (cm⁻¹) | B_e (cm⁻¹) | α_e (cm⁻¹) | D_e (cm⁻¹) | r_e (Å) |
|---|---:|---:|---:|---:|---:|---:|
| H2 | 4401.213 | 121.336 | 60.8530 | 3.0622 | 0.0471 | 0.74144 |
| HD | 3813.15 | 91.65 | 45.655 | 1.986 | 0.02605 | 0.74142 |
| D2 | 3115.5 | 61.82 | 30.4436 | 1.0786 | 0.01141 | 0.74152 |

分光学的解離エネルギー D0（0 K; spectroscopic）：

| 分子 | D0 (cm⁻¹) | D0 (eV) | 参照 |
|---|---:|---:|---|
| H2（ortho; N=1） | 35999.582834(11) | 4.46338 | [ArXiv1902_09471] |
| HD（N=0） | 36405.78253(7) | 4.51374 | [Holsch2023HD] |
| D2（N=0） | 36748.362282(26) | 4.55622 | [ArXiv2202_00532] |

分子遷移（一次線リスト）：ExoMol line list（H2/HD）から、Einstein A の大きい順に top10 を抽出し、波数 ν̃ と A を基準値として固定する（選別規則を固定して恣意性を避ける）。D2 の ExoMol line list は本稿の作業環境では確認できていないため、別ソース（MOLAT）で補う。[ExoMolRACPPK][ExoMolADJSAAM]

（表記）本稿では vacuum wavelength を用いる（air/vacuum の変換は別途の系統として扱う）。

**同位体依存（縮約質量スケーリング；参考図）**：
WebBook の分子定数（H2/HD/D2）は、縮約質量 μ に対し ωe∝μ^{-1/2}、B_e∝μ^{-1} のスケーリングで整合する
（NIST 同位体質量の比：|1−ratio|≲10^-3）。
これは「波による束縛」の最低限の系統であり、P-model で分子束縛を再導出する際にも、この同位体系統を破らないことが
必要条件になる（追加自由度の恣意fitを避けるための制約として利用する）。

**熱化学（解離エンタルピー；298 K）**：
分子の「結合エネルギー」について、分光定数とは独立な一次ソースとして、NIST WebBook の Gas phase thermochemistry
（ΔfH°gas）を用い、解離エンタルピー（298 K）を固定する（本稿の目的は「基準値の固定」であり、
D0（0 K の分光学的解離エネルギー）を置換しない）。[NISTWebBookThermo]
結果（NIST WebBook；eV per molecule / kJ/mol）：

| 分子 | 解離（298 K）eV | kJ/mol |
|---|---:|---:|
| H2→2H | 4.5188 | 435.996 |
| HD→H+D | 4.5540 | 439.398 |
| D2→2D | 4.5959 | 443.44 |

`output/quantum/atomic_hydrogen_baseline.png`  
H I の代表線（Lyα〜Hα）と multiplet をまとめて示す。  
本節の分光ターゲット（vacuum λ）を固定する。

`output/quantum/atomic_helium_baseline.png`  
He I の代表線を示す。  
多電子系の「基準値固定」の入口として用いる。

`output/quantum/molecular_isotopic_scaling.png`  
H2/HD/D2 の分子定数が縮約質量 μ のスケーリングで整合することを示す。  
将来の再導出で破ってはならない同位体系統（必要条件）を可視化する。

`output/quantum/molecular_dissociation_d0_spectroscopic.png`  
高精度分光から得た D0（0 K）の基準値を示す。  
熱化学（298 K）との独立性も含め、結合エネルギーのターゲットとして固定する。

`output/quantum/molecular_transitions_exomol_baseline.png`  
ExoMol/MOLAT の一次線リストから抽出した代表遷移を示す。  
選別規則（top-N）を固定し、恣意的な line picking を避ける。

**注意**：  
（次段階）分子（化学結合）を含む本体は、一次データの拡張（遷移・結合エネルギー・同位体依存）と、QED精密との整合（安全確認）を保ったまま「追加自由度を最小」にして再導出へ進む必要がある。

**凍結**：原子（H I/He I）と分子（H2/HD/D2）の分光・結合の基準値を「ターゲット」として一次固定し、同位体系統（縮約質量スケーリング）と熱化学（298 K）を独立cross-checkとして凍結した。  
**棄却**：将来の再導出がこれらターゲットから総合不確かさの 3σ を超えて系統的に外れる、または同位体系統（ωe は縮約質量の平方根に反比例、B_e は縮約質量に反比例）を破るなら、その再導出は棄却される（恣意的な line picking は不可）。  
**次**：未確定の一次線リスト（D2 など）を一次ソースで補完し、ターゲット集合を拡張した上で、最小自由度での再導出と QED 精密（4.2.6）との整合を同時に満たす ansatz を探索する。

---

#### 4.2.9 物性（凝縮系；Si格子定数の基準値）

凝縮系（4.2.9–4.2.14）は、本文では「結論に必要な最小セット（ベースライン＋棄却条件）」を先に要約し、詳細は各節（および補遺）へ分離する。

| 項目（凝縮系） | 凍結した基準（代表値） | 棄却条件（要点） |
|---|---|---|
| Si 格子定数/格子間隔 | a=5.431020511 Å；d(220)=1.920155716 Å | `abs(a_pred−a_target)>3σ_a` または `abs(d_pred−d_target)>3σ_d` で棄却 |
| Si 比熱 Cp°(T)（solid/liquid; Shomate） | Cp°(298.15K)=19.99；Cp°(1000K)=26.32；Cp°(2000K; liquid)=27.20（J/mol·K） | `abs(Cp_pred−Cp_target)>3σ_proxy`（σ_proxy=max(0.02Cp,0.2)）で棄却 |
| Si 低温 Cp°(T)（JANAF）＋Debye | θ_D=622.1 K（fit）を代表値として凍結 | 低温域での Cp°(T) が表の不確かさスケールで外れる主張は棄却 |
| Si 抵抗率 ρ(T) と温度係数 | 室温近傍で `d ln ρ / dT > 0`（必要条件）＋包絡（0.0205–0.957 %/K） | `d ln ρ / dT ≤ 0` は棄却；包絡外の主張は一次不確かさ根拠が無い限り棄却 |
| Si 熱膨張 α(T) | 符号反転 T≈123.7 K（凍結）＋fit誤差スケール | `α≈A·Cv` の定数モデルは符号不一致で棄却；予測主張は代表点で `abs(α_pred−α_target)>3σ_fit` なら棄却 |
| OFHC copper κ(T)（RRR依存） | κ(4K)～κ_max～κ(300K) の曲線（RRR=50–500） | 代表点で `abs(κ_pred−κ_target)>3σ_fit`、かつ RRR 依存の順序を破る主張は棄却 |

**目的**：物性の最初の到達点として、固体の幾何学的基準値（格子定数）を一次固定し、  
今後の再導出と棄却判定へ接続する。

**入力**：CODATA 2022（NIST Cuu）の Si 格子定数 a と、理想 Si(220) 格子間隔 d。

**処理**：一次ソース（NIST Cuu）の HTML をローカルへ固定し、抽出値をそのままターゲットとして凍結する。  
追加のフィットは行わない（固定値の転記ミス検出を主目的とする）。

**指標**：a と d（Si(220)）。  
棄却条件：第一原理計算などがこの値を「予言する」と主張する場合、`abs(a_pred−a_target)>3σ_a`  
または `abs(d_pred−d_target)>3σ_d` なら棄却する（σはCODATAの不確かさ）。

**結果**：

| 量 | 値 | σ |
|---|---:|---:|
| Si 格子定数 a | 5.431020511 Å | 8.9×10^-8 Å |
| Si (220) 格子間隔 d | 1.920155716 Å | 3.2×10^-8 Å |

`output/quantum/condensed_silicon_lattice_baseline.png`  
Si の a と d（Si(220)）を、以後の再導出が参照する “ターゲット値” として並べて固定する。  
誤差（σ）は比較基準のスケールであり、fit で吸収してはならない量として扱う。

**注意**：この節は物性ターゲットの固定であり、現段階で P-model が a を第一原理で予言したことを主張しない。

---

#### 4.2.10 物性（凝縮系；Si固体/液体の比熱 Cp(T)（Shomate））

**目的**：物性の “比熱” の最小ターゲットとして、Cp(T) の基準曲線を一次データから固定し、  
将来の再導出（最小自由度）と棄却判定へ接続する。

**入力**：NIST Chemistry WebBook の condensed phase thermochemistry（Si; Thermo-Condensed；Shomate 係数）。

**処理**：WebBook の Shomate 係数（solid/liquid）から Cp°(T) を再構成し、基準曲線として固定する。

**式**：

$$
Cp^\circ(T)=A + B t + C t^2 + D t^3 + \frac{E}{t^2},\qquad t\equiv T/1000.
$$

**指標**：代表温度での Cp°(T)（下表）と、phase 切替（solid→liquid）。  
棄却条件：WebBook が不確かさを明示しないため、保守的に `σ_proxy=max(0.02 Cp, 0.2 J/mol·K)` を採用し、  
`abs(Cp_pred−Cp_target)>3σ_proxy` なら棄却する（proxy であり、のちに一次不確かさが判明したら置換する）。

**結果**：

| 相 | T (K) | Cp° (J/mol·K) |
|---|---:|---:|
| solid | 298.15 | 19.99 |
| solid | 1000 | 26.32 |
| liquid | 2000 | 27.20 |

`output/quantum/condensed_silicon_heat_capacity_baseline.png`  
solid/liquid の Cp°(T) を Shomate 係数から再構成し、温度依存の形を固定する。  
298 K から高温へ滑らかに繋がる一方、低温側（Debye 領域）が欠落することも視覚的に示す。

**注意**：WebBook の condensed Cp 表は 298 K からであり、低温（Debye の T^3 領域）の一次固定は別途追加する必要がある。

---

#### 4.2.11 物性（凝縮系；Si低温比熱 Cp(T)（JANAF）＋Debye（θ_D））

**目的**：WebBook（298 K 以降）に対し、低温側（少なくとも 100–300 K）の Cp(T) を一次固定し、  
Debye の 1-parameter（θ_D）で “圧縮したターゲット” を固定することで、将来の再導出（最小自由度）と棄却判定へ接続する。

**入力**：NIST-JANAF table（Si-004；Cp° の一次表）。

**処理**：100–300 K の Cp°(T)（JANAF）を Debye の 1-parameter（θ_D）で最小二乗フィットする。  
低温では `Cp≈Cv` とみなし、Cv の Debye 式を Cp の proxy として扱う。

**式**：

$$
C_V(T;\theta_D)=9R\left(\frac{T}{\theta_D}\right)^3
\int_0^{\theta_D/T}\frac{x^4 e^x}{(e^x-1)^2}\,dx.
$$

**指標**：Debye θ_D と、残差散らばりから推定した `σ_proxy`。  
棄却条件：JANAF に明示σがないため proxy を用い、`abs(\theta_{D,\mathrm{pred}}-\theta_{D,\mathrm{target}})>max(3σ_proxy, 5 K)` なら棄却する。

**結果**：

| 指標 | 値 |
|---|---:|
| Cp°(100 K) | 7.268 J/mol·K |
| Cp°(200 K) | 15.636 J/mol·K |
| Cp°(298.15 K) | 20.000 J/mol·K |
| Debye θ_D（100–300 K；Cv≈Cp proxy） | 622.1 K（σ_proxy≈17.4 K） |
| Cross-check（298.15 K；WebBook−JANAF） | −0.0094 J/mol·K |

`output/quantum/condensed_silicon_heat_capacity_debye_baseline.png`  
JANAF（100–300 K）を Debye（θ_D）1-parameter で近似し、θ_D をターゲットとして凍結する。  
WebBook との 298 K cross-check も併記し、データ整合を確認する。

---

#### 4.2.12 物性（凝縮系；Si抵抗率 ρ(T) と室温近傍の温度係数）

**目的**：電磁気の採用宣言と整合する範囲で、固体の “輸送” の代表値として ρ(T) を一次固定し、  
室温近傍の温度係数（d ln ρ / dT）を **観測ターゲットとして固定**する（将来の再導出と棄却判定へ接続）。

**入力**：NBS IR 74-496（Legacy IR；Appendix E の ρ(T)（0–50°C））。

**処理**：表から ρ(20°C), ρ(23°C), ρ(30°C) を抽出し、室温近傍の係数を有限差分で proxy 評価する。  
ドーピング/タイプで分布が大きく変わるため、ここでは「精密な±σ」ではなく「必要条件の包絡」を凍結する。

**式**：

$$
\left.\frac{d\ln\rho}{dT}\right|_{\mathrm{23^\circ C}}
\approx \frac{\ln\rho(30^\circ\mathrm{C})-\ln\rho(20^\circ\mathrm{C})}{10\,\mathrm{K}}.
$$

**指標**：ρ(23°C) と `d ln ρ / dT`（proxy）。  
棄却条件：この測定レンジでは係数は正であり、`d ln ρ / dT ≤ 0` は棄却（必要条件）。  
また、観測包絡（0.0205–0.957 %/K）を外れる主張は、まず一次不確かさの根拠を示さない限り棄却する。

**結果**：

| 指標 | 値 |
|---|---:|
| ρ(23°C)（n=21） | 7.761×10^-4〜1.159×10^3 Ω·cm |
| d ln ρ / dT（20–30°C; proxy） | 0.0205〜0.957 %/K |

`output/quantum/condensed_silicon_resistivity_temperature_coefficient_baseline.png`  
0–50°C の ρ(T) 抽出と、室温近傍の d ln ρ / dT（proxy）を散布として固定する。  
材料・ドーピング差で span が大きいことを “系統” として可視化する。

---

#### 4.2.13 物性（凝縮系；Si熱膨張係数 α(T)）

**目的**：格子定数・比熱に続き、長さスケールの温度応答（熱膨張）を一次固定し、  
将来の再導出と棄却判定へ接続する（phonon の自由度がどこへ入るかを閉じる入口）。

**入力**：NIST TRC Cryogenics（Material Properties: Silicon；熱膨張係数 α(T)）。

**処理**：NIST の経験式（fit）を用いて α(T) を 13–600 K で評価し、代表点（下表）をターゲットとして凍結する。  
ページ記載の fit 標準誤差（T<50 K と T≥50 K）を系統見積りとして併記する。

**指標**：代表温度での α(T) と、fit 標準誤差スケール。  
棄却条件：`abs(α_pred(T)−α_target(T))>3σ_fit(T)` が代表点で成立するなら棄却する（σ_fit はページ記載の標準誤差）。

**結果**：

| 指標 | 値 |
|---|---:|
| データ範囲 | 13–600 K |
| curve-fit標準誤差（相対；ページ記載） | 0.03×10^-8/K（T<50 K）, 0.5×10^-8/K（T≥50 K） |
| αの符号反転（負→正） | T≈123.7 K |

| T (K) | α (10^-8/K) |
|---:|---:|
| 20 | −0.34 |
| 50 | −29.22 |
| 100 | −33.88 |
| 293.15 | 255.47 |
| 600 | 383.91 |

`output/quantum/condensed_silicon_thermal_expansion_baseline.png`  
α(T) の符号反転（低温）から室温以上の増大までを一枚に固定する。  
fit の誤差スケール（ページ記載）も併記し、必要精度の目安を残す。

**考察**：一次固定した α(T) は負→正の符号反転を含むため、`α≈A·Cv` のように αが常に正になる定数モデルは棄却される。  
より精密な ansatz 試験（定数モデル評価を含む）と holdout 監査は、補遺Aへまとめる。

---

#### 4.2.14 物性（凝縮系；OFHC copper 熱伝導率 κ(T)）

**目的**：低温輸送の一次ターゲットとして、RRR（不純物散乱）依存の κ(T) 曲線を固定し、  
将来の再導出と棄却判定へ接続する。

**入力**：NIST TRC Cryogenics（Material Properties: OFHC Copper；κ(T) rational fit；RRR=50/100/150/300/500）。

**処理**：NIST が与える `log10 κ(T)` の rational fit を用いて κ(T) を計算し、共通温度グリッド（4–300 K）へ投影して凍結する。  
RRR ごとの fit 相対誤差（1–2%）を 1σ の proxy として扱い、代表点での棄却条件を定義する。

**式**：

$$
\log_{10}\kappa(T)=\frac{a + cT^{1/2}+ eT + gT^{3/2} + iT^2}
{1 + bT^{1/2}+ dT + fT^{3/2} + hT^2}.
$$

**指標**：κ(4 K), κ_max, peak温度（T_at_max）, κ(300 K) と、RRR 依存の単調性。  
棄却条件：`σ_fit≈(fit_error%)×κ` を用い、`abs(κ_pred−κ_target)>3σ_fit` が代表温度で成立するなら棄却する。

**結果**：

| 指標 | 値 |
|---|---:|
| κ(4 K) | 320（RRR 50）〜3182（RRR 500） W/(m·K) |
| κ_max | 1476（T≈26 K；RRR 50）〜7614（T≈14 K；RRR 500） W/(m·K) |
| κ(300 K) | 392〜401 W/(m·K)（RRR依存は小） |
| NIST fit 相対誤差 | 1–2% |

`output/quantum/condensed_ofhc_copper_thermal_conductivity_baseline.png`  
RRR ごとの κ(T) を同一温度グリッド（4–300 K）へ投影して比較する。  
低温ピーク（~14–26 K）の高さが RRR で増大し、室温では収束することを視覚的に固定する。

**凍結**：凝縮系（Si/Cu）では、格子定数・比熱・抵抗率・熱膨張・熱伝導率の一次ターゲット（代表点と誤差スケール）を固定し、再導出の“合否”を数値で判定できる入口を作った。加えて、温度帯 split の holdout（train→test 外挿）を α(T), Cp(T), κ(T) に適用し、外挿での残差増大を “手続き系統” として監査出力へ固定した（8章の監査サマリ）。  
**棄却**：これらのターゲットを満たさない主張（代表点での 3σ逸脱、必要条件の破れ、包絡外れ）は棄却される。特に Si の α(T) は符号反転を含むため、`α≈A·Cv` の定数モデルは全域で棄却される。  
**holdout**：温度帯 split の holdout（train→test 外挿）では、運用ゲートとして test 側の `max |z|≤3`（補助指標として `reduced χ²≤9`）を採用し、これを満たす例を holdout-ok と呼ぶ。最小の Debye+Einstein（2枝）だけだと Si α(T) が崩壊し（test max |z|≈15–45）、レンジ選択が支配的な系統になり得ることが分かる。  
一方で、phonon DOS の重みを凍結し、mode-dependent ω(T)（INS; Kim2015 Fig.2）と体積弾性率 B(T) を一次拘束として取り込んだ低次元 γ(ω) 基底（3係数＋固定重なり）では、strict と holdout の両方が成立する例が得られる（test max |z|≈1.9–2.5、reduced χ²≤2）。  
Si Cp(T) は Debye 1p / Shomate 5p の単独では holdout で崩壊し（test max |z|≈6–8、Shomate は低温側で max |z|≈90）、レンジ選択が支配的な系統になり得る。  
一方で、低温帯（100–300 K）の Debye θ_D を max |z| 最小で凍結し、高温帯（298–1685 K）は WebBook の Shomate 係数を一次拘束として凍結した “hybrid frozen” では、両 split で test max |z|≤3 を満たす（holdout-ok）。  
Cu κ(T) は ansatz class の違いが holdout 合否として現れ、rational は安定だが log-polynomial は低温ピーク外挿で崩壊する。  
**次**：物性/熱の cross-check（α(T) と Cp(T) の整合、κ(T) の RRR 依存など）を保ったまま、一次拘束を増やした最小 ansatz を探索し、pass/fail の形で監査出力へ固定する（詳細ログは補遺A）。

---

#### 4.2.15 統計力学・熱力学（黒体放射の基準量）

**目的**：温度・平衡分布（熱統計）を P-model と接続する前提として、  
黒体放射の基準量（σ, a, nγ∝T^3）を一次ソース（SI/CODATA）で固定し、拡張仮説の棄却条件へ接続する。

**入力**：NIST Cuu（CODATA）定数（c, h, k_B, σ）と、数学定数 π, ζ(3)。

**処理**：黒体放射の標準式から σ を計算し、放射定数 a と光子数密度係数を導出して固定する。

**式**：

$$
\sigma=\frac{2\pi^5 k_B^4}{15\,h^3 c^2},\qquad
u(T)=aT^4,\quad a=\frac{4\sigma}{c},\qquad
n_\gamma(T)=\frac{2\zeta(3)}{\pi^2}\left(\frac{k_B T}{\hbar c}\right)^3.
$$

**指標**：σ（基準量）、a（エネルギー密度スケール）、nγ の係数（T^3 の前係数）。

**結果**：

| 指標 | 値 |
|---|---:|
| σ | 5.670374419e-8 W/(m^2 K^4) |
| a | 7.56573325e-16 J/(m^3 K^4) |
| nγ 係数 | 2.0286846e7 1/(m^3 K^3) |
| ⟨E⟩/(k_B T) | 2.70118 |

`output/quantum/thermo_blackbody_radiation_baseline.png`  
左：エネルギー密度 u(T) の T^4 スケーリング。右：光子数密度 nγ(T) の T^3 スケーリング。  
今後の拡張（P の揺らぎや結合の仮説）がこの基準量・スケーリングと矛盾すれば棄却される。

**凍結**：黒体放射の基準量（σ, a, nγ∝T^3）と、`⟨E⟩/(k_B T)` を一次定数から導出して基準量として凍結した。  
**棄却**：P-model の拡張が、平衡黒体の `T^4/T^3` スケーリングや上の基準量を総合不確かさの 3σ を超えて系統的に破るなら、その拡張は棄却される。  
**次**：温度や P の揺らぎに関する拡張を導入する場合も、まずこの黒体基準を温度の校正点として維持し、非平衡（スペクトル歪み等）を主張するなら観測可能な追加パラメータとエネルギー収支を併記する。

---

#### 4.2.16 統計力学・熱力学（黒体：エントロピーと第2法則整合）

**目的**：温度とエントロピーの操作的基準として、  
黒体（光子気体）の平衡熱力学 `u(T), p(T), s(T)` と `s/(nγ k_B)=const` を固定し、  
以後の P-model 拡張が第2法則や平衡スケーリングと矛盾する場合の棄却条件を与える。

**入力**：SI/CODATA 定数（c, h, k_B）と、数学定数 π, ζ(3)。

**処理**：`u=aT^4` を起点に `p=u/3`, `s=(4/3)aT^3` を導出し、  
光子数密度 `nγ(T)` と比 `s/(nγ k_B)` が温度に依らず一定であることを基準量として固定する。

**式**：

$$
u=aT^4,\quad p=\frac{u}{3},\quad s=\frac{4}{3}aT^3,
\qquad
n_\gamma=\frac{2\zeta(3)}{\pi^2}\left(\frac{k_B T}{\hbar c}\right)^3,
\qquad
\frac{s}{n_\gamma k_B}=\frac{2\pi^4}{45\zeta(3)}.
$$

**指標**：`s/(nγ k_B)`（無次元）と、`u,p,s,nγ` の温度スケーリング（T^4, T^4, T^3, T^3）。

**結果**：

| 指標 | 値 |
|---|---:|
| s/(nγ k_B) | 3.60157 |
| s(T) 係数（=(4/3)a） | 1.0087644e-15 J/(m^3 K^4) |

例（代表温度）：

| T (K) | u (J/m^3) | p (Pa) | s (J/(m^3 K)) | s/(nγ k_B) |
|---:|---:|---:|---:|---:|
| 2.725 | 4.17e-14 | 1.39e-14 | 2.04e-14 | 3.60157 |
| 77 | 2.66e-8 | 8.87e-9 | 4.61e-10 | 3.60157 |
| 300 | 6.13e-6 | 2.04e-6 | 2.72e-8 | 3.60157 |
| 5772 | 8.40e-1 | 2.80e-1 | 1.94e-4 | 3.60157 |

`output/quantum/thermo_blackbody_entropy_baseline.png`  
左：`s(T)=(4/3)aT^3` のスケーリングを、CMB〜室温〜太陽表面温度まで同一図で固定する。  
右：`s/(nγ k_B)` が温度に依らず一定になることを確認し、平衡光子気体の基準量として凍結する。

**考察**：本節は「P-model独自の温度定義」を導入する前に、  
“平衡の温度・エントロピーは黒体（光子気体）で校正できる” という操作的基準を先に固定する。  
したがって、P-model の拡張（例：P揺らぎや結合）が、平衡での `T^3/T^4` スケーリングや `p=u/3` を系統的に破るなら、  
それは第2法則（および既知の熱統計）と矛盾するため棄却される。

**注意**：ここでの棄却条件は「平衡熱力学のスケールを変えない」ことを要求するだけであり、  
非平衡・開放系（熱浴、測定、情報）までを含む一般論ではない。  
P-model が非平衡の新規効果を主張する場合は、別に測定可能な機構とエネルギー収支を提示する必要がある。

（物性/熱：反証条件パックの固定；監査用）
- 反証条件パック（JSON；8章の一覧）を正とする。
- 適用（監査条件）：各 baseline の metrics 出力に含まれる `falsification`（±3σ、±3σ_proxy、必要条件envelope 等）を棄却閾値として採用し、target の再フィットや閾値の事後変更を行わない。

**凍結**：黒体（光子気体）の平衡関係（`u=aT^4`, `p=u/3`, `s=(4/3)aT^3`, `nγ∝T^3`）と、無次元比 `s/(nγ k_B)` を操作的基準として凍結した。  
**棄却**：P-model の拡張が、平衡で `p=u/3`、`T^3/T^4` スケーリング、`s/(nγ k_B)=const` を総合不確かさの 3σ を超えて系統的に破るなら棄却される。  
**次**：非平衡・開放系まで扱う拡張を導入する場合は、測定可能な機構（ノイズ/散逸/情報）とエネルギー収支を定義し、まず平衡黒体基準と矛盾しない範囲で追加自由度を導入する。

---

## 5. 差分予測（Differential Predictions / Falsifiability）

本章では、Part III の量子テーマについて、差分予測（何が観測で区別可能か）と棄却条件（3σ）を固定し、  
将来のデータ更新で「どの精度とどの手続きが決着点になるか」を読者が追える形で示す。  
全体像は Table 2 に集約し、Bell と Born則については 5.1–5.2 で反証手続きを “パック” として形式化する。

Table 2: 量子テーマの差分予測と棄却条件サマリ

| テーマ | 観測量 | 標準QM予測 | P-model予測 | 現在精度 | 必要精度 | 棄却条件 |
|---|---|---|---|---|---|---|
| Bell（CHSH） | ∣S∣ | ideal 2√2≈2.828（LR上限2） | selection依存：∣S∣≈1.10–2.90（Weihs+Delft；手続きで変動） | σ_stat≈0.02–0.20；R_sel≈3.65–19.3（中央値7.60）；z_delay(dt)≈5.54（time-tagのみ） | 手続き系統≪統計（R_sel<1）、かつ min(∣S∣)−2 > 3σ_total | setting-independent な母集団（w_ab が a,b に依存しない）を満たす手続きが確立され、複数datasetで一貫して ∣S∣>2 が 3σ超で再現 → selection入口を棄却 |
| Bell（CH） | J_prob | J_prob>0（LR上限0） | selection依存：J_prob は window sweep で動く（NIST は 0 を跨ぐ／Kwiat は本範囲で跨がない） | σ_stat≈(0.9–3.9)×10^-5；Δ_sys≈(0.12–3.5)×10^-3；R_sel≈5.82–654（中央値≈330） | 手続き系統≪統計（R_sel<1）、かつ min(J_prob)−0 > 3σ_total | setting-independent 手続きの下で複数datasetで一貫して J_prob>0 が 3σ超で再現 → selection入口を棄却 |
| Born則逸脱（自己重力拡張） | χ≡(E_G/ħ)T | δ=0（Born） | χ≈(m/m_crit)^2；m_crit≡sqrt(ħ R/(G T)) | 現状：m≲10^8 amu級で χ≲10^-8（検出域外） | χ≳0.1–1（m≳0.3–1 m_crit）かつ位相/可視度誤差≲χ/3 | χ≳1 の領域で逸脱が 3σで見えない（上限が予測を下回る）→ 拡張仮説を棄却 |
| COW | Δφ | Δφ=−m g H^2/(ħ v0)（例：−70.09 rad） | 同（弱場）＋補正オーダー δφ/φ≈1.39×10^-9（例：δφ≈−9.75×10^-8 rad） | 位相の相対≈10^-2（想定） | 相対≲4.64×10^-10（3σ） | 現精度では差分検出不可；将来、相対≲5×10^-10 かつ大ΔU（MAQRO/ACES等）で不一致が 3σ超なら棄却 |
| 原子干渉計 | φ | φ≈k_eff g T^2（例：2.31×10^7 rad） | 同（弱場）＋δφ/φ≈1.39×10^-9（例：δφ≈−3.22×10^-2 rad） | 相対≈10^-9（想定） | 相対≲4.64×10^-10（3σ） | 現精度では差分検出困難；長自由落下/大ΔU（MAQRO/ACES等）で不一致が 3σ超なら棄却 |
| 量子時計 | Δf/f（or ε） | Δf/f≈ΔU/c^2（ε=0） | 同（弱場）；P−GR差分は δ(Δf/f)≈−6.05×10^-23（Δh≈399 m） | σ(Δf/f)≈2.89×10^-17 | 絶対≲2.02×10^-23（3σ） | ε（=0）からの逸脱が 3σ超、または長基線で δ(Δf/f) が 3σ超で不一致なら棄却 |
| de Broglie精密 | z（α整合） | z=0 | z=0 | z=0.588 | ∣z∣>3 が棄却域 | ∣z∣>3（独立αの不整合） |
| 重力誘起decoherence | V(t), σ_y | Pikovski式（ensemble dephasing） | 追加 σ_y が必要（例：σ_z=0.1/1/10 mmで V=0.5 に t≈4.0e4/4.0e3/4.0e2 s ⇒ σ_y≈1.09×10^-20/10^-19/10^-18） | 未検出 | σ_y を ≲10^-20–10^-18 に拘束 | V(t) の実測が固定した σ_y モデルと 3σ超で不一致（または独立制約で σ_y が必要値より十分小さい） |

### 5.1 Bell：棄却条件パック（operational falsification pack）

**目的**：Bell の “違反の有無” を結論にするのではなく、公開一次データに対して selection ノブ  
（coincidence window / offset / trial 定義）が統計量をどれだけ動かすかを系統として固定し、  
selection入口（手続き依存）による説明が棄却される条件を「運用閾値のパック」として形式化する。

**入力**：公開 time-tag（または event-ready）データ（Weihs 1998, NIST 2013, Kwiat/Christensen 2013, Delft 2015/2016）。  
[Zenodo7185335][NISTBelltestdata][IllinoisQIBellTestdata][DelftHensen2015][DelftHensen2016Srep30289]

**処理**：自然窓（natural window）は、dt ヒストグラムのピークに対して “偶発同時計数（accidental）の proxy” を統計的に定義し、  
背景（オフピーク）を差し引いた信号が 99% 以上（plateau fraction）を含む最小窓を推薦窓とする。  
推薦窓はグリッドへスナップして凍結し、窓/offset の掃引は最適化ではなく **系統幅 Δ_sys の評価**として扱う。

**式**：selection感度（系統/統計の比）
$$
\Delta_{\rm sys}(T)\equiv \max_k T(k)-\min_k T(k),\qquad
R_{\rm sel}\equiv \frac{\Delta_{\rm sys}(T)}{\sigma_{\rm stat,med}}
$$
ここで k は selection ノブ（window, offset, trial 定義など）、T は統計量（例：CHSH の ∣S∣、CH の J_prob）である。  
また遅延シグネチャ（setting依存の proxy）は、pairing 後の dt 分布に対して
$$
dt \equiv t_B-t_A-\mathrm{offset},\qquad
z_{\rm delay}\equiv \frac{abs(\Delta{\rm median})}{\sigma_{\Delta{\rm median}}}
$$
（Δmedian は setting0 と setting1 の median 差）として定義し、z_delay を「setting依存の検出強度」として固定する。

**指標**：  
R_sel（selection感度）、z_delay（遅延シグネチャ）、および cross-dataset の最小値（保守側）を “棄却条件パック” として採用する。  
運用閾値は R_sel≥1（系統幅が 1σ_stat 以上で動く）と z_delay≥3（3σ相当の検出）で固定する。

**結果**：

| dataset | 統計量 | selection ノブ | R_sel | z_delay（max） |
|---|---|---|---:|---:|
| Weihs 1998 | CHSH ∣S∣ | window | 19.3 | 5.54 |
| Delft 2015 | CHSH S | event-ready offset | 3.65 | n/a |
| Delft 2016 | CHSH S | event-ready offset | 7.60 | n/a |
| NIST 2013 | CH J_prob | window | 654 | 43.9 |
| Kwiat/Christensen 2013 | CH | window | 5.82 | n/a |

Table 3: Bell cross-dataset 統合（falsification packの要約；運用）

| 指標（cross-dataset） | 値（代表） | 意味（要点） |
|---|---:|---|
| R_sel（min/median/max） | 3.65 / 7.60 / 654 | 手続き系統（sweep幅）が統計誤差を上回る度合い（selection感度）。保守側は min を採用する。 |
| z_delay（dt; min） | 5.54 | fast-switch/time-tag の dt 定義で、setting 依存遅延（重み）の proxy が 3σ を超えること。 |
| profile corr rank（eps=1e-10） | 5 | sweep profile の cross-dataset 相関が “少数モード” で支配される度合い（共通の手続きモードの存在）。 |
| strongest profile corr pair | Weihs 1998 と NIST：corr≈−0.888 | 最も強い dataset 間の相関（手続きモードの共有を示す）。 |
| 15項目バジェット top（median） | offset 3.88、window 3.13、eff drift 0.35 | どの系統ノブが支配的か（中央値の規模）。 |
| item correlation top pair | window–accidental corr≈0.99995 | 誤差要因の間の強相関（独立性が成立しない組を示す）。 |
| pairing crosscheck max（Δ_impl/σ_boot） | 0.086 | 実装差（pairing規約等）による統計量の差分が統計誤差の 1σ 未満に収まる（実装依存の疑いを減らす）。 |

`output/quantum/bell/systematics_decomposition_15items.png`  
15項目の系統バジェット（中央値）と、dataset別の総和、項目間相関（ヒートマップ）を示す。  
“どのノブが支配的か” と “独立とみなせない組” を固定する。

`output/quantum/bell/cross_dataset_covariance.png`  
cross-dataset の sweep profile を共通グリッドへ射影し、共分散（相関）構造を可視化する。  
将来 dataset が追加されても、同一I/Fで統合（相関の更新）できる入口として用いる。

cross-dataset の selection感度は min(R_sel)=3.65、中央値 7.60（最大 654）であり、  
少なくとも本稿で扱う範囲では “selection ノブが統計量を動かす” こと自体は 1σ を大きく超えて定量化できる。  
また time-tag（photon）で z_delay を定義できるデータでは min(z_delay)=5.54（Weihs）、最大 43.9（NIST）であり、  
遅延分布の setting依存（proxy）は 3σ を超えて検出される。

`output/quantum/bell/falsification_pack.png`  
左：R_sel（selection感度）を dataset 横断で比較し、運用閾値（R_sel=1）に対してどれだけ上にあるかを固定する。  
右：z_delay（遅延シグネチャ）を定義できる dataset について、3σ閾値（z_delay=3）に対する余裕を示す。

`output/quantum/bell/longterm_consistency.png`  
Bell の “入口指標” を cross-dataset で俯瞰し、R_sel の下限（保守側）と z_delay の下限を固定する。  
これにより、将来 dataset が追加されたときも同一I/Fで pass/fail を更新できる。

**考察**：本稿の主張は「loophole の存在を新たに主張する」ことではなく、  
selection の定量的感度（どのノブでどれだけ動くか）を一次データで固定し、  
反証条件（どの手続きで棄却されるか）へ接続する点にある。  
したがって、Bell の反証は “違反の有無” ではなく、setting-independent な母集団が保証された手続きの確立と、その下での頑健再現に依存する。

**注意**：ここでの閾値（R_sel=1, z_delay=3）は、本リポジトリの運用閾値であり、普遍的な物理定数ではない。  
event-ready（Delft）や trial-based（Kwiat/Christensen）では z_delay の定義が同型にならないため n/a とし、  
可能な範囲で同一I/Fへ射影して更新する。

（反証条件パックの固定；監査用）
- 反証条件パック（JSON；8章の一覧）を正とする。
- 適用（監査条件）：pack に列挙された dataset に同一I/Fを適用し、natural window（plateau fraction 0.99；例外は override）と sweep グリッドを事後に変更しない。
- 棄却閾値（運用；pack の thresholds を正）：`selection_origin_ratio_min=1`、`delay_signature_z_min=3`（legacy: `ks_delay_min_legacy=0.01`）。
  - 補足：pack は cross-dataset 統合の summary（covariance/15系統/loophole）と参照出力も含め、監査入口を 1つの JSON に閉じた。

### 5.2 Born則逸脱：m_crit 閾値と将来実験（MAQRO/OTIMA）の精度要件

**目的**：Born則を第一原理導出したとは主張しない立場の下で、  
もし自己重力（平均場）などの拡張仮説を採用した場合に “どのスケールで差分が見えるか” を  
m_crit（閾値）と必要精度（3σ）へ落として固定する。

**入力**：波束サイズ R、干渉時間 T、質量 m（装置側の到達値）。  
分子干渉（OTIMA など）は m≲10^6–10^8 amu級、宇宙マクロ干渉（MAQRO のようなコンセプト）は  
長い T を取り得ることを想定する（具体設計は本稿の射程外）。

**処理**：自己重力の “効き具合” は、重力自己エネルギーのスケール E_G∼Gm^2/R を用いて  
無次元量 χ≡(E_G/ħ)T を制御パラメータとして扱う。χ≪1 では差分は実質ゼロ、χ∼1 で差分が顕在化し得る。  
3σ 検出の条件は（i）位相/可視度の測定誤差が χ/3 程度以下、（ii）カウント統計が 3/√N≲χ を満たすこととして固定する。

**式**：
$$
\chi \equiv \frac{E_G}{\hbar}T \sim \frac{Gm^2}{\hbar R}\,T
      =\left(\frac{m}{m_{\rm crit}}\right)^2,\qquad
m_{\rm crit}\equiv \sqrt{\frac{\hbar R}{G T}}
$$
カウント統計の 3σ 検出目安は
$$
abs(\delta)\gtrsim \frac{3}{\sqrt{N}}
$$
（δ は Born則の相対逸脱の代表量、N は有効試行数）である。

**指標**：m/m_crit（到達度）と χ（差分スケール）、および 3σ で必要な (N, σ_phase/σ_V) を採用する。

**結果**：

| シナリオ | R | T | m | m_crit | χ=(m/m_crit)^2 | 3σの要件（目安） |
|---|---:|---:|---:|---:|---:|---|
| OTIMA-like（例） | 1 nm | 1 ms | 10^8 amu | 7.6×10^11 amu | 1.7×10^-8 | σ_phase≲6×10^-9 rad かつ N≳3×10^16（現実的でない） |
| MAQRO-like（例） | 100 nm | 100 s | 10^10 amu | 2.4×10^10 amu | 0.17 | σ_phase/σ_V≲0.06 かつ N≳3×10^2 |
| MAQRO-like（閾値） | 100 nm | 100 s | 2.4×10^10 amu | 2.4×10^10 amu | 1 | σ_phase/σ_V≲0.33 かつ N≳9 |

本稿の立場では、現状の分子干渉は m≪m_crit（χ≪1）のため Born則逸脱は予測されず、  
直接検証の決着点は “χ を 0.1〜1 へ押し上げる（長い T と十分な m を両立する）” 宇宙系の実験に依存する。

**出力**：Born則の採用/逸脱条件のフローチャート図と、m_crit/χなどの指標出力。

**出力（監査用）**：
- Born則の採用/逸脱条件のフローチャート図（本文の図1）。
- m_crit/χ などの指標テーブル（機械可読）。
（再現に用いるファイル一覧は 8章にまとめる。）

**棄却条件**：χ≳1（m≳m_crit）の領域で、逸脱が 3σ で検出されない（上限が拡張仮説の予測を下回る）場合、その拡張仮説は棄却する。  

**注意**：χ は “自己重力が位相へ 1 rad 程度入る” スケール指標であり、実際の prefactor や観測量（位相か可視度か）は  
採用する拡張モデル（Schrödinger–Newton、環境結合、非マルコフ性など）に依存する。  
したがって本節の数値は、実験計画の設計変数（R, T, m, N）を差分予測へ接続するための基準として用いる。

### 5.3 原子核（Δω→B.E.）：反証条件パックの独立cross-check統合

**目的**：差分チャネル精度ゲート  
（`σ_rel(total)<=σ_req,rel`）を、独立観測（分離エネルギーと電荷半径kink）へ接続し、  
“チャネル差分だけで判定していない” ことを運用上の条件として固定する。

**入力**：差分チャネル（P-SEMF/P-Yukawa）の指標（差分規模・必要精度）と、独立cross-check指標（分離エネルギー gap と 半径 kink strict）。

**出力（監査用）**：
- 差分チャネル（P-SEMF/P-Yukawa）の差分規模と必要精度の定量結果。
- 独立cross-check（分離エネルギー gap と 半径 kink strict）の定量結果。
（再現に用いるファイル一覧は 8章にまとめる。）

**処理**：差分チャネル（P-SEMF/P-Yukawa）の `σ_req,rel` を維持したまま、  
独立cross-check束（Sn/S2n gap と Δ²r kink strict）を同一JSONへ束ね、  
ゲート条件へ `channel_gate_requires_independent_crosschecks` を追加する。

**式**：
$$
\sigma_{\rm req,rel}=\frac{abs(\Delta B)}{3B_{\rm obs}},\qquad
{\rm gate:}\ \sigma_{\rm rel}({\rm total})\leq \sigma_{\rm req,rel}
$$
独立cross-check側は、半径kink strict で
$$
\max\left(abs\left(\Delta^2 r_{\rm pred}-\Delta^2 r_{\rm obs}\right)/\sigma_{\Delta^2 r}\right)\leq 3
$$
を採用する。

**指標**：`z_median`、`z_Δmedian`、`median σ_req,rel`（P-SEMF/P-Yukawa）、  
および独立cross-checkの `Sn/S2n RMS` と `A_min=100` の半径kink strict pass/fail。

**結果**：global-R と local-d の両I/Fは中央値指標で fail を維持する一方、  
A-trend 指標は 3σ 内に残る。差分チャネルの必要相対精度は  
P-SEMF が約2.70%、P-Yukawa が約33.43%で、P-SEMF 側が高精度ゲートを支配する。  
独立cross-checkは、分離エネルギーの gap 指標が有効点を持ち、  
半径kinkは pairing 凍結で `A_min=100` strict pass（Sn≈0.688σ, Sp≈2.014σ）を満たす。

`output/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.png`  
上段は従来の z 指標（中央値とA-trend）を維持し、下段で差分チャネル精度要求と  
独立cross-check（Sn/S2n・半径kink）を同時に表示する。  
これにより、反証条件パックの判定が「単一チャネルのfit都合」に寄らないことを  
運用上のI/Fとして固定できる。

**考察**：本更新は閾値（3σ運用）自体を変更しない。  
判定に使う情報の束を増やして、チャネル差分の採用条件を厳密化した。  
したがって、将来の追加物理を評価する場合でも、  
同一の反証閾値と独立cross-check束で比較可能である。

**注意**：本節の cross-check は「十分条件」ではなく「必要ガードレール」である。  
独立cross-checkが通っても、差分チャネル側の `σ_rel(total)<=σ_req,rel` を満たさない核種群は  
3σ決着対象へ昇格しない（判定保留）。

（反証条件パックの固定；監査用）
- 反証条件パック（JSON；8章の一覧）を正とする。
- 適用（監査条件）：`z_median`/`z_Δmedian` の 3σ 閾値に加え、差分チャネルの precision gate `σ_rel(total)<=σ_req,rel` と、独立cross-check（Sn/S2n gap と Δ²r kink strict; A_min=100 pairing凍結）を conditions として固定する。
- 棄却閾値（運用；pack の thresholds を正）：`z_median_abs_max=3`、`z_delta_median_abs_max=3`（A-bin 固定）。

### 5.4 核力波動干渉：差分予測と棄却条件

**目的**：4.2.7.4〜4.2.7.6 で固定した `Δω→B.E.` 写像を、
標準理論proxy（SEMF, Yukawa距離proxy）との差分 `ΔB` と
必要精度 `σ_req=abs(ΔB)/3` に落とし、核種ごとの決着難易度を
5章の反証I/Fとして固定する。

**入力**：AME2020 全核種（n=3554）の差分定量結果と、決着点候補の上位核種リスト。

**出力（監査用）**：
- 全核種の差分定量（差分規模と必要精度、A帯別統計）。
- 決着点候補の上位核種リスト（top核種）。
（再現に用いるファイル一覧は 8章にまとめる。）

**処理**：
P-model（local spacing `d` + `ν_sat` 凍結）を基準に、
チャネル差分 `P-SEMF` と `P-Yukawa` を核種ごとに計算する。
つづいて 3σ 決着条件として `σ_req` と `σ_req,rel` を定義し、
全核種統計、A帯別統計、優先核種を同一I/Fで固定する。

**式**：
$$
\Delta B = B_{\rm pred}^{\rm P} - B_{\rm pred}^{\rm std},\qquad
\sigma_{\rm req,3\sigma}=\frac{\mathrm{abs}(\Delta B)}{3},\qquad
\sigma_{\rm req,rel}=\frac{\sigma_{\rm req,3\sigma}}{B_{\rm obs}}.
$$

**指標**：
`median abs(ΔB)`、`p84 abs(ΔB)`、`max abs(ΔB)`、`median σ_req,rel`。
運用上は heavy-A での `σ_req,rel` と top核種を優先度指標にする。

**結果**：

| 差分チャネル | median abs(ΔB) (MeV) | p84 abs(ΔB) (MeV) | max abs(ΔB) (MeV) | median σ_req,rel |
|---|---:|---:|---:|---:|
| P-SEMF | 75.15 | 230.77 | 472.66 | 0.0270 |
| P-Yukawa proxy | 1150.06 | 1866.18 | 2532.92 | 0.3343 |

| A帯 | P-SEMF median abs(ΔB) (MeV) | P-SEMF median σ_req,rel | P-Yukawa median abs(ΔB) (MeV) | P-Yukawa median σ_req,rel |
|---|---:|---:|---:|---:|
| light（A<=40） | 28.97 | 0.0583 | 136.86 | 0.2325 |
| mid（41<=A<=120） | 23.55 | 0.0114 | 636.95 | 0.3034 |
| heavy（A>=121） | 145.22 | 0.0340 | 1530.60 | 0.3542 |

| rank（P-SEMF） | 核種 | A | abs(ΔB) (MeV) | σ_req,rel |
|---|---|---:|---:|---:|
| 1 | Og | 295 | 472.66 | 0.0755 |
| 2 | Og | 294 | 470.44 | 0.0753 |
| 3 | Og | 293 | 469.62 | 0.0755 |
| 4 | Ts | 294 | 465.74 | 0.0745 |
| 5 | Ts | 293 | 463.34 | 0.0743 |

`output/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification.png`  
左上は差分 `abs(ΔB)` の全核種分布、右上は top核種のチャネル別寄与を示す。  
左下は必要絶対精度 `σ_req`、右下は必要相対精度 `σ_req,rel` の分布で、  
heavy-A 側が主要な判別ターゲットであることを固定する。

top20 リスト（再現用）では、P-SEMF と P-Yukawa のいずれも同一の超重核帯に集中し、  
現状の観測誤差 `σ_B_obs` は `σ_req` を下回る（運用I/F上は resolvable）ことを示す。  
したがって次段は測定統計そのものより、モデル系統と独立cross-check束の維持が支配的になる。

**考察**：
差分規模は軽核より重核で増幅し、特に `A>=121` で
`median abs(ΔB)` が P-SEMF 145 MeV、P-Yukawa 1531 MeV に達する。
このため、核力波動干渉の差分検証は「軽核の微小差」を追うより、
重核帯で `σ_req,rel` を満たす核種群を優先する方が運用効率が高い。

**注意**：
ここでの `σ_req` は 3σ 判定用の操作的しきい値であり、
第一原理EFT計算との差を直接同定する量ではない。
以下で、独立cross-checkを組み込んだ反証条件パックへ
この差分I/Fを接続する。

**反証条件パック（監査用）**：

**目的**：
5.4 の差分I/Fを、3σ運用閾値と独立cross-checkを含む
単一の判定束（falsification pack）として固定する。

**入力**：核（Δω→B.E.）の反証条件パック（3σ運用閾値・精度ゲート・独立cross-checkの束）。

**固定ファイル（監査用）**：
- 反証条件パック（JSON；8章の一覧）

**処理**：
まず残差ゲート（`abs(z_median)` と `abs(z_Δmedian)`）を固定し、
次にチャネル精度ゲート `σ_rel(total)<=σ_req,rel` を適用する。
最後に独立cross-check（分離エネルギー・半径kink）との整合を必須条件として接続する。

**指標**：
`abs(z_median)<=3`、`abs(z_Δmedian)<=3`、`σ_rel(total)<=σ_req,rel`、  
`radius_strict_pass_A100_pairing=true`、`separation_gap_available=true`。

**結果**：

| ゲート | 閾値 | global R | local d |
|---|---|---:|---:|
| median residual | `abs(z_median)<=3` | -3.22（fail） | 5.63（fail） |
| A-trend residual | `abs(z_Δmedian)<=3` | -2.61（pass） | 2.39（pass） |

| 核種カテゴリ | P-SEMF `median σ_req,rel` | P-Yukawa `median σ_req,rel` | 運用上の意味 |
|---|---:|---:|---|
| light（A<=40） | 0.0583 | 0.2325 | 軽核は系統確認帯（差分は中程度） |
| mid（41<=A<=120） | 0.0114 | 0.3034 | P-SEMFは高精度ゲート（約1.1%） |
| heavy（A>=121） | 0.0340 | 0.3542 | 差分最大帯。優先判別ターゲット |

| 独立cross-check | 指標 | 固定値 | 判定 |
|---|---|---:|---|
| 分離エネルギー | Sn RMS total (MeV) | 12.80 | gap指標利用可 |
| 分離エネルギー | S2n RMS total (MeV) | 23.43 | gap指標利用可 |
| 半径kink baseline | max abs σ（Sn/Sp） | 3.07 / 3.72 | fail |
| 半径kink pairing凍結 | max abs σ（Sn/Sp） | 0.69 / 2.01 | pass |

`output/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.png`  
上段は残差ゲート（median と A-trend）、下段はチャネル精度要求と
独立cross-check（分離エネルギー・半径kink）を同時表示する。  
これにより、差分予測の採否を「単一fitの都合」ではなく、
複数I/Fの同時満足として判定できる。

**考察**：
5.4 の差分規模は heavy-A で最大だが、判定は差分の大きさだけでなく
独立cross-check整合を満たすかで確定する。
したがって本稿の 3σ 判定は、「差分が大きい」ことの確認ではなく
「差分I/Fと独立I/Fが同時に矛盾しない」ことを要求する運用である。

**注意**：
`σ_proxy` は内部散らばりの運用量であり、実験統計誤差そのものではない。
本節の判定は運用I/Fの固定であり、最終的な棄却/採用は
次段階では、有効モデルI/Fとの接続を含めて評価する。

---

## 6. 議論（Discussion）

本章は 4.2.7 と 5.4 を接続し、波動干渉理論の意義・限界・
将来測定での決着条件を整理する。

### 6.1 波動干渉理論の位置づけ

**目的**：
有効モデルI/F（操作的写像）と第一原理解釈（波動干渉）の役割分担を固定し、
棄却判定までつながる運用条件を明示する。

**入力**：
4.2.7.4–4.2.7.7（重陽子/He-4/軽核系統/図A–E）と
5.4（差分予測・反証条件パック）を用いる。

**処理**：
グルーオン交換との対応は「置換」ではなく、
核子有効自由度で観測量へ投影した操作的同値として扱う。
具体的には、`J_E(R)` は QCD 直接計算の代替ではなく、
`Δω→B.E.` をデータI/Fへ接続する縮約量として固定する。
また Coulomb 項・詳細スピン構造・高次多体項は
コア写像の外に置き、独立cross-check（Sn/S2n、半径kink）で
系統として監査する。

**式**：
$$
\sigma_{\rm rel}({\rm total}) \leq \sigma_{\rm req,rel},\qquad
\sigma_{\rm req,rel}=\frac{\mathrm{abs}(\Delta B)}{3B_{\rm obs}}.
$$

**指標**：
`z_median`、`z_Δmedian`、`σ_req,rel`（light/mid/heavy）、
および独立cross-checkの pass/fail を同時に評価する。

**結果**：
P-SEMF チャネルで必要相対精度は
light=5.83%、mid=1.14%、heavy=3.40% となり、
決着は mid/heavy の高精度帯が支配する。
超重核上位（Og/Ts）では `σ_req,rel` が約 7.4–7.6% で、
差分規模 `abs(ΔB)` は十分大きい。
したがって実運用上のボトルネックは
測定統計よりも「モデル系統の凍結」と
独立cross-checkの同時充足である。

**考察**：
波動干渉理論は、核力を「近接場干渉による周波数低下」として
一貫語彙で扱える点に意義がある。
一方で Coulomb/多体補正を外部系統として管理するため、
第一原理QCDの完全置換を主張する段階ではない。
本稿の到達点は、追加物理の採否を
同一3σゲートで逐次判定できる実装I/Fを固定した点にある。

**注意**：
本節は宇宙論側の救済議論（DDR/BAO/H0）を扱わない。
それらは Out of Scope とし、Part II の対応節で別管理する。

---

## 7. 結論（Conclusion）

**核**：凍結値（写像のアンカー、ν飽和、effective potential の選択、pairing 系統など）は凍結パラメータ一覧へ集約して固定した。棄却条件（適用条件を含む）は反証条件パックを正とし、全核種残差と独立cross-check（分離エネルギー、電荷半径kink）を同一I/Fで監査できる形に閉じた（いずれも 8章の一覧）。

**Bell**：凍結値（natural window と運用閾値、共分散の評価手順）は凍結パラメータ一覧へ集約して固定した。棄却条件（適用条件を含む）は反証条件パックを正とし、selection 感度（sys）と bootstrap（stat）を分離したまま、どの手続きで棄却されるかを固定した（いずれも 8章の一覧）。

**物性/熱（および干渉/時計/QED精密の基準）**：凍結値は凍結パラメータ一覧へ集約して固定し、各baseline metrics の `falsification` を束ねた反証条件パックを棄却条件の正として閉じた。これにより、holdout/整合性の失敗が観測された場合に、どの項目で棄却されるかをプロトコルとして提示できる（いずれも 8章の一覧）。

---

## 8. データ/コードの入手性（Reproducibility / Availability）

本稿で用いたデータは公表されている一次ソースに基づき、出典は付録にまとめる。  
解析コードと処理条件は「同じ入力→同じ結果」が再現できる形で整理している。  
一次ソース（URL/参照日）は付録「データ出典（一次ソース）」にまとめ、処理条件（窓/外れ値規則/凍結値）は本文に明記する。

本稿（Part III）では、論文本文に具体的なファイル名・フォルダ名・スクリプト名を列挙しない。  
固定アーティファクト一覧（凍結パラメータ、反証条件パック、監査サマリ、公開マニフェスト）と再現コマンド、監査ゲートは検証用資料（Verification Materials）を正とする。[PModelVerificationMaterials]


---


