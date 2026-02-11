# Step 7.7：光の量子干渉（単一光子・HOM・スクイーズド光）

## 目的

Phase 7 では「時間＝波の媒質」という仮定を、量子現象へ接続するための **観測量の入口**を整備する。  
ここでの狙いは「量子の完全理論」を主張することではなく、量子光学の代表的な干渉実験について、

- 何を観測量（定義）として固定するか
- time-tag（到着時刻）や遅延（delay）が解析手順のどこへ入るか
- P-model 側で「時間構造のゆらぎ」が入るなら、どの量（位相・遅延・雑音）に現れ、どの形で反証されるか

を、一次ソースに基づき **再現可能な形**へ落とすことにある。

---

## 1) 単一光子干渉（Mach–Zehnder）

干渉計の一方の出力強度（またはカウント率）を位相 φ の関数として測ると、

$$
I(φ) = I_0\,[1 + V\cos(φ)]
$$

と書け、visibility は

$$
V = (I_{max}-I_{min})/(I_{max}+I_{min})
$$

で定義される。

Kimura et al.（arXiv:quant-ph/0403104v2）は、150 km 伝送後の単一光子干渉で **visibility が 80% を超える**ことを要約に明記している。

### time-tag / 遅延の入り方（実験手続き）

- 位相 φ は（中心周波数 ω と光路差 ΔL による）φ≃2πΔL/λ で与えられる。
- time-tag は「ある時間幅で積算したカウント」として visibility の推定量に間接的に入る（積算時間が長いと位相ゆらぎが平均化され得る）。
- 実験的には、環境ゆらぎ（ΔL ゆらぎ、ωゆらぎ）が visibility を下げる主因となる。

### 最小写像（反証可能な形）

位相ゆらぎがガウス雑音（σφ）で与えられる場合、

$$
V_{obs} = V_0\,\exp(-σ_φ^2/2)
$$

となる。ここで σφ は等価な光路差雑音 σL により σφ=2πσL/λ と書ける。  
本リポジトリでは、V から σL（nm）への写像を固定して可視化する。

---

## 2) Hong–Ou–Mandel（HOM）効果（2光子干渉）

HOM は 2光子がビームスプリッタに同時入射したとき、（理想的には）同時計数が抑制される現象である。  
観測量は「相対遅延 τ を掃引したときの同時計数（coincidence）率 C(τ)」であり、dip の深さ（visibility）を定義して比較する。

arXiv:2106.03871v2 は、量子ドット光子の HOM について、13 ns と 1 μs の分離でも高い visibility を保つことを本文に明記している（QD2）。

### time-tag / 遅延の入り方（実験手続き）

- time-tag は coincidence の構築（Δt のヒストグラム化、窓幅の選別）に直接入る。
- 解析手順として「どの時刻差を coincidence と数えるか（窓幅・オフセット・pairing）」が必要であり、選別は visibility に影響し得る。  
  この観点は Step 7.4（ベル：coincidence 選別）と共通である。

### 公開データ（一次プロダクト）

Zenodo（10.5281/zenodo.6371310）には図ごとの raw が公開されており、例えば Extended Data Fig.3 の低周波ノイズ PSD が含まれる。  
本リポジトリでは、この PSD を「indistinguishability を落とす低周波ノイズが支配的（<10^4 Hz）」という主張の一次入口として固定する。

---

## 3) スクイーズド光（量子雑音の低減）

スクイーズド光の代表的な観測量は、ショット雑音に対する分散比 V（線形）と、その dB 表記

$$
S_{dB} = -10\log_{10}(V)
$$

である。Vahlbruch et al.（arXiv:0706.1431v1）は、10 dB（power）の benchmark を報告している。

### time-structure の位置づけ（最低限）

スクイージング測定は（一般に）coincidence の窓選別ではなく、ホモダイン検波による雑音スペクトル評価として構成される。  
したがって time-tag 選別が主役になる測定ではないが、位相ロックの揺らぎや損失は観測される dB を劣化させる。

損失のみを考えると、観測分散比 V_obs は

$$
V_{obs} = (1-η) + ηV_{in}
$$

（η: 全効率）で下限が付く。特に V_in→0 としても V_obs≥1-η なので、
10 dB（V_obs=0.1）を観測するには η≥0.9 が必要である。

---

## 再現（本リポジトリ）

一次ソース取得：

```powershell
python -B scripts/quantum/fetch_photon_interference_sources.py
```

観測量の固定（図＋メトリクス）：

```powershell
python -B scripts/quantum/photon_quantum_interference.py
```

- 出力：`output/quantum/photon_quantum_interference.png`
- 数値：`output/quantum/photon_quantum_interference_metrics.json`

