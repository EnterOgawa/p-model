# P-model: A Unified Theory of Time-Wave Dynamics
**時間波ダイナミクスに基づく統一理論 (The P-model)**

[![Release](https://img.shields.io/github/v/release/EnterOgawa/p-model?label=Latest%20Release)](https://github.com/EnterOgawa/p-model/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18981366.svg)](https://doi.org/10.5281/zenodo.18981366)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY-ND 4.0](https://img.shields.io/badge/License-CC_BY--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nd/4.0/)

**Document v2.0 (Bilingual Release)  2026-04-23 UTC**

## 👤 Author
- **Shunji Ogawa**
- **ENTERSYSTEM Co., Ltd.**
- **Email:** `s_ogawa@entersystem.co.jp`

## 🌐 For English Readers
**P-model** proposes that "all matter is waves, and the medium is time itself." From a single scalar field `P(x)`, it derives gravity, clocks, light propagation, rotation, and quantum phenomena, all through one unified mapping.

Key results: (1) `β=1` predicted from the null geodesic of the effective metric and confirmed by LLR and MESSENGER; (2) the fine-structure constant `α` derived, not fitted, from a single frozen action with no free parameters, matching the CODATA measured value to 0.02%. 48 verification items, 5 falsifiable predictions for 2027-2035+.

**Full English translation (Parts I–V) is now available on Zenodo as part of the v2.0 Bilingual Release.** Scientific content is unchanged from the Japanese v2.0; no theoretical results have been modified. Please **Star** this repo to stay updated on future predictions and verification results.

📄 **[Download English PDFs from Zenodo →](https://doi.org/10.5281/zenodo.18981366)**

---

## 🕊 Foreword: 理念と先人たちへの敬意
P-modelは、既存の物理学を否定するためのものではありません。

アインシュタインをはじめ、物理学という巨大な山に挑み続けてきたすべての先人たちに、心からの感謝と敬意を。彼らの観測データと理論の蓄積がなければ、このモデルは生まれませんでした。

この理論の出発点はとてもシンプルです。「宇宙は、見かけほど複雑ではないはずだ」という直感。マクロな星の運動もミクロな粒子のふるまいも、根っこでは同じ仕組みで動いている。それを数式とデータで示したい。その一心で書き上げました。

とりわけ、微細構造定数 α ≈ 1/137 という数字には、物理学の巨人たちが繰り返し挑んできた歴史があります。ゾンマーフェルトが 1916 年にこの定数を導入し、ディラックはその値の意味を生涯問い続けました。ファインマンは α を「物理学の最大の謎」と呼び、「すべての優れた理論物理学者はこの数字を壁に貼って悩む」と書きました。朝永振一郎、シュウィンガー、ファインマンの三人は量子電磁力学の繰り込み理論で α を使った精密計算を可能にしましたが、α がなぜその値なのかという問いには誰も答えられませんでした。P-model の v2.0 で、この問いに対するひとつの回答を提示します。それが正しいかどうかは、これからの検証が決めます。

本稿でいう「閉じた」とは、現代物理の主要領域に対して
P-model 側の数式辞書と観測写像を与え終えたという意味であり、
理論の最終的な正否は将来の観測による経験的採否に委ねます。

---

## 🌊 時間波ダイナミクスとは何か

P-modelの核心は、たったひとつの視点の転換です。

> **時間は、ただ流れるだけの「背景」ではない。宇宙のあらゆるスケールを貫く「波」そのものだ。**

この発想には、4つの意図があります。

**1. ミクロとマクロを分けない**<br>
量子力学と一般相対性理論。現代物理学はこの2つを別々の理論で記述しています。P-modelは「そもそも分ける必要がない」と考えます。どちらも、時間波のふるまいの異なる側面にすぎません。

**2. 宇宙を測る「物差し」自体が揺れている**<br>
宇宙は膨張している、とされています。でも、もし物差しの方が変わっていたら。P-modelは、ダークエネルギーのようなパラメータを一切使わずに、観測データに自然な説明を与えます。

**3. 量子コンピュータの動作原理がヒントになった**<br>
量子の「不気味な遠隔作用」に頼るのではなく、波の共鳴や干渉という局所的な仕組みで説明する。量子コンピュータが実際にどう動いているかを考えるうちに、この理論は形になりました。

**4. 4次元のまま、特異点を消す**<br>
高次元を仮定する必要はありません。時間が波として振る舞うだけで、ブラックホールの特異点のような「物理法則の破綻」を自然に回避できます。

---

## 📊 検証スコアボード(現在地)

全48件の検証結果です。通ったもの、保留のもの、棄却されたものすべて隠さず公開しています。

| 判定 | 件数 | どんな項目か |
|---|---:|---|
| **Pass** | 20 | 弱場テスト(LLR, Cassini, Viking 等)、β終端、XRISM、SPARC、量子系 |
| **Watch** | 17 | EHT(精度待ち)、GW偏光(検出器増待ち)、宇宙論(高z待ち)、Bell selection 等 |
| **Reject** | 6 | caseA計量、純スカラー極限、Pantheon+直接fit、量子サブセット3件 |
| **Reference** | 5 | 光偏向γ*、赤方偏移ε*、フレームドラッグ、GPS、速度飽和δ |

β(光の応答指数)は理論から `β=1` と予言し、LLR と MESSENGER の2つの独立チャネルで確認。**β 終端監査: Pass。**

α(微細構造定数)は、標準模型では 20 個以上ある自由パラメータのひとつとして実験値を手で入れます。P-model では、凍結作用から自由パラメータなしに導出される理論の出力値であり、CODATA の測定値との差は 0.02%。**v2.0 の理論側の主結果のひとつです。**

宇宙論は率直に言って弱いです。Pantheon+ では ΛCDM に AIC で負けており、DDR には深刻な張力があります。ここは、より遠い宇宙のデータが揃うまで決着しません。

---

## 📚 論文五部作

| Part | タイトル | ひとことで言うと |
|---|---|---|
| **I** | コア理論: 理論的基礎と写像原理 | `P` と `φ` の定義、`β=1` の導出、凍結パラメータの一元管理 |
| **II** | 宇宙物理編: 宇宙物理学および宇宙論的検証 | 太陽系からブラックホール、重力波、宇宙論まで27件の検証 |
| **III** | 量子物理編: 微視的および量子的現象の再評価 | Bell テスト再解析、核物理、BBN、V-A 構造、21件の検証 |
| **IV** | 再現性監査・公開成果物レジストリ・更新運用 | 検証結果の登録、再実行条件の記録、公開成果物のレジストリ |
| **V** | 未来への予測 | 「いつ何を観測すれば決着するか」を8項目で固定 |

> *論文本文は Zenodo で公開中です。v2.0 Bilingual Release(2026-04-23)により、日本語版と英語版の両方がダウンロードできます。*
>
> *The papers are available on Zenodo. The v2.0 Bilingual Release (2026-04-23) provides both Japanese and English versions.*

📄 **[論文 PDF / Papers (Part I–V, JP+EN) on Zenodo →](https://doi.org/10.5281/zenodo.18981366)**

---

## 🔮 未来への予測決着は、これから

この理論が正しいかどうかは、まだ分かりません。分かるのはこれからです。P-modelは具体的な数字を先に置いて、未来の観測を待ちます。

| 予測 | 何が起きるはずか | いつ頃 | 種別 |
|---|---|---|---|
| **ブラックホールの影** | GR より約 **4.8% 大きい** | ngEHT ~2030 / BHEX ~2031 | コア |
| **重力波の偏光** | `P_μ` ベクトル波の横波成分 | LIGO O5 ~2027-2028 | コア |
| **宇宙論の独自項** | 高 z で `ln(1+z)` に由来するずれ | DESI / Euclid / Roman ~2027-2035 | コア |
| **独立銀河回転曲線** | 別の銀河群でも `κ_a=1/(2π)` が成立 | 独立 sample ~2026-2032 | コア |
| **マクロ量子干渉** | 巨大分子の干渉パターンが壊れる | MAQRO型 ~2035+ | 拡張仮説 |
| **量子計算実験** | デコヒーレンスに理論的な下限ライン | platform benchmark ~2026-2032 | 拡張仮説 |
| **弱場重力量子** | P-model 固有の観測配置でズレ検出 | 原子干渉計 / 光格子時計 ~2026-2035+ | コア |
| **距離測定の前提検証** | 食い違いは測定前提の違いに帰着 | 標準サイレン / JWST ~2027-2030 | コア |

最もクリティカルなのは、**ブラックホールの影の 4.8% 差**。2030年前後に ngEHT が測定精度1%に達すれば、この理論の採否が決まります。

---

## この理論の特徴

**β=1 はフィットではなく予言**<br>
有効計量の null 測地線から導出された理論値です。3つの独立チャネル(Cassini, LLR, MESSENGER)で確認済み。

**α  1/137 を「測る」のではなく「導く」**<br>
標準模型では、微細構造定数 α は 20 個以上ある自由パラメータのひとつであり、実験で測って手で入れる数字です。P-model では、α は単一の凍結作用から 2 つの独立読み出しの一致条件で導出される結果です。測定値ではなく、理論が出力した数字。その導出値と CODATA の測定値との差は 0.02%。

**棄却された箇所も全部公開**<br>
理論が合わなかった場所(caseA 棄却、CMB の形状改善ゼロ、DDR の深刻課題)を隠しません。反証条件は検証の前に凍結し、結果がどうであれ報告します。

**参照枠に依存しない**<br>
本文は P-model の定義と写像だけで閉じています。GR や量子力学との比較は Reference note として隔離し、P-model の定義を既存理論で置き換えることはしません。

**W/Z ボソン質量を理論内部から導出**<br>
2成分結合 Q-ball により W および Z ボソンの質量を導出(family (17,1,1)、精度 0.002%)。標準模型ではこれらは独立パラメータですが、P-model では凍結作用から得られる結合局在条件の帰結です。

---

## 🔬 再現性と反証可能性

すべての検証に「棄却手順(Rejection Protocol)」を適用しています。

1. **入力**: NASA, ESA, LIGO, NIST, EHT, AME2020 等の公開一次データ
2. **凍結**: `β=1` 等のパラメータは Part I で固定。変更するなら Part I を改訂して全パートを再生成
3. **棄却閾値**: 反証条件パック(JSON)として機械可読で事前固定

---

## 自分の手で検証する

**全パート一括生成**
```bash
python -B scripts/summary/run_all.py --offline --jobs 2
```

補足: キャッシュ未整備で一次データ取得が必要な場合は `--offline` を `--online` に置き換えてください。

**パートごとの個別生成**
```bash
# Part I(コア理論)
python -B scripts/summary/paper_build.py --profile paper --mode publish --outdir output/private/summary --skip-docx

# Part II(宇宙物理編)
python -B scripts/summary/paper_build.py --profile part2_astrophysics --mode publish --outdir output/private/summary --skip-docx

# Part III-A(量子基盤理論)
python -B scripts/summary/paper_build.py --profile part3a_quantum_foundations --figure-lang ja --mode publish --outdir output/private/summary --skip-docx

# Part III-B(量子検証応用)
python -B scripts/summary/paper_build.py --profile part3b_quantum_verification --figure-lang ja --mode publish --outdir output/private/summary --skip-docx

# Part IV(検証資料)
python -B scripts/summary/paper_build.py --profile part4_verification --figure-lang ja --mode publish --outdir output/private/summary --skip-docx

# Part V(未来への予測)
python -B scripts/summary/paper_build.py --profile part5_future_predictions --figure-lang ja --mode publish --outdir output/private/summary --skip-docx
```

Ver.1.1 系では Part III を `III-A(量子基盤理論)` と
`III-B(量子検証応用)` に分割して運用します。

生成された公開データは `output/public/` に、HTML / `.tex` / PDF は `output/private/summary/` に、最終論文 PDF のみは `papers/` に配置されます。
`paper_build.py` は既定で DOCX を生成しないため、`--skip-docx` は現行運用を明示するための互換オプションです。
Bell 再解析で使う `output/public/quantum/bell/*/normalized_events.npz` は Git 本体ではなく Release assets から取得してください。

---

## 🚀 Getting Started

1. **論文を読む**: Part I-V(Zenodo 公開中)
2. **コードを動かす**: `scripts/` と `output/` で自分の環境で検証
3. **議論する**: 反証・バグ報告は GitHub Issues へ

---

### 📄 ビルド要件
TeX ソースは **LuaLaTeX** で組版しています。TeX Live 等のフルインストール環境が必要です。バージョン管理は `pmodel_version.sty` で一元化しています。

> **For English readers:** The `.tex` sources require **LuaLaTeX** with `luatexja` for Japanese typesetting. Ensure your TeX distribution includes `lualatex`.

---

## ライセンス

**論文テキスト・図表:** [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)  
クレジット付きの共有は歓迎、改変配布は禁止。

**検証コード・データ:** [MIT License](https://opensource.org/licenses/MIT)  
自由に実行・利用・改変可能。

---

P-model v2.0 は、この視点に基づく数式閉包を固定した版です。現代物理の主要領域(重力・光伝播・回転・電磁気・弱相互作用・量子測定・核・物性・熱)に対する導出と観測写像は一通り完了しました。以後に残る課題は新しい物理法則の欠落ではなく、観測監査・数値拘束・橋渡し量の固定です。正否は、これからの観測が決めます。

> *"全ての事象はシンプルである。時間は幾何学の次元ではなく、波であり、マクロな星の自転も、ミクロな素粒子のスピンも、すべてはその波の局所的な渦に過ぎない。"*
