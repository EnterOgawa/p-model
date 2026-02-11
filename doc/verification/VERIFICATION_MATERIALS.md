# 検証用資料（Verification Materials）

目的：Part I–III の論文本文（publish）では、読者の理解を妨げるため **具体的なファイル名・フォルダ名・スクリプト名を記載しない**。  
その代わり、本資料に「再現コマンド」「固定アーティファクト一覧」「監査ゲート」「出力の固定名」「ハッシュ（必要なら）」を集約し、第三者が同じ成果物を再生成できるようにする。

本資料は論文から参照される **検証の正** とし、論文本文は「結果・凍結・棄却条件・差分予測」のみに集中させる。

---

## 0. 公開・参照（論文からの参照先）

論文側では、以下の参照先（URL/DOI）を 1 箇所に固定して引用する。

- 参照先（推奨）：GitHub Releases または Zenodo DOI（リリースごとに固定）
- 参照先（代替）：GitHub リポジトリ（tag を固定して参照）

（ここは公開時に埋める）
- GitHub：`https://github.com/EnterOgawa/p-model`
- Releases：`https://github.com/EnterOgawa/p-model/releases`
- DOI（任意）：未割当（必要なら Zenodo 連携で release ごとに DOI を発行）

---

## 1. リポジトリ規約（再現性）

入力/コード/生成物の規約：
- 入力データ：`data/<topic>/`
- 実行コード：`scripts/<topic>/`
- 生成物（固定名）：`output/<topic>/`
- 運用ドキュメント：`doc/`

論文生成（HTML/DOCX）とQC：
- 論文HTML生成：`scripts/summary/paper_build.py`
- 論文QC：`scripts/summary/paper_qc.py`
- まとめ実行：`scripts/summary/run_all.py`（並列は原則 `--jobs 2`）

---

## 2. 最小の再現（推奨フロー）

最小フロー（publish HTML + QC）：
- 論文生成（Part I）：`python -B scripts/summary/paper_build.py --profile paper --mode publish --outdir output/summary --skip-docx`
- 論文生成（Part II）：`python -B scripts/summary/paper_build.py --profile part2_astrophysics --mode publish --outdir output/summary --skip-docx`
- 論文生成（Part III）：`python -B scripts/summary/paper_build.py --profile part3_quantum --mode publish --outdir output/summary --skip-docx`
- QC：`python -B scripts/summary/paper_qc.py`

監査（Part III：pack再生成→ゲート判定→要約）：
- `python -B scripts/summary/part3_audit.py`

フル再現（重い）：
- `python -B scripts/summary/run_all.py --offline --jobs 2`

---

## 3. 固定アーティファクト（Part I）

凍結パラメータ集・反証条件・公開マニフェスト：
- `output/theory/frozen_parameters.json`
- `output/summary/decisive_falsification.json`
- `output/summary/weak_field_falsification.json`
- `output/summary/release_manifest.json`
- `output/summary/env_fingerprint.json`

一次ソース（URL/参照日/ローカル保存先）：
- `doc/PRIMARY_SOURCES.md`

---

## 4. 固定アーティファクト（Part II：宇宙物理）

例（DDR の反証条件パック）：
- `output/cosmology/cosmology_ddr_pmodel_falsification_pack.json`

LLR（全グラフ生成・固定バッチ）：
- `python -B scripts/llr/llr_batch_eval.py --offline`

（他の検証テーマの入口・固定出力は、`doc/PUBLISHING.md` と `output/summary/release_manifest.json` を正として更新する）

---

## 5. 固定アーティファクト（Part III：量子物理）

凍結パラメータ一覧（量子）：
- `output/quantum/frozen_parameters_quantum.json`

反証条件パック：
- Bell：`output/quantum/bell/falsification_pack.json`
- 核（Δω→B.E.）：`output/quantum/nuclear_binding_energy_frequency_mapping_falsification_pack.json`
- 物性/熱：`output/quantum/condensed_falsification_pack.json`

監査サマリ：
- 物性/熱 holdout：`output/quantum/condensed_holdout_audit_summary.json`
- Part III 監査：`output/summary/part3_audit_summary.json`

---

## 6. GitHub へ置けるか？（結論：可能）

可能。ただし、以下のどれを GitHub の「通常コミット」に含めるかは運用判断になる：

- 含めやすい：`scripts/`、`doc/`、小さめの `output/summary/*.json`、`output/summary/*.html`
- 注意が必要：大量/巨大な `output/`（図やHTMLの base64 埋め込みで数十MB化し得る）、バイナリ、巨大CSV
- 原則同梱しない：巨大な raw 入力（一次ソースURL＋取得スクリプトで再現する方針）

推奨案（公開の安定性）：
1. GitHub 本体：コード（`scripts/`）＋論文草稿（`doc/paper/`）＋運用doc（`doc/`）
2. 成果物：GitHub Releases（zip）に `output/summary/` の publish HTML と manifest を同梱
3. DOI が必要なら：Releases を Zenodo 連携して DOI 発行（引用を固定しやすい）
