# 公開・再現手順（Phase 8 / Step 8.3）

目的：第三者が **同じ成果物（HTML）を再生成できる**こと、および「どこまでが一次ソース/入力で、どこからが生成物か」を明確にする。

---

## 1) 生成物（公開の入口）

- 論文HTML（publish）：`output/summary/pmodel_paper.html`
- 短報HTML（publish）：`output/summary/pmodel_short_note.html`
- 一般向けレポート（HTML）：`output/summary/pmodel_public_report.html`

注意：論文HTMLは図を `data:image/png;base64,...` で埋め込むため、PNGだけ差し替えてもHTMLは更新されない。図の差し替え後は必ず `build_materials.bat`（quick）で再生成する。

---

## 2) 最小の再生成（普段の作業用）

### 2.1 HTMLのみ更新（軽量）

以下で **論文HTML + 短報HTML + 一般向けレポートHTML** を更新する（DOCXは生成しない）。

- `cmd /c output\summary\build_materials.bat quick-nodocx`

目安（この環境）：約3秒（HTMLのみ）。

### 2.2 論文publishの軽量QC（推奨）

publish成果物に対して、機械的に検出できる崩れ/混入をチェックする。

- `python -B scripts/summary/paper_qc.py`
  - 出力：`output/summary/paper_qc.json`

チェック内容（概要）：
- 引用キー/図表インデックス整合（paper_lint strict；paper + short note）
- 図番号（図1..）と `fig-001..` の連番・単調増加
- 数式画像の `alt="数式"`
- HTMLの `\\` 混入（LaTeX文字列が紙面に露出しないこと）
- Markdown本文（`$$...$$` 外）に LaTeX エスケープ（例：`\gamma`）が混入していないこと

### 2.3 公開マニフェスト（推奨）

「公開の入口（必須ファイル）」「再現コマンド」「ファイルの存在/サイズ（任意でsha256）」を機械可読で固定する。

- `python -B scripts/summary/release_manifest.py`
  - 出力：`output/summary/release_manifest.json`
  - 高速化：`python -B scripts/summary/release_manifest.py --no-hash`（sha256省略）

備考：
- 実行環境の補助情報として `output/summary/env_fingerprint.json` も自動生成する（Python/OS/pip freeze）。

### 2.4 配布バンドル（zip; 推奨）

第三者へ渡すための zip を生成する（rawデータは同梱しない）。

- paper（読み物＋最小ドキュメント）：`python -B scripts/summary/release_bundle.py --mode paper --no-hash`
  - 出力：`output/summary/pmodel_release_bundle_paper.zip`
- repro（コード＋docs＋入口＋summary成果物）：`python -B scripts/summary/release_bundle.py --mode repro --no-hash`
  - 出力：`output/summary/pmodel_release_bundle_repro.zip`

備考：
- `output/summary/pmodel_paper.html` は図を base64 埋め込みするため、サイズが大きくなり得る（数十MB）。
- バンドルは `data/` を含めない（一次ソースURLと取得スクリプトで再現する）。

---

## 3) フル再生成（重い：作業区切り/公開前）

全検証をオフラインで再実行し、出力を更新する（並列は原則 `--jobs 2`）。

- `python -B scripts/summary/run_all.py --offline --jobs 2`

目安（この環境）：約9分（DOCX生成を含む。Wordなしなら短縮）。

環境に Microsoft Word がある場合は、DOCX（`output/summary/*.docx`）も生成される。

---

## 4) 公開の前提（一次ソース）

一次ソース（URL/参照日/ローカル保存先）は `doc/PRIMARY_SOURCES.md` を正とする。  
新しい取得元を追加/変更した場合は、`doc/WORK_HISTORY.md` に追記し、可能なら `data/<topic>/sources/` に raw をキャッシュする（offline 再現を確保する）。

---

## 5) 典型フロー（推奨）

1. 作業を進める（スクリプト更新/出力更新）
2. `cmd /c output\summary\build_materials.bat quick-nodocx`
3. `python -B scripts/summary/paper_qc.py`（ok=true を確認）
4. `python -B scripts/summary/release_manifest.py`（ok=true を確認）
5. 公開前に `python -B scripts/summary/run_all.py --offline --jobs 2` を実行して出力を固定

---

## 6) 公開範囲（含める/含めない）

原則：
- 読者が読むための成果物（HTML）と、第三者が再現するための入口（scripts/doc）を分ける。
- 巨大な raw データは同梱せず、一次ソースURL（`doc/PRIMARY_SOURCES.md`）と取得/キャッシュ方法を明示する。

最低限（読み物としての公開）：
- `output/summary/pmodel_paper.html`
- `output/summary/pmodel_short_note.html`
- `output/summary/pmodel_public_report.html`
- `doc/ROADMAP.md` / `doc/STATUS.md` / `doc/PRIMARY_SOURCES.md`

再現性の入口（コード公開の最小セット）：
- `scripts/`（実行コード）
- `doc/paper/`（本文・参考文献・図表インデックス）
- `output/summary/build_materials.bat`（資料生成の入口）
- `output/summary/release_manifest.json`（公開対象の機械可読一覧）
- `output/summary/env_fingerprint.json`（実行環境の補助情報）

巨大データの扱い（同梱しない例）：
- `data/cosmology/desi_dr1_lss/raw/`（DESI LSS raw）
- `data/llr/edc/`（LLR EDC batch）
  - 代替：参照URLと取得手順を一次ソースとして固定し、必要なら生成物（`output/<topic>/` の固定図/CSV/JSON）を同梱する。
