# WORK HISTORY RECENT
This file starts with an ASCII-only preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

# 作業履歴（直近3ログ）

このファイルは AI が作業開始時に参照する **直近3ログ**（`doc/WORK_HISTORY.md` から自動抽出）である。
完全な履歴は `doc/WORK_HISTORY.md` を参照する。

---

## 2026-02-11（UTC） 設計方針変更：論文本文から file/folder/script を排除し、検証用資料へ集約

目的：
- Part I–III の論文（publish）では、具体的なファイル名・フォルダ名・スクリプト名を本文に書かない方針へ変更する。
- その代わり、再現コマンド／固定アーティファクト一覧／監査ゲート等は「検証用資料（Verification Materials）」へ集約し、論文は参照のみを載せる。
- publish HTML に repo-path が混入した場合に機械検出できるよう、QCを強化する。

作業：
- 検証用資料を追加：
  - `doc/verification/VERIFICATION_MATERIALS.md`（再現コマンド・固定アーティファクト一覧・GitHub公開案の集約）
- 論文（Part I–III）本文から file/folder/script 表記を除去し、検証用資料への参照へ置換：
  - `doc/paper/10_part1_core_theory.md`
  - `doc/paper/11_part2_astrophysics.md`
  - `doc/paper/12_part3_quantum.md`
  - `doc/paper/07_llr_appendix.md`（LLR付録の再現コマンドを置換）
- 一次ソース一覧（付録）からローカル固定入力パスを除去（量子核のキャッシュ表記）：
  - `doc/paper/20_data_sources.md`
- 参考文献に Verification Materials の参照キーを追加：
  - `doc/paper/30_references.md`（`[PModelVerificationMaterials]`）
- publish HTML の repo-path 混入を paper_qc で検出：
  - `scripts/summary/paper_qc.py`（`paper_html_no_repo_paths` / `part2_html_no_repo_paths` / `part3_html_no_repo_paths`）
- 再生成・検証：
  - `python -B scripts/summary/paper_build.py --profile paper --mode publish --outdir output/summary --skip-docx`
  - `python -B scripts/summary/paper_build.py --profile part2_astrophysics --mode publish --outdir output/summary --skip-docx`
  - `python -B scripts/summary/paper_build.py --profile part3_quantum --mode publish --outdir output/summary --skip-docx`
  - `python -B scripts/summary/paper_qc.py`（ok=True）
  - `python -B scripts/summary/part3_audit.py`（ok=True）

結果：
- Part I–III の publish HTML から、具体的な repo 内パスや実行コマンドの露出を排除し、検証用資料への参照に統一できた。
- publish HTML に `output/`/`data/`/`scripts/`/`doc/` が混入した場合に `paper_qc` が失敗するため、以後の再発を自動検出できる。

[work] 2026-02-11T02:47:38Z Phase 8 / Step 8.2.13：設計方針変更として、論文本文（publish）から file/folder/script 表記を排除し、`doc/verification/VERIFICATION_MATERIALS.md` へ集約。paper_qc に repo-path 混入検出を追加し、paper_build/paper_qc/part3_audit ok=True を確認。


---

## 2026-02-11（UTC） ロードマップ追記：Verification Materials の公開参照（URL/DOI）固定

目的：
- 論文（Part I–III）から参照する検証用資料（Verification Materials）の参照先を、URL/DOIで固定できる形にする。
- 参考文献キー `[PModelVerificationMaterials]` の Web 欄が `TBD` のまま残ることを防ぎ、公開時の最終手順をロードマップで明示する。

作業：
- `doc/ROADMAP.md`：
  - 8.2.14（公開参照の固定）を追加（準備中）。
- `doc/STATUS.md` / `doc/AI_CONTEXT_MIN.json`：
  - 次タスクとして 8.2.14 を明示し、状態辞書を同期。

結果：
- 公開先（GitHub Releases / DOI）を決めた段階で、論文側の参照キーと検証用資料の参照先を一括で確定できる状態になった。

[work] 2026-02-11T02:56:28Z Phase 8 / Step 8.2.14（準備中）：Verification Materials の公開先（URL/DOI）固定タスクをロードマップへ追加し、STATUS/AI_CONTEXT を同期。


---

## 2026-02-11（UTC） GitHub 公開準備：README（英語）追加＋`.gitignore` 整備＋ローカル git 初期化

目的：
- GitHub 上に P-model プロジェクトを作成する前段として、リポジトリの入口（README）を用意し、巨大生成物/データを誤ってコミットしない最小ガード（.gitignore）を入れる。
- 次タスク（Step 8.2.14）で「Verification Materials の参照先（URL/DOI）を固定」できる状態にする。

作業：
- 追加：`README.md`（英語でプロジェクトの意味・位置づけ・再現資料の方針を記述）
- 追加：`.gitignore`（`output/` を原則無視、`output/summary/build_materials.bat` は例外で追跡。`data/` は manifest JSON のみ追跡）
- `git init -b main` を実行し、ローカル git リポジトリを初期化

結果：
- GitHub でリポジトリを作成して remote を設定すれば、そのまま初回 push ができる入口が整った。
- 次は GitHub 側で repo を作成し、`doc/verification/VERIFICATION_MATERIALS.md` と参考文献キー `[PModelVerificationMaterials]` の `TBD` を固定URL/DOIに置換する（Step 8.2.14）。

[work] 2026-02-11T03:11:03Z Phase 8 / Step 8.2.14（進行中）：README（英語）追加＋`.gitignore` 整備＋ローカル git 初期化（main）。次：GitHub repo作成→URL/DOI確定→paper参照キー更新。
