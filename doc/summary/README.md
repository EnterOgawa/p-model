# SUMMARY REPORT
ASCII preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

# 比較レポート（一覧表示）：再現手順

目的：各検証（Cassini/Viking/LLR/GPS/Mercury）の比較図を、**pmodel_public_report.html の形式（解説付きカードUI）**で 1つのHTMLにまとめて閲覧する（掲載対象は手動で選別）。

---

## 実行（PowerShell）

まず全検証を実行して `output/<topic>/` を更新する（初回はネットワークが必要な項目あり）：

`python -B scripts/summary/run_all.py --online --open`

Horizons の取得が SSL で失敗する場合（社内Proxy等）は、CAを指定する：

`python -B scripts/summary/run_all.py --online --horizons-ca-bundle C:\path\corp_root.pem --open`

最終手段（非推奨。証明書検証を無効化）：

`python -B scripts/summary/run_all.py --online --horizons-insecure --open`

2回目以降（キャッシュがある場合）はオフライン再現：

`python -B scripts/summary/run_all.py --offline --open`

ブロック中/保留テーマ（`doc/BLOCKED.md`）も併せて実行したい場合：

`python -B scripts/summary/run_all.py --online --include-blocked --open`

一般向けレポート生成だけ行う場合（推奨）：

`python -B scripts/summary/public_dashboard.py`

（補足）旧レポート（技術向け）生成だけ行う場合：

`python -B scripts/summary/pmodel_report.py --open`

---

## 生成物（output/summary）

- `output/summary/pmodel_public_report.html`（一般向け・統一形式の一覧レポート）
- `output/summary/pmodel_public_dashboard.png`（代表図のダッシュボード）
- `output/summary/run_all_status.json`（run_all の実行結果/ログパス）
- `output/summary/work_history.jsonl`（自動ログ：手動編集しない）
- `output/summary/paper_table1_results.md`（論文用 Table 1：検証サマリ）
- `output/summary/pmodel_report.html`（旧：技術向け一覧レポート）
- `output/summary/pmodel_report_summary.json`（指標/参照パスのサマリ）

---

## 注意

- `scripts/summary/run_all.py` は各検証スクリプトを実行して `output/<topic>/` を更新し、最後に `pmodel_public_report.html` を生成する（論文用 Table 1 も更新する）。
- `scripts/summary/public_dashboard.py` は既存の `output/<topic>/` から **選別した** 画像を使って、解説付きカードUI（統一形式）のHTMLを生成する（再計算はしない）。
- 個別の検証を更新したら、対応するRunbookに従って `output/<topic>/` を作り直してからレポートを更新する。
