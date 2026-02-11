<INSTRUCTIONS>
# waveP 作業ルール（Codex/人間共通）

目的：Pモデル（時間波密度P）を「物質＝波」の前提で体系化し、検証可能・再現可能な形で相対論の置換候補として構築する。

---

## 0) 新規セッション開始（必須）

### 0.1 フォルダ構造（チートシート）
- `doc/`：運用ドキュメント（状態管理の正）
  - 運用の正：`doc/AI_CONTEXT_MIN.json` / `doc/STATUS.md` / `doc/WORK_HISTORY.md` / `doc/WORK_HISTORY_RECENT.md` / `doc/ROADMAP.md` / `doc/PRIMARY_SOURCES.md`
  - 退避：`doc/old/`（現状未使用のMarkdown）
- `scripts/`：再現可能な実行コード（topic別）
  - `scripts/summary/`：レポート/論文の一括生成（`run_all.py`）
  - `scripts/cosmology/`：宇宙論（BAO/一次統計、MAST/JWST 等）
- `data/`：入力データ（外部取得・観測一次データ・再利用）
- `output/`：生成物（図・CSV・HTML・Word 等）
- `gps/` `cassini/` `viking/` `mercury/`：レガシー（更新は `scripts/`/`data/`/`output/` を正）

### 0.2 新規開始時の参照手順（最短ルート）
1. `doc/AI_CONTEXT_MIN.json` を読む（現在地Phase/Step、再現コマンド、主要出力を把握）
2. `doc/STATUS.md` を読む（最新結果＋次にやること）
3. `doc/ROADMAP.md` を読む（公式ロードマップの完了/未完了と、今いるStepを確認）
4. `doc/WORK_HISTORY_RECENT.md` を読む（直近3ログ。必要なら `doc/WORK_HISTORY.md` で詳細）
5. `doc/PRIMARY_SOURCES.md` を読む（一次ソースURLとローカル保存先、取得手順）
6. 作業に着手したら、終了時に必ず以下を更新：
   - `doc/WORK_HISTORY.md`（追記）→ `python -B scripts/summary/update_work_history_recent.py`（recent更新）
   - `doc/STATUS.md`（最小サマリで上書き）
   - `doc/AI_CONTEXT_MIN.json`（最小辞書で上書き）
   - （資料生成はユーザー実行）必要なときにユーザーが `output/summary/build_materials.bat`（例：`C:\develop\waveP\output\summary\build_materials.bat`）を実行して、論文/一般レポート等を再生成する。したがってAIは毎回 `run_all.py` を回す必要はなく、継続して必要な更新作業（上の3点）を優先する。
     - 引数無し（= `full`）は **論文HTML/DOCXを全て再生成**する（`pmodel_paper` / `pmodel_paper_part2_astrophysics` / `pmodel_paper_part3_quantum` / `pmodel_short_note`）。
     - Word が無い環境では DOCX はスキップされ得る（`html_to_docx.py` の戻り値 `RC=3`）。
     - 軽量に回す場合は `quick` / HTMLのみは `quick-nodocx` を使う。

### 0.3 CLIコマンド注意（必須）
- Windows（PowerShell）では `nl` コマンドは禁止（存在しないため混乱/失敗の原因になる）。
- 行番号が必要な場合は PowerShell の `Get-Content` + `ForEach-Object` 等で代替する。
- WSL（bash）上では `nl` の使用を許可する。

## 1) フォルダ規約（最重要）

### 1.0 資料は `doc/`
- Markdown等の説明資料・設計メモ・参考スライド等は **必ず** `doc/` 配下に置く。
- 例：`doc/theory/01_all_matter_is_wave.md`、`doc/P_model_handoff.md`、`doc/slides/時間波密度Pで捉える「重力・時間・光」.pptx`

### 1.1 生成物は `output/`
- 画像・CSV・PPTXなど **生成物は必ず** `output/<topic>/` 配下に出力する。
- 例：`output/gps/summary_batch.csv`、`output/cassini/cassini_shapiro_y_full.csv`
- 例（HTML）：`output/summary/pmodel_public_report.html`、`output/summary/pmodel_paper.html`
- 例（Word）：`output/summary/pmodel_public_report.docx`、`output/summary/pmodel_paper.docx`
  - 共有用DOCXは `scripts/summary/run_all.py` と `scripts/summary/paper_build.py` が自動生成する（WindowsはWordの自動化でHTML→DOCX変換）。
  - 環境にWordが無い場合はスキップされるため、`output/summary/run_all_status.json` とログで確認する。
  - **作業の区切りでの再生成はユーザーが `output/summary/build_materials.bat` で実行する（AI側で毎回必須ではない）。**
    - 一般向け：`output/summary/pmodel_public_report.html` / `output/summary/pmodel_public_report.docx`
    - 論文：`output/summary/pmodel_paper.html` / `output/summary/pmodel_paper.docx`

### 1.2 スクリプトは `scripts/`
- 実行用コードは必ず `scripts/<topic>/` 配下に置く。
- 例：`scripts/gps/compare_clocks.py`、`scripts/cassini/cassini_shapiro_y_full.py`

### 1.3 入力データは `data/`
- 外部取得データ・観測データ・再利用する入力は `data/<topic>/` に置く。
- 例：`data/gps/BRDC00IGS_...rnx`、`data/llr/demo_llr_like.crd`

### 1.4 トピック（topic）区分
- `theory` / `gps` / `cassini` / `viking` / `mercury` / `llr` / `eht` / `pulsar` / `gw` / `bepicolombo`（保留）
- 新規テーマを増やす場合は、同名の `data/<topic>`・`scripts/<topic>`・`output/<topic>` を作る。

### 1.5 既存フォルダの扱い（重要）
- ルート直下の `gps/` `cassini/` `viking/` `mercury/` は **過去資産（レガシー）** として残っている。
- 今後の更新・新規出力は必ず `scripts/` と `output/` と `data/` を正とする。

---

## 2) 「同一機能＝同一ファイル名」ルール（必須）
- 同じ目的のスクリプトを `*_v2.py` 等で増やさない。
- 既存の同等機能がある場合は **必ずそのファイルを編集して拡張**する。
- 迷ったら、まず `rg -n <keyword> scripts/` で既存実装を探してから着手する。

---

## 3) 理論の符号・定義（重力の統一規約）

このリポジトリでは、以下に統一する：
- **質量近傍で P が増える**（`P > P0`）。
- ニュートン形ポテンシャル：
  - `φ ≡ -c^2 ln(P/P0)`（無限遠で0、質量近傍で負）
- 重力加速度（“P勾配へ滑り落ちる”のニュートン形）：
  - `a = -∇φ = c^2 ∇ln(P/P0)`
- **Shapiro遅延などで現れる係数2（GRの(1+γ)=2）は、光伝播側の自由度 `β` が担う**（`n(P)=(P/P0)^(2β)`、PPN対応：`(1+γ)=2β`）。  
  したがって `φ` の定義に `sqrt()`（1/2乗）を混ぜて係数を吸収しない（`φ` は固定、必要なら `β` を微調整する）。
- 文書・コードで `Φ` を使う場合は、必ず符号規約を明記して混在させない。

関連ドキュメント：`doc/P_model_handoff.md`

---

## 3.5 GR/SR との位置づけ（必須の言い回し）

混乱防止のため、GR/SR と P-model（時間波）の関係は次の表現に統一する：
- **GR/SR は観測を説明する優れた幾何学的近似（有効理論）だが、時間の実体を与えない。**
- **P-model は時間を波として仮定し、同じ観測結果を別の機構から再導出する。**
- 一致する部分は「現実が同じだから結果が似る」ためであり、概念は別である。
- 一致は近似として現れるが、強場極限では差が予言として現れる。

---

## 4) 実装規約（再現性）

### 4.1 パスの取り方
- カレントディレクトリに依存しないこと。
- Pythonでは原則として以下の形でリポジトリルートを決める：
  - `ROOT = Path(__file__).resolve().parents[2]`
- 出力先は `OUT_DIR.mkdir(parents=True, exist_ok=True)` で事前作成する。

### 4.2 出力フォーマット
- 時刻は原則 ISO 8601（UTC, timezone付き）で保存。
- 単位は明記（m, s, Hz など）。CSVヘッダに単位が入るのが望ましい。
- 生成物には「入力データ」「パラメータ（β/δ等）」「実行条件」をログに残す（printでも可）。

### 4.3 ネットワーク（HORIZONS等）
- インターネット接続は利用可能。必要な場合はスクリプト側で明示する。
- SSL/Proxy対策は環境変数を優先（例：`HORIZONS_CA_BUNDLE` / `HORIZONS_INSECURE`）。
- 可能なら取得結果を `output/<topic>/` に保存し、オフラインで再利用できる形にする。

### 4.4 Corrfunc / WSL 実行規約（必須）
- Corrfunc を利用する計算（例：BAOの `ξ(s,μ)` / `ξℓ` 再計算）は **必ず WSL（Linux）で実行**すること（Windowsネイティブ実行は禁止）。
- WSL で Corrfunc を実行する場合は、原則として **24スレッド**で実行すること（例：`--threads 24`）。
- WSL で実行する Python 環境は、**本リポジトリ直下の venv（`.venv_wsl`）を必ず使用**する（迷子防止・依存固定）。
  - 例：`wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/develop/waveP && .venv_wsl/bin/python -B scripts/cosmology/cosmology_bao_xi_from_catalogs.py ... --threads 24"`
  - `.venv_wsl` が無い場合は、WSL 側で `cd /mnt/c/develop/waveP && python3 -m venv .venv_wsl && .venv_wsl/bin/python -m pip install -U pip && .venv_wsl/bin/python -m pip install numpy matplotlib Corrfunc` で作成する。
- BOSS DR12v5 LSS の取得元は、高速ミラー `https://dr12.sdss3.org/sas/dr12/boss/lss/` を正とする（`scripts/cosmology/fetch_boss_dr12v5_lss.py` / `data/cosmology/boss_dr12v5_lss/manifest.json`）。

### 4.6 一次ソース参照先（必須）
- 一次データ（観測そのもの）・一次論文（TeX/PDF）の「参照先URL」と「ローカル保存先」は、原則 `doc/PRIMARY_SOURCES.md` にまとめて管理する。
  - 新しい取得元を追加/変更した場合は、`doc/WORK_HISTORY.md` に **必ず追記**し、可能なら `data/<topic>/sources/` に raw をキャッシュする。
- 参考（困った時の一次データ取得）：MAST（NASA Mikulski Archive for Space Telescopes）
  - astroquery ドキュメント：`https://astroquery.readthedocs.io/en/latest/mast/mast.html`
  - JWST のスペクトル等を含む公開観測の一次データ取得に利用できる可能性がある（導入時は `scripts/<topic>/fetch_mast_*.py` を作り、`data/<topic>/` へ保存して offline 再現を確保する）。

### 4.5 ContextWindow 最小化（必須）
- ContextWindow 圧縮後の復帰用に、**作業ごとに**最小の状態辞書を `doc/AI_CONTEXT_MIN.json` に保存する（上書き更新）。
- 圧縮後は、まず `doc/AI_CONTEXT_MIN.json` を読み、この辞書を起点に作業を再開する。
- 形式は **人間可読である必要はない**（minify された1行JSON推奨）。ただし、最低限 `現在地（Phase/Step）`、`次にやること`、`主要参照ファイル`、`主要出力`、`再現コマンド` を含める。

### 4.7 run_all 並列実行（必須）
- `scripts/summary/run_all.py` は、原則として **並列（`--jobs 2`）**で実行する（信頼性優先のため控えめに固定）。
  - 例：`python -B scripts/summary/run_all.py --offline --jobs 2`
  - 例（online）：`python -B scripts/summary/run_all.py --jobs 2`
- 例外：メモリ不足や不安定が疑われる場合のみ `--jobs 1` に戻して切り分ける。

---

## 5) 変更時のドキュメント更新
- 理論の定義・符号・主要式を変えたら `doc/P_model_handoff.md` を更新する。
- 検証結果（数値・図）を更新したら、対応する `output/<topic>/` の成果物名を固定し、差し替える。

---

## 5.5 論文化HTMLの表記（数式の標準ルール）

目的：読者に「\\gamma」「\\delta」等の LaTeX エスケープ文字列が見えて混乱することを防ぐ。

- 本文（説明文）では **LaTeXの `\\` 表記を使わない**（例：`\\gamma` / `\\delta` / `\\phi` / `\\beta` は禁止）。
  - 記号は Unicode（例：γ, δ, φ, β, τ）で書く。
  - 添字などで誤読が起きやすい場合は **数式ブロック（`$$ ... $$`）に逃がす**。
- 数式は `doc/paper/*.md` では **必ず数式ブロック**（`$$ ... $$`）で書き、publish生成時に **数式画像（PNG）**として表示する。
  - 論文HTML（`output/summary/pmodel_paper.html`）の数式画像 `<img>` の `alt` は **汎用表記（`数式`）** とする（LaTeXを入れない）。
- 最終確認（目安）：`rg -n "\\\\" output/summary/pmodel_paper.html` が **no matches** であること。

---

## 5.6 論文フォーム（標準：Part II を正とする）

目的：Part II の体裁を「論文フォームの基準」とし、Part III も同一フォームへ揃える。

- 対象：`doc/paper/11_part2_astrophysics.md` の 4.x/5.x/6.x を標準フォームとする（Part III も同型で書く）。
- フォーム（必要ない項目は省略可・順序固定）：`目的 / 入力 / 処理 / 式 / 指標 / 結果 / 考察 / 注意`
  - 各フォームタイトルは **太字（例：`**目的**：`）**で、箇条書きにはしない（行頭から書く）。
  - 「図」はフォームタイトルにしない（`**図**` を作らない）。図は「結果」の直後に置き、各図に 2〜4行の説明を付ける（説明は箇条書きにしない）。
- 章（`## 4/5/6` など）は、見出し直後に **短い説明文**（何をまとめる章か）を必ず入れる。
- 改行：1行が長くなりすぎないよう適切に改行を入れる（HTML/DOCXでの可読性を優先）。
- 複数結果：複数の結果（複数dataset/複数指標）を並べる場合は、原則として **表形式（Markdown table）**でまとめる。
  - 表の中では `|` 記号は使わない（列崩れの原因になる）。絶対値は `abs(x)` や `∣x∣` を用いる。
- 体裁（HTML/DOCX）：論文の枠線は不要（表/カード/画像の枠は無し）、背景は白に統一する。
- DOCX：章/項/小項（Heading 2/3/4）それぞれの直前に改ページを入れる（読みやすさ優先）。

---

## 6) 作業履歴と進捗（重複防止）

- 重要な変更（モデル式/一次ソース/レポート構成/バッチ結果の更新など）を行ったら `doc/WORK_HISTORY.md` に**追記**する（既存内容の削除はしない）。
- `doc/WORK_HISTORY.md` は **UTC時系列（古→新）**で管理する（追記は末尾へ）。順序が崩れた場合は、時系列に整頓してから運用を継続する。
- AI が作業開始時に参照する **小さな履歴**として、`doc/WORK_HISTORY_RECENT.md`（直近3ログ）を併用する。
  - `doc/WORK_HISTORY.md` を更新したら、`doc/WORK_HISTORY_RECENT.md` も必ず更新する（上書き）。
  - 更新は `python -B scripts/summary/update_work_history_recent.py` を正とする（3ログ固定）。
- 現在地（ロードマップのチェック）は `doc/STATUS.md` を更新する（最終更新UTCも更新）。
  - `doc/STATUS.md` は「最新の作業結果」と「次に行うべき作業」だけの **最小サマリ**として扱い、毎回上書き更新する（履歴の蓄積は `doc/WORK_HISTORY.md` を正とする）。
- 日々の運用（更新対象）は原則として以下のファイルに集約する：
  - `doc/AI_CONTEXT_MIN.json` / `doc/STATUS.md` / `doc/WORK_HISTORY.md` / `doc/WORK_HISTORY_RECENT.md` / `doc/ROADMAP.md` / `doc/PRIMARY_SOURCES.md`
  - それ以外の doc 直下 Markdown は、現状未使用なら `doc/old/` に移動して保管する（参照のみ）。
- ロードマップ（完了/未完了の管理）は `doc/ROADMAP.md` を正とする（Phase 1-8 の大枠固定／追加作業は既存Phase配下にStepとして追記／新規Phase追加は禁止）。
- 自動ログ（機械可読の実行履歴）は `output/summary/work_history.jsonl` に追記する（**手動編集しない**）。
  - 主要スクリプトは `scripts/summary/worklog.py` の `append_event()` を使って記録する。
- このチャットでの「作業完了報告」には、**必ず**以下を添える（1〜3行でよい）：
  - ロードマップ上の現在地（Phase/項目番号）
  - その作業で進捗がどう動いたか（完了/進行中/次にやること）
  - 参照先（`doc/STATUS.md` の該当章、必要なら `doc/WORK_HISTORY.md`）

</INSTRUCTIONS>
