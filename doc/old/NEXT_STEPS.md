# NEXT STEPS (Phase 10: Multi-event strengthening)
This file starts with an ASCII-only preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

Update: 2026-01-09 (UTC). Phase 15（Step 15.2: SNR/FARメタ情報の追加と要約図への併記）まで完了。

# 次にやること（Phase 10：多事例の追加強化）

目的：
- “1例で合う”ではなく、強場（GW/連星）で **多事例でも同じ手順で比較できる**状態にする。
- 一次ソース＋固定出力＋棄却条件の枠組みを崩さず、適用範囲と限界を明確化する。

## Phase 10：進捗チェック（ここを更新する）

- [x] Step 10.1：重力波（GW）をさらに1イベント追加（GW170104）し、同一手順で再現可能にする
- [x] Step 10.2：強重力の放射制約を拡張（NS-WD連星などの一次ソース制約を追加し、双極放射の棄却を強化）
- [x] Step 10.3：重力波（BNS: GW170817）も追加して、窓/帯域/指標のロバスト性を確認

再現（必須）：
- `python -B scripts/summary/run_all.py --offline`（一般向けレポート＋DOCX）
- `python -B scripts/summary/paper_build.py --mode publish`（論文HTML＋DOCX）

---

# 次のロードマップ（Phase 11：強重力の品質改善）

## Phase 11：進捗チェック（ここを更新する）

- [x] Step 11.1：GWの多イベント結果を1枚に要約（R^2/match）し、Table 1 と整合させる
- [x] Step 11.2：BNSをもう1イベント追加（例：GW190425）し、同一手順の適用限界を確認
- [x] Step 11.3：波形比較の指標と前処理（whiten/bandpass/窓）を固定し、再現性をさらに上げる（失敗は理由付きで出力）

# 次のロードマップ（Phase 12：GW比較の標準化と拡張）

## Phase 12：進捗チェック（ここを更新する）

- [x] Step 12.1：波形比較の窓を「時間」ではなく「周波数帯（例：70-300 Hz）」で固定し、イベント間で match をより比較可能にする
- [x] Step 12.2：イベント一覧をデータ駆動（`data/gw/event_list.json`）にし、追加/更新をコード改変なしでできるようにする
- [x] Step 12.3：GWTC-3（GWTC-3-confident）から BBH/NSBH を各1件追加し、同一手順での再現（online→offline）を確認する（GWTC-3 には BNS が無いため NSBH を使用）

# 次のロードマップ（Phase 13：GWイベント管理の整備）

- [x] Step 13.1：`data/gw/event_list.json` に「イベント種別（BBH/BNS/NSBH）」と推奨パラメータ（窓/帯域）を持たせ、`argv` 手書きを減らす（profiles方式）
- [x] Step 13.2：失敗理由（no valid track 等）を集計し、イベント種別ごとに既定値を調整できる入口を作る（`output/gw/gw_event_list_diagnostics.json`）
- [x] Step 13.3：追加イベントを1件だけ `event_list.json` 編集で投入し、レポート/表/要約図がコード改変なしに更新されることを確認（例：GW200224_222234）

# 次のロードマップ（Phase 14：GWの拡張と論文統合）

- [x] Step 14.1：profiles を拡充し、イベント個別の `extra_argv` をさらに減らす（例：BBHの「hilbert短窓」など）
- [x] Step 14.2：GWTC-3-confident から高SNR BBH を 1〜2 件追加し、event_list編集のみで online→offline 再現を確認（GW191216_213338 / GW200311_115853）
- [x] Step 14.3：論文の統合方針を固定（本文=要約＋代表図、残りは付録へ）し、図番号の固定運用を始める

# 次のロードマップ（Phase 15：GWのロバスト化）

- [x] Step 15.1：match の「窓が短すぎる」ケースを検知して注記/無効化（過大評価の回避、Table 1 と要約図にも反映）
- [x] Step 15.2：イベント一覧（event_list）に SNR/FAR などのメタ情報を追加し、要約図に併記（“強いイベントほど合う/崩れる”の傾向確認）
- Step 15.3：検出器ごとの失敗（fit_failed）を減らす既定値調整（BBHの自動窓・frange、V1/L1の扱い）

## 最優先（次にやる）：Phase 15（Step 15.3：GWの既定値調整）

目的：
- 多イベント比較で「特定の検出器だけ失敗する」ケース（fit_failed）を減らし、要約図/表の情報量と信頼性を上げる。

やること（方針）：
- `output/gw/gw_event_list_diagnostics.json` の検出器別集計を起点に、失敗が多い検出器（現状：V1/L1）とイベント種別（BBH/BNS/NSBH）を切り分ける。
- `data/gw/event_list.json` の profiles（bbh_default 等）側で、track抽出の既定値（method/track-window/track-frange/stftパラメータ）を最小限調整し、fit_failed を減らす。
- 変更後は `python -B scripts/summary/run_all.py --offline` で一括再現し、`output/gw/gw_event_list_diagnostics.json` の `fit_failed` 件数が減ることを完了条件とする。

# 参考（Phase 8：差分予測の拡張と“決定打”の探索）

参照（現在地/全体像）：
- `doc/STATUS.md`（現在地・最新数値）
- `doc/ROADMAP.md`（Phase 0-9 の全体）

参照（公開・投稿方針）：
- `doc/paper/40_publication_plan.md`

参照（論文草稿・決定版ビルド）：
- `output/summary/pmodel_paper.html`（論文HTML, publish）
- `python -B scripts/summary/paper_build.py --mode publish`（Table 1＋論文HTML＋lint）

閲覧（生成物）：
- `output/summary/pmodel_public_report.html`（一般向け統一レポート）

---

## 参考（完了）：Phase 8（差分予測の拡張と“決定打”の探索）
※Phase 8 は Step 8.1〜8.9 が完了済み（下の進捗チェック参照）。

目的：Phase 7 で整えた「判定可能な手順」を土台に、GRとP-modelをより鋭く分けられる“決定打”の候補を増やす。

- 差が出る観測量（候補）を整理し、**一次ソースで実装できるもの**を優先して追加する（同じバッチ/レポート形式へ統合）。
- EHTの必要精度ギャップ（`output/summary/decisive_falsification.*`）を埋めるルート（支配誤差と更新点）を明文化し、観測更新時に差し替えできる状態を保つ。

合意した優先順（前回セッション最後＋今回確認）：
- Step 8.2：EHT 系統（κ/散乱/形状指標の一次ソース詰め）
- Step 8.3：回転（フレームドラッグ）を理論＋検証へ追加
- Step 8.5：宇宙論（背景P）を観測量へ接続し棄却条件まで（完了）
- Step 8.7–8.9：動的P（放射）導入（完了） → 二重パルサー導出比較（完了） → GW150914を波形比較（完了）

## Phase 8：進捗チェック（ここを更新する）

- [x] Step 8.1：差分予測候補を2-3本に絞る（一次ソース＋誤差予算）
- [x] Step 8.2：EHTを“判別実験”として成立させる（κ・散乱・スピン/傾斜・形状指標）
- [x] Step 8.3：回転（フレームドラッグ）を理論＋検証へ追加
- [x] Step 8.4：二重パルサー（軌道減衰：放射の制約）を追加（観測/GR一致度＋追加放射上限）
- [x] Step 8.5：宇宙論（赤方偏移：背景P）を“観測量”へ接続し、棄却条件まで落とす
- [x] Step 8.6：重力波（GW150914：chirp位相）を追加（公開strain→f(t)抽出→四重極チャープ則）
- [x] Step 8.7：動的P（放射・波動）を Theory（2.x）へ導入（静的極限との接続と観測量定義）
- [x] Step 8.8：二重パルサー（4.10）を「導出＋観測比較」へ拡張（Pdot_b をP-model側から再導出）
- [x] Step 8.9：GW150914（4.11）を「観測量の式（波形）比較」へ拡張（位相/周波数/波形）

再現（必須）：
- `python -B scripts/summary/run_all.py --offline`（一般向けレポート＋DOCX）
- `python -B scripts/summary/paper_build.py --mode publish`（論文HTML＋DOCX）

Step 8.2（EHTを判別に近づける）チェック
- [x] 8.2.1a κ（リング/シャドウ）の“誤差予算（参考）”を反証条件パックへ反映（Kerr係数レンジ→参考:+κ）
- [x] 8.2.1b 散乱/放射モデルの κ 制約（一次ソース）を追加（Zhu 2018 の屈折散乱ゆらぎ（distortion 0.72→6.3 μas）などを固定入力へ反映し、図でスケール感を明示：`output/eht/eht_scattering_budget.png` / `output/eht/eht_refractive_scattering_limits.png`）
- [x] 8.2.2 係数（平均直径）以外の観測量（形/非円形性など）を候補に追加し、判別条件を整理（W/d, A。M87* の W/d 上限も反映）

Step 8.3（回転項）チェック
- [x] 8.3.1 回転質量の拡張方針（最小の定義/近似/適用範囲）を理論章へ追加（`doc/paper/10_manuscript.md` の 2.7）
- [x] 8.3.2 既知実験（GP-B, LAGEOS等）の一次ソースを確定し、比較図を用意（入力: `data/theory/frame_dragging_experiments.json` / 図: `output/theory/frame_dragging_experiments.png`）

Step 8.4（強重力タイミング）チェック
- （完了）二重パルサー（軌道減衰）と GW150914（chirp）は公開一次ソースに基づき統合済み。
- 次は Phase 8 完了後の次テーマを確定してから進める。

補足：
- Phase 6/7（公開準備→決定的検証パック）は完了済み。詳細は `doc/STATUS.md` / `doc/WORK_HISTORY.md` を参照。

---

# Phase 7（決定的検証パック）

`doc/ROADMAP.md` の Phase 7 を実行するための作業分解（Step）。

## Step 7.1：グローバルパラメータの凍結（β/δ）
- β（PPN γ相当）を一次データで拘束し、以後は固定
- δ（飽和）は既知の高γ観測と矛盾しない上限として固定（測定値ではなく制約）

## Step 7.2：外挿（predict）で必須セットを増やす
- 重力赤方偏移（誤差棒つき）
- 光偏向（可能なら系列）
- 既存検証（Viking/Mercury/LLR/EHT）をβ固定で再評価して統一形式へ
- 総合スコアボード（全検証）：OK/要改善/不一致で俯瞰（Table 1 の現状）

## Step 7.3：反証条件（差が出る観測量）を1〜2本に絞って強化
- 必要精度、観測手順、棄却条件を明文化

## Step 7.4：独立再現（再現スクリプト＋データ固定）
- Web取得データを `data/` に固定し、`--offline` で再現できることを確認

---

## Phase 7：進捗チェック（ここを更新する）

- [x] Step 7.1：β/δ 凍結（`output/theory/frozen_parameters.json`）
- [x] Step 7.2：外挿（predict）再評価＋総合スコアボード（`output/summary/validation_scoreboard.*`）
- [x] Step 7.3：反証条件（必要精度・棄却条件）を論文へ統合
- [x] Step 7.4：完全オフライン再現（別環境でも主要指標一致）

Step 7.3（反証条件）作業チェック
- [x] 7.3.1 差分予測を2本以内に固定（EHT/δ）
- [x] 7.3.2 必要精度を数値化し、図＋JSONを固定名で出力（例：`output/summary/decisive_falsification.(png|json)`）
- [x] 7.3.3 論文HTMLと一般向けレポートへ同じ形で統合（図＋日本語解説）
- [x] 7.3.4 棄却条件（例：|z|>3）を本文に明記

Step 7.4（完全オフライン再現）作業チェック
- [x] 7.4.1 `python -B scripts/summary/run_all.py --offline` が failures=0 で完走（必須タスク）
- [x] 7.4.2 `python -B scripts/summary/paper_build.py --mode publish` が failures=0 で完走
- [x] 7.4.3 必要キャッシュ/入力マニフェストを固定し、STATUS に“公開入口”として明記
