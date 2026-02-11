# STATUS

最終更新（UTC）：2026-02-11T05:09:58Z

## 現在地（ロードマップ）
- Phase 7 / Step 7.20.1（Bell：ヌルテスト一式の固定）完了（初版）。
- Phase 7 / Step 7.20.2（Bell：ブラインド凍結ポリシー固定）完了（初版）。
- Phase 7 / Step 7.20.3（Bell：独立推定器クロスチェック）完了（改訂；Weihs+NIST+Kwiat supported）。
- Phase 7 / Step 7.20.7（Part III：監査ハーネス）完了（初版）。
- Phase 7 / Step 7.20.4（核：holdout監査）完了（初版）。
- Phase 7 / Step 7.20.6（物性/熱：温度帯holdout監査）完了（改訂；α(T)+Cp(T)+κ(T)+B(T) を横断し、`max|z|≤3`（補助：`reduced χ²`）の運用ゲートで holdout-ok を判定できる形に統一）。
- Phase 7 / Step 7.20.8（Part III publish：可読性改善）完了（改訂；凝縮系の長い ansatz ログを補遺Aへ退避）。
- Phase 8 / Step 8.2.11（Part I：本文のファイル名（.json）除去）完了。
- Phase 8 / Step 8.2.12（Part I：量子2.8の最小辞書化）完了。
- Phase 7 / Step 7.14.4（Si 抵抗率 ρ(T) 抽出）改訂：PDF 抽出の数値パース（`1. 910-01` 等の分割）を修正し、抽出の欠損/誤値を解消。
- Phase 7 / Step 7.16.8（Bell：15項目系統誤差分解）改訂：Part III 4.2.1 に (i)「sweep幅（選択自由度）と freeze 近傍 drift」を区別する説明（例：Weihs 自然窓で `Δ|S|≈0.0013`）と、(ii) selection項目を除いた「装置/補正」側の合成寄与（L2/L1）表を追記。
- Phase 8 / Step 8.2.13（設計方針変更：検証用資料へ集約）完了（Part I–III 本文から具体的な file/folder/script 表記を排除し、検証用資料を参照する形へ移行。paper_qc に publish HTML の repo-path 混入検出を追加）。
- Phase 8 / Step 8.2.14（公開参照の固定：Verification Materials のURL/DOI確定）完了（GitHub repo 作成＋References/Verification Materials の `TBD` を固定URLへ置換）。
- Phase 8 / Step 8.2.15（GitHub運用方針改訂：検証中心化）完了（READMEをストーリー化、論文は別公開、固定出力JSONを追跡対象へ）。
- 注記：論文等のドキュメントは別公開とし、本repoでは追跡しない（`doc/paper/` は git 管理外）。
- 次：（任意）Phase 7 / Step 7.20.6 の追加拡張（物性/熱 holdout 対象の追加：bulk modulus / ρ(T) 係数など）。
- 保留（公開待ち）：Phase 4 / Step 4.6（GN-z11）、Phase 7 / Step 7.4.7（Giustina click log/thesis）、Phase 4 / Step 4.13（XRISM）。

## 最新結果（最小）
- Bell ヌルテスト（設定シャッフル／時間シフト）を固定出力し、pack に同梱（version=1.4）：
  - `output/quantum/bell/null_tests_summary.json`
  - `output/quantum/bell/<dataset>/null_tests.json`
  - `output/quantum/bell/falsification_pack.json`（`cross_dataset.null_tests_summary`）
- Bell ブラインド凍結ポリシー（window/offset）を JSON で固定し、pack に同梱（version=1.4）：
  - `output/quantum/bell/freeze_policy.json`
  - `output/quantum/bell/falsification_pack.json`（`policy.blind_freeze` / `cross_dataset.inputs.freeze_policy_json`）
- Bell pairing 実装差クロスチェック（初版）を固定出力し、pack に同梱（version=1.4）：
  - `output/quantum/bell/<dataset>/crosscheck_pairing.json`
  - `output/quantum/bell/crosscheck_pairing_summary.json`（Weihs+NIST+Kwiat supported；Δ_impl/σ_boot を固定）
  - `output/quantum/bell/falsification_pack.json`（`cross_dataset.pairing_crosscheck_summary`）
- 再現：`python -B scripts/quantum/bell_primary_products.py`
- Part III 凍結パラメータ一元管理（量子）を更新：
  - `python -B scripts/quantum/frozen_parameters_quantum.py` → `output/quantum/frozen_parameters_quantum.json`（Bell pack の新sha256＋freeze_policy参照を反映）
- Part III 監査ハーネス（pack生成→ゲート判定→要約）を 1 コマンド化：
  - `python -B scripts/summary/part3_audit.py` → `output/summary/part3_audit_summary.json`（ok=True）
- 核 holdout 監査（AME2020 全核種の残差を領域分割して固定）：
  - `python -B scripts/quantum/nuclear_holdout_audit.py` →
    `output/quantum/nuclear_holdout_audit_summary.json` / `..._groups.csv` / `..._outliers.csv` / `...png`
- 物性/熱 holdout 監査（温度帯 split の感度を sys として固定）：
  - `python -B scripts/quantum/condensed_silicon_thermal_expansion_gruneisen_holdout_splits.py`
  - `python -B scripts/quantum/condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_model.py`（DOS+γ(ω)基底のholdout-ok例）
  - `python -B scripts/quantum/condensed_silicon_heat_capacity_holdout_splits.py`（Debye low-T + WebBook Shomate の “hybrid frozen” を追加し、holdout-ok例を固定）
  - `python -B scripts/quantum/condensed_silicon_bulk_modulus_holdout_splits.py`（Ioffe由来 B(T) の温度帯 holdout を追加）
  - `python -B scripts/quantum/condensed_ofhc_copper_thermal_conductivity_holdout_splits.py`
  - `python -B scripts/quantum/condensed_holdout_audit.py` → `output/quantum/condensed_holdout_audit_summary.json`（CSV/PNG；minimax推奨モデル＋audit_gates で strict/holdout を機械判定）

## 保留項目（データ未公開/未取得）
- Phase 4 / Step 4.6（JWST/MAST；GN-z11 公開待ち；t_obs_release_utc=2026-04-09T20:45:23.996448+00:00）
- Phase 7 / Step 7.4.7（Bell追加一次データ：Giustina 2015 click log / Giustina 2016 thesis）
- Phase 4 / Step 4.13（XRISM 追加公開obsid）

## 次にやること
- 優先度0：初回 GitHub Release（tag）を作成し、release bundle（paper/repro）を添付して、論文側の参照先を「version固定（Releases/DOI）」へ更新する（必要なら Zenodo DOI も付与）。
- 優先度1（任意）：物性/熱：holdout の対象を追加（bulk modulus / ρ(T) 係数など可能な範囲）。
- 優先度2（任意）：物性/熱：cross-check（Cp(T)・α(T)・B(T) など）を保ったまま、一次拘束を増やした最小 ansatz を探索し、監査出力へ固定する（補遺に詳細ログ）。
- 優先度3（任意）：Bell：cross-dataset 統合を維持したまま、dataset追加（公開時）と bootstrap/jackknife の安定性を上げる（公開待ちでない範囲）。
- （必要時）Part III publish を再生成し QA：`python -B scripts/summary/paper_build.py --profile part3_quantum --mode publish --outdir output/summary` → `paper_qc` / `paper_lint`。
