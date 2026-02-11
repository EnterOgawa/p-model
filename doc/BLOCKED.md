# ブロック中/保留テーマ（ロードマップ外）

目的：
- 「今は一次データが手に入らず進められないが、公開された瞬間に検証へ合流できる」テーマをまとめる。
- `doc/ROADMAP.md` の必須経路をスリムに保ちつつ、準備状況を失わない。

運用：
- このファイルは “進捗の外” に置く（`doc/STATUS.md` の必須チェックリストには入れない）。
- 公開/配置が整ったら、ロードマップ（Phase 3）に戻す。
- `scripts/summary/run_all.py` はデフォルトではブロック中テーマを実行しない（必要なら `--include-blocked`）。

---

## BepiColombo（MORE）：一次データ未公開のため保留

状況：
- MORE（range/Doppler）一次データが PSA 側で未公開/未配置の可能性があり、**観測 vs P-model** の検証ができない。
- そのため一般向けレポート（`output/summary/pmodel_public_report.html`）には **掲載しない**。
  - 例外：環境変数 `WAVEP_INCLUDE_BEPICOLOMBO=1` を付けてレポート生成した場合のみ表示（内部確認用）。

方針：
- 公開された瞬間に Cassini と同形式（観測 vs P-model）へ移行できるよう、取得準備・幾何予測（SPICE）・公開状況監視は維持する。

再開の入口（まず見る）：
- PSA公開状況：`output/bepicolombo/more_psa_status.json`
  - `data_raw/` 等が 200 になったら再開（404/未配置が解消したらGO）。

関連スクリプト：
- 取得準備：`scripts/bepicolombo/more_fetch_collections.py`
- 幾何ベース予測（SPICE）：`scripts/bepicolombo/bepicolombo_shapiro_predict.py`

現在の生成物（注：予測であり観測ではない）：
- `output/bepicolombo/bepicolombo_shapiro_geometry.png`
- `output/bepicolombo/bepicolombo_shapiro_geometry_summary.json`
- `output/bepicolombo/bepicolombo_conjunction_catalog.png`
- `output/bepicolombo/bepicolombo_conjunction_catalog_summary.json`
