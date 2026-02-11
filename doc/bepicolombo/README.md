# BEPICOLOMBO (MORE/SPICE) REPRO (offline)
ASCII preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

# BepiColombo（水星探査）検証：準備と再現手順

目的：
- Cassini の次段として、BepiColombo（MPO/MORE：Mercury Orbiter Radio-science Experiment）を使った検証を追加する。
- まず ESA PSA（Planetary Science Archive）で一次データが公開されているかを一次ソースで確認し、再現可能に記録する。

現状：
- PSA の `bc_mpo_more` は **バンドルXMLとドキュメントは公開**されているが、`data_raw/` 等のディレクトリは（現時点では）未公開の可能性がある。
- そのため、現段階では「公開状況の確認」と「仕様書の取得」までを自動化する。

---

## ワンコマンド再現（PowerShell）

オンライン（PSAへアクセスして更新）：
- `python -B scripts/bepicolombo/more_psa_status.py`

オフライン（キャッシュを使ってレポート用PNGを再生成）：
- `python -B scripts/bepicolombo/more_psa_status.py --offline`

SPICE kernels 公開状況（オンライン、PSAへアクセスして更新）：
- `python -B scripts/bepicolombo/spice_psa_status.py`

SPICE kernels 公開状況（オフライン、キャッシュから要約を再生成）：
- `python -B scripts/bepicolombo/spice_psa_status.py --offline`

SPICE kernels 最小セット取得（オンライン、PSAへアクセスして取得/更新）：
- `python -B scripts/bepicolombo/fetch_spice_kernels_psa.py`

SPICE kernels 最小セット取得（オフライン、キャッシュが揃っている場合のみ）：
- `python -B scripts/bepicolombo/fetch_spice_kernels_psa.py --offline`

太陽会合 Shapiro 予測（SPICE幾何、オフライン可）：
- `python -B scripts/bepicolombo/bepicolombo_shapiro_predict.py --min-b-rsun 1.0`

太陽会合イベント一覧（SPICE幾何、オフライン可）：
- `python -B scripts/bepicolombo/bepicolombo_conjunction_catalog.py --min-b-rsun 1.0 --max-b-rsun 10.0`

ドキュメント索引（オフライン、lblxから生成）：
- `python -B scripts/bepicolombo/more_document_catalog.py`

コレクション取得（オンライン、data_raw 等が公開されたらメタデータを取得）：
- `python -B scripts/bepicolombo/more_fetch_collections.py`

コレクション取得（オフライン、キャッシュから要約を再生成）：
- `python -B scripts/bepicolombo/more_fetch_collections.py --offline`

公開レポート更新（オフライン可）：
- `python -B scripts/summary/public_dashboard.py`

---

## 生成物（output/bepicolombo）

- `output/bepicolombo/more_psa_status.png`（一般向け：公開状況の要約）
- `output/bepicolombo/more_psa_status.json`（機械可読）
- `output/bepicolombo/spice_psa_status.png`（一般向け：SPICE kernels 公開状況/最新inventoryの要約）
- `output/bepicolombo/spice_psa_status.json`（機械可読）
- `output/bepicolombo/bepicolombo_shapiro_geometry.png`（一般向け：太陽会合の Shapiro 予測図、SPICE幾何）
- `output/bepicolombo/bepicolombo_shapiro_geometry.csv`（機械可読：b(t)/Δt(t)/y(t)）
- `output/bepicolombo/bepicolombo_shapiro_geometry_summary.json`（機械可読：会合中心/ピーク等の要約）
- `output/bepicolombo/bepicolombo_conjunction_catalog.png`（一般向け：太陽会合イベント一覧、SPICE幾何）
- `output/bepicolombo/bepicolombo_conjunction_catalog.csv`（機械可読：イベント一覧）
- `output/bepicolombo/bepicolombo_conjunction_catalog_summary.json`（機械可読：イベント一覧の要約）
- `output/bepicolombo/more_document_catalog.json`（機械可読：ドキュメント索引）
- `output/bepicolombo/more_document_catalog.csv`（人間可読：ドキュメント索引）
- `output/bepicolombo/more_fetch_collections.json`（機械可読：data_raw 等の取得状況）

---

## 入力データ（data/bepicolombo）

- `data/bepicolombo/psa_more/`（PSAから取得した bundle/doc 等をキャッシュ）
  - `base_index.html` / `parent_index.html` / `document_index.html`（PSAのディレクトリ一覧をキャッシュ、オフライン再現用）
- `data/bepicolombo/psa_spice/`（PSAから取得した SPICE bundle/inventory 等をキャッシュ）
  - `base_index.html` / `spice_kernels_index.html`（PSAのディレクトリ一覧をキャッシュ、オフライン再現用）
  - `bundle_bc_spice_v*.xml`（bundle）
  - `collection_spice_kernels_inventory_v*.csv`（inventory）
- `data/bepicolombo/kernels/psa/`（PSAから取得した SPICE kernels をキャッシュ）
  - `kernels_meta.json`（取得したファイル一覧と選択されたカーネルのパス）
  - `lsk/naif0012.tls`（Leap seconds）
  - `spk/de432s.bsp`（惑星暦）
  - `spk/bc_mpo_fcp_*.bsp`（MPO軌道）

---

## 補足（一次ソース起点の準備メモ）

- `doc/bepicolombo/MORE_DATA_NOTES.md`

---

## 定期確認（例：週1）

一次データ（`data_raw/` 等）が公開された瞬間に取りこぼさないため、次の順で定期的に確認する：
- 1) 公開状況を更新（オンライン）：`python -B scripts/bepicolombo/more_psa_status.py`
- 2) data_raw 等が 200 になったらコレクションを取得：`python -B scripts/bepicolombo/more_fetch_collections.py`
- 3) 公開レポートを更新（オフライン可）：`python -B scripts/summary/public_dashboard.py`
