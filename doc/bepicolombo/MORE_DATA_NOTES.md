# MORE_DATA_NOTES
This file starts with an ASCII-only preamble to avoid a known Windows tool unicode slicing issue.
Japanese content begins after this preamble.

# BepiColombo（MPO/MORE）一次データ検証：準備メモ（一次ソース起点）

目的：
- Cassini（一次データ）に続いて、BepiColombo（MPO/MORE）で Shapiro（重力遅延）等の検証を追加するための準備を固める。
- 「どこに、どの形式でデータが出るのか」を一次ソース（ESA PSA）と同梱ドキュメントで確定し、再現可能にしておく。

---

## 0) 現状（2026-01-02 UTC 時点）

- 一次ソース：ESA PSA `bc_mpo_more/`
- `document/` は公開されており、ドキュメントはローカルにキャッシュ済み。
- `data_raw/` などデータ本体ディレクトリは、現時点では 404（未公開/未配置の可能性）。

参照（機械可読）：
- `output/bepicolombo/more_psa_status.json`（`generated_utc` を参照）

---

## 1) PSA バンドルが指す「本来あるはずのコレクション」

バンドルXML（`data/bepicolombo/psa_more/bundle_bc_mpo_more.xml`）には、以下が `Bundle_Member_Entry` として列挙される：
- `data_raw` / `data_calibrated`
- `browse_raw` / `browse_calibrated`
- `calibration_raw`
- `miscellaneous`
- `document`

つまり、検証に必要な観測データは `data_raw` や `data_calibrated` 側に出る設計だが、現状は公開状態が追いついていない可能性がある。

---

## 2) 同梱ドキュメント群（一次ソースの索引）

PSAの `document/*.lblx` から、タイトル/文書番号/版などをオフラインで索引化できる。

索引（生成物）：
- `output/bepicolombo/more_document_catalog.json`
- `output/bepicolombo/more_document_catalog.csv`

この索引から読み取れる重要ポイント（“データ形式の当たり”）：
- **CCSDS 503.0-B-2: Tracking Data Message（TDM）** が含まれる  
  → ドップラー/レンジ等の「追跡観測量」を交換する標準。MOREデータがTDM（XML/KVN）で配布される可能性が高い。
- **CCSDS 505.0-B-2: XML specification for Navigation Data Messages（NDM）** が含まれる  
  → “航法データメッセージ”の枠組み（パッケージ/メタ）を規定。TDM/軌道/姿勢などと合わせて運用される可能性がある。
- **CCSDS 506.1-B-1: Delta-DOR raw data exchange format** が含まれる  
  → ΔDOR（VLBI系）の原データ交換の標準。MORE検証に含まれる場合は別系統の観測量になる。
- DSN（NASA）側の **Weather / Media Calibration** 文書が含まれる  
  → 対流圏・媒体補正（大気・電離圏/プラズマ）に関わる。ns級/高精度化の系統誤差源になりやすい。

NOTE：
- 上は「ドキュメント索引から読み取れる“可能性”」であり、実際に `data_raw/` 側のコレクション（collection_*.xml 等）を取得して初めて、配布形式と内容が確定する。

---

## 3) P-model 検証で “最終的に必要になるもの”（データが公開されたら）

Cassini の再現と同等以上に “手順が固定” できるよう、BepiColombo（MORE）では最低限以下を確定する：

- 観測量：
  - ドップラー（周波数比 y(t) / range-rate）またはレンジ（往復遅延）
  - 2-way/3-way の別、送受信時刻（UTC/TT）、積算時間、バンド（X/Ka 等）
- 幾何：
  - 太陽会合（b_min）や観測窓（±日）を定義できる情報
  - SPICEカーネル or HORIZONS で再現できる形（オフラインキャッシュ含む）
- 補正：
  - プラズマ補正（周波数多重ならKa+X等）・対流圏・局座標・EOP 等
  - どの補正が “観測側で既に入っているか” を一次データ仕様から確定（CassiniのTDF疑似残差問題と同種の落とし穴）
- P-model 側パラメータ：
  - `β`（係数2を担う自由度）は Cassini と整合する値を初期値に固定し、BepiColomboで追加拘束する。

---

## 4) 再現コマンド（オフライン）

- 公開状況の確認（オンライン）：
  - `python -B scripts/bepicolombo/more_psa_status.py`
- 公開状況の再描画（オフライン）：
  - `python -B scripts/bepicolombo/more_psa_status.py --offline`
- data_raw 等のコレクション取得（オンライン、公開されたらメタデータを取得）：
  - `python -B scripts/bepicolombo/more_fetch_collections.py`
- data_raw 等のコレクション取得（オフライン）：
  - `python -B scripts/bepicolombo/more_fetch_collections.py --offline`
- ドキュメント索引の生成（オフライン）：
  - `python -B scripts/bepicolombo/more_document_catalog.py`
- 公開レポート更新（オフライン可）：
  - `python -B scripts/summary/public_dashboard.py`
