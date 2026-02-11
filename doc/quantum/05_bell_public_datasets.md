# 公開ベルテストデータ（取得・再解析の対象一覧）

## 目的

Step 7.4 以降で「time-tag / trial 定義 / post-selection が統計母集団を変えるか」を一次データで検証するため、
公開データの所在と、各データでできる検証（できない検証）を一覧化する。

---

## A) 取得・解析パイプラインが存在（本リポジトリで再現できる）

### A1) NIST belltestdata（Shalm et al. 2015 / photon / loophole-free）
- 種別：time-tag 生ログ（Alice/Bob 別）
- 本リポジトリ：
  - 取得：`python -B scripts/quantum/fetch_nist_belltestdata.py`
  - 再解析（最小）：`python -B scripts/quantum/nist_belltest_time_tag_reanalysis.py`
  - 出力：`output/quantum/nist_belltest_time_tag_bias.png`
- このデータで見たい点：
  - setting と click delay（sync基準）の分布差
  - coincidence window / pairing の選択がペア集合を変える兆候
- 注意：
  - ここでの再解析は公式の p-value 再現ではない（Step 7.4 の目的は「入口の定量化」）。

### A2) Delft / Hensen et al. 2015（electron spin / event-ready / loophole-free / CHSH）
- 種別：trial/event-ready 定義が明示された表形式データ（公開 zip）
- 本リポジトリ：
  - 取得：`python -B scripts/quantum/fetch_delft_hensen2015.py`
  - 再解析：`python -B scripts/quantum/delft_hensen2015_chsh_reanalysis.py`
  - 出力：`output/quantum/delft_hensen2015_chsh.png`
- このデータで見たい点：
  - CHSH S（論文と同定義）の再現
  - event-ready window / readout window の定義変更で S や有効trial数がどう動くか（selectionの入口）

### A2b) Delft / Hensen et al. 2016（Sci Rep 6, 30289；electron spin / event-ready / CHSH）
- 種別：trial/event-ready 定義が明示された表形式データ（公開 zip；旧/新 detector の2サブセット）
- 本リポジトリ：
  - 取得：`python -B scripts/quantum/fetch_delft_hensen2016_srep30289.py`
  - 再解析：`python -B scripts/quantum/delft_hensen2015_chsh_reanalysis.py --profile hensen2016_srep30289`
  - 出力：`output/quantum/delft_hensen2016_srep30289_chsh.png`
- このデータで見たい点：
  - 旧/新 detector を統合した CHSH（psi±別＋combined）の再現
  - event-ready window start（offset）で trial数や S がどう動くか（selection感度）

### A3) Weihs et al. 1998（photon / time-tag / “coincidence-time” 論争で参照されることが多い）
- 種別：time-tag 生ログ（Alice/Bob 別 zip；*_V.dat / *_C.dat）
- 本リポジトリ：
  - 取得：`python -B scripts/quantum/fetch_weihs1998_zenodo_7185335.py`
  - 再解析（window sweep）：`python -B scripts/quantum/weihs1998_time_tag_reanalysis.py`
  - 出力：`output/quantum/weihs1998_chsh_sweep__weihs1998_longdist_longdist1.png`
  - 追加（複数subdirまとめ）：`output/quantum/weihs1998_chsh_sweep_summary__multi_subdirs.png`
- このデータで見たい点：
  - coincidence window の選択で CHSH |S| がどれだけ動くか（selectionの入口）
  - time-tag の bit encoding（setting/outcome）と、解析手続きの仮定が結果に与える影響

### A4) Christensen et al. 2013（PRL 111, 130406；Kwiat group / photon / CH / PC-triggered）
- 種別：time-tag 生ログ（channel=1(Alice),2(Bob),15(PC trigger)）＋ trial トリガ（PC の on/off）
  - 注意：channel 15 は「表の trigger」で、実際は **中点（hidden trigger）** が入る（data_organization.txt に明記）。
- 本リポジトリ：
  - 取得：`python -B scripts/quantum/fetch_kwiat2013_prl111_130406.py`
  - 統合出力（Bell 固定I/F）：`python -B scripts/quantum/bell_primary_products.py --datasets kwiat2013_prl111_130406_05082013_15`
  - 出力：`output/quantum/bell/kwiat2013_prl111_130406_05082013_15/window_sweep_metrics.json`（CH J_prob sweep）
- このデータで見たい点：
  - photon（NIST以外）でも、trial/trigger 由来の「自然な窓（window）」の取り方で統計量 J がどれだけ動くか
  - trigger からの click delay 分布が setting 依存するか（KS で入口確認）

---

## B) 候補（公開データの所在を調査中／一次ソース未固定）

### B1) Aspect（1981–1982 / photon）
- 想定：論文中の集計表はあるが、生ログ（time-tag）の公開は未確認。
- 本Stepの主目的（time-tag/post-selectionの検証）には、一次の time-tag が必要。

### B3) Giustina et al. 2015（photon / loophole-free）
- 現状：
  - 論文PDF＋supplement（arXiv source 内）は一次ソースとして固定できた。
  - ただし photon time-tag（click log）の一次公開（再解析可能な形）は未固定。
- 本リポジトリ（一次ソース固定のみ）：
  - 取得：`python -B scripts/quantum/fetch_giustina2015_prl115_250401.py`
    - （追加）`--include-aps-supplemental`（APS supplemental が取れる環境であれば best-effort で取得）
  - 保存：`data/quantum/sources/giustina2015_prl115_250401/`（`manifest.json` を同梱）
- 補足（未固定の理由の一次側の観察）：
  - arXiv の supplemental PDF（`supplemental_material_Vienna_20151220.pdf`）は、data recording/analysis（time-tag threshold 等）を記述するが、**click log（time-tag 生データ）の公開場所を明示しない**。
  - arXiv 版の Data availability には「実験データは request（対応著者へ連絡）」旨の記載があり、少なくとも現時点では公開クリックログ（time-tag一次データ）の自動取得先が見当たらない。
  - 同PDFには「space-time layout will be made available in [13]」という記述があり、[13] は “In preparation / PhD thesis” 参照である（= 本リポジトリで自動取得できる公開データの形ではない）。
  - APS supplemental から手動取得した PDF（`Supplemental_material_final.pdf`）も一次ソースとして固定したが、**PDF内に click log（time-tag 生データ）は含まれず**、また PDF 本文に生データの公開先（URL/DOI等）は見当たらない（現状のブロックは継続）。
- 関連（PhD thesis の参照先候補）：
  - Giustina 2016（University of Vienna；thesis）は Phaidra/uTheses に登録があるが、現時点の uTheses は `fulltext_locked=1` のため自動取得できない（公開状態を監視するI/Fは固定済み）。
    - 取得：`python -B scripts/quantum/fetch_giustina2016_utheses_39955.py`
    - probe：`output/quantum/bell/giustina2016_utheses_probe.json`
- 追加（公開 click log の探索ログ；初版）：
  - 取得：`python -B scripts/quantum/search_giustina2015_clicklog_public_sources.py`
  - 出力：`output/quantum/bell/giustina2015_clicklog_public_search.json`（現時点：候補なし）
- 次に必要なこと（このStepでやりたい検証をするため）：
  - time-tag（Alice/Bob の click/event log）を取得できる一次ソースを特定し、同様にキャッシュ＋manifest化する。
  - APS supplemental（link.aps.org）は Cloudflare challenge（HTTP 403 + `cf-mitigated: challenge`）になり、環境によって自動取得ができない。
    - 手動DL（ブラウザ）で取得できる場合は、次を行う：
      1. APS supplemental（`https://link.aps.org/supplemental/10.1103/PhysRevLett.115.250401`）からファイルをダウンロード
      2. `data/quantum/sources/giustina2015_prl115_250401/aps_supplemental/` に（ファイル名そのままで）配置
      3. `python -B scripts/quantum/fetch_giustina2015_prl115_250401.py --offline` を実行して `manifest.json` に追記（hash固定）
    - その上で、Weihs/NIST と同様に window sweep / trial-based 指標へ統合する（解析スクリプトはデータ形式を見て確定する）。

### B4) その他（Big Bell Test, Christensen 2013 など）
- Big Bell Test Collaboration 2018（Nature；human-choice inputs）
  - 種別：Source Data（xlsx; Fig.2 の「人間入力」側の一次プロダクト）。Bell の time-tag（Alice/Bob の click log）ではない。
  - 本リポジトリ：
    - 取得：`python -B scripts/quantum/fetch_big_bell_test_2018.py`
    - 要約（Fig.2a sessions by country）：`python -B scripts/quantum/big_bell_test_2018_sessions_by_country.py`
    - 出力：`output/quantum/big_bell_test_2018_sessions_by_country.png`
- 原則：一次データ（time-tag / trial log）が公開されていること、再解析が再現可能であることを満たすものから統合する。

---

## 次に追加したいデータ（優先順）
1. B系（photon）の time-tag 一次データ（NIST以外）
2. trial-based（event-ready）データ（Delft以外）
