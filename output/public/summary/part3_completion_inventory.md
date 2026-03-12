# Part III 完成条件棚卸し（欠落のみ）

- generated_utc: 2026-03-04T11:28:54.360842+00:00
- source: `doc/paper/12_part3_quantum.md`
- sections_scanned: 7
- missing_sections: 3

## ルール（検出）

- Input: Detect **入力**： and require non-empty/non-placeholder content (publish view).
- Frozen: Detect markers: 凍結/凍結値/固定/固定値 or frozen_parameters/freeze (publish view).
- Statistic: Detect **指標**： and require non-empty/non-placeholder content (publish view).
- Reject: Detect markers: 棄却条件/棄却/reject/no-go/pass-fail (publish view).
- Output: Detect **出力**： (publish view). Fallback: accept an 'output/' reference (legacy).
- note: INTERNAL_ONLY blocks are excluded (publish mode behavior).

## 熱

- 5.5.1 BBN（He-4 25%）の最小導出：背景P熱史 → 凍結 → 質量比（doc/paper/12_part3_quantum.md:3155） missing=Statistic, Reject, Output

## 未分類

- 5.5.2 BBN の multi-isotope 監査ゲート（D/H, He-3/He-4, Li-7）（doc/paper/12_part3_quantum.md:3308） missing=Statistic, Output
- 5.5.3 P_μ による V-A 結合の必然性（壁2）（doc/paper/12_part3_quantum.md:3386） missing=Statistic, Output
