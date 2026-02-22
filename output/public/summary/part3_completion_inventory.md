# Part III 完成条件棚卸し（欠落のみ）

- generated_utc: 2026-02-20T12:39:04.959049+00:00
- source: `doc/paper/12_part3_quantum.md`
- sections_scanned: 28
- missing_sections: 0

## ルール（検出）

- Input: Detect **入力**： and require non-empty/non-placeholder content (publish view).
- Frozen: Detect markers: 凍結/凍結値/固定/固定値 or frozen_parameters/freeze (publish view).
- Statistic: Detect **指標**： and require non-empty/non-placeholder content (publish view).
- Reject: Detect markers: 棄却条件/棄却/reject/no-go/pass-fail (publish view).
- Output: Detect **出力**： (publish view). Fallback: accept an 'output/' reference (legacy).
- note: INTERNAL_ONLY blocks are excluded (publish mode behavior).

- 欠落は検出されなかった。
